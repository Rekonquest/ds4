// DS4 (DwarfStar) â€” BPE tokenizer.
//
// Ports the public surface of `ds4_tokenize`,
// `ds4_tokenize_rendered_chat`, and the vocab struct from `ds4.c:21870..`
// (see `ds4.h` for the abstract API). The DeepSeek pre-tokenize rules
// (digit groups, CJK / kana, letter runs, punctuation) live in
// `tokenizer_data::pre_tokenize`.
//
// Tokenizers loaded from GGUF metadata use the model's token list and
// merge ranks. The byte-level fallback remains available for tests and
// for incomplete model metadata.

use std::collections::HashMap;
use std::sync::Arc;

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

use crate::gguf::GgufFile;
use crate::tokenizer_data as td;

// ---------------------------------------------------------------------------
// Vocab (BPE merge table + special tokens).
// ---------------------------------------------------------------------------

/// BPE vocab: token-string -> id, plus the list of tokens in rank
/// order. The merge table itself is a `HashMap<(left, right), rank>`
/// keyed by token-id pairs.
#[derive(Debug, Clone)]
pub struct BpeVocab {
    pub token_to_id: HashMap<Vec<u8>, u32>,
    pub id_to_token: Vec<Vec<u8>>,
    pub merges: HashMap<(u32, u32), u32>,
    /// Reserved ID for tokens not in the vocab.
    pub unk_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub user_id: u32,
    pub assistant_id: u32,
    pub think_start_id: u32,
    pub think_end_id: u32,
    pub dsml_id: u32,
    pub byte_to_id: Option<[u32; 256]>,
}

impl BpeVocab {
    /// Build a vocab from raw fields. The `id_to_token` array is
    /// indexed by token id; `token_to_id` is the inverse map for fast
    /// lookup of bytes->id.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: Vec<Vec<u8>>,
        merges: HashMap<(u32, u32), u32>,
        unk_id: u32,
        bos_id: u32,
        eos_id: u32,
        user_id: u32,
        assistant_id: u32,
        think_start_id: u32,
        think_end_id: u32,
        dsml_id: u32,
    ) -> Ds4Result<Self> {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }
        Ok(BpeVocab {
            token_to_id,
            id_to_token: tokens,
            merges,
            unk_id,
            bos_id,
            eos_id,
            user_id,
            assistant_id,
            think_start_id,
            think_end_id,
            dsml_id,
            byte_to_id: None,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.id_to_token.len() as u32
    }
}

// ---------------------------------------------------------------------------
// Public tokenizer.
// ---------------------------------------------------------------------------

/// High-level tokenizer. Wraps a `BpeVocab` and runs the pre-tokenizer
/// followed by model BPE merge ranking when metadata is available.
/// Designed to be cloned cheaply because the inner vocab is wrapped in
/// an `Arc` so multiple sessions can share it.
#[derive(Debug, Clone)]
pub struct Ds4Tokenizer {
    vocab: Arc<BpeVocab>,
}

impl Ds4Tokenizer {
    /// Build a tokenizer from a vocab.
    pub fn from_vocab(vocab: BpeVocab) -> Ds4Result<Self> {
        Ok(Ds4Tokenizer {
            vocab: Arc::new(vocab),
        })
    }

    /// Convenience builder: build a v1 byte-level tokenizer that maps
    /// each byte b in `0..=255` to the id `byte_to_id[b]` and the
    /// BPE-unk to `unk_id`. This lets `tokenize` round-trip on simple
    /// ASCII inputs without a full merge table.
    #[allow(clippy::too_many_arguments)]
    pub fn from_byte_mapping(
        byte_to_id: [u32; 256],
        unk_id: u32,
        bos_id: u32,
        eos_id: u32,
        user_id: u32,
        assistant_id: u32,
        think_start_id: u32,
        think_end_id: u32,
        dsml_id: u32,
    ) -> Ds4Result<Self> {
        let mut tokens = Vec::with_capacity(byte_to_id.len());
        for b in 0u32..256 {
            let b_byte = b as u8;
            // We don't actually need the bytes to enumerate in order;
            // the tokenizer reads them back by index via byte_to_id.
            tokens.push(vec![b_byte]);
        }
        // We rebuild a token_to_id where the "token bytes" we store are
        // the byte itself (one-byte vector). The byte-level tokenizer
        // never looks these up directly â€” it goes via byte_to_id.
        let mut token_to_id = HashMap::with_capacity(256);
        for b in 0u32..256 {
            token_to_id.insert(vec![b as u8], b);
        }

        Ok(Ds4Tokenizer {
            vocab: Arc::new(BpeVocab {
                token_to_id,
                id_to_token: tokens,
                merges: HashMap::new(),
                unk_id,
                bos_id,
                eos_id,
                user_id,
                assistant_id,
                think_start_id,
                think_end_id,
                dsml_id,
                byte_to_id: Some(byte_to_id),
            }),
        })
    }

    /// Build a tokenizer from GGUF tokenizer metadata. Returns
    /// `Ok(None)` when the file does not carry `tokenizer.ggml.tokens`.
    pub fn from_gguf(gguf: &GgufFile) -> Ds4Result<Option<Self>> {
        let Some(raw_tokens) = gguf
            .kv_raw("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array_str())
        else {
            return Ok(None);
        };
        if raw_tokens.is_empty() {
            return Ok(None);
        }

        let tokens: Vec<Vec<u8>> = raw_tokens
            .iter()
            .map(|s| token_string_to_bytes(s))
            .collect();
        let merge_strings = gguf
            .kv_raw("tokenizer.ggml.merges")
            .and_then(|v| v.as_array_str())
            .unwrap_or_default();
        let merges = build_merge_ranks(&tokens, &merge_strings);
        let meta = &gguf.metadata;
        let unk_id = gguf_u32(gguf, "tokenizer.ggml.unk_token_id").unwrap_or(0);
        let bos_id = meta.bos_token_id.unwrap_or(unk_id);
        let eos_id = meta.eos_token_id.unwrap_or(unk_id);
        let user_id = meta.user_token_id.unwrap_or(unk_id);
        let assistant_id = meta.assistant_token_id.unwrap_or(unk_id);
        let think_start_id = meta.think_start_token_id.unwrap_or(unk_id);
        let think_end_id = meta.think_end_token_id.unwrap_or(unk_id);
        let dsml_id = meta.dsml_token_id.unwrap_or(unk_id);

        let vocab = BpeVocab::new(
            tokens,
            merges,
            unk_id,
            bos_id,
            eos_id,
            user_id,
            assistant_id,
            think_start_id,
            think_end_id,
            dsml_id,
        )?;
        Self::from_vocab(vocab).map(Some)
    }
    pub fn vocab_size(&self) -> u32 {
        self.vocab.vocab_size()
    }
    pub fn unk_id(&self) -> u32 {
        self.vocab.unk_id
    }
    pub fn bos_id(&self) -> u32 {
        self.vocab.bos_id
    }
    pub fn eos_id(&self) -> u32 {
        self.vocab.eos_id
    }

    /// Tokenize a string with the standard JoyAI/DeepSeek pre-tokenizer.
    pub fn tokenize(&self, s: &str) -> Ds4Result<Vec<u32>> {
        let mut out = Vec::with_capacity(s.len() + 16);
        // First try to match a special sentinel at the current
        // position. The BPE word loop in `ds4.c:22200..22220` does the
        // same: specials come first, then pieces.
        let mut consumed = 0usize;
        while consumed < s.len() {
            let rest = &s[consumed..];
            if let Some((tok_id, n)) = td::match_special_at(
                rest,
                self.vocab.bos_id,
                self.vocab.eos_id,
                self.vocab.user_id,
                self.vocab.assistant_id,
                self.vocab.think_start_id,
                self.vocab.think_end_id,
                self.vocab.dsml_id,
            ) {
                out.push(tok_id);
                consumed += n;
                continue;
            }
            // Find the longest char-aligned run starting at `consumed`
            // that doesn't begin with a special sentinel; split it
            // via pre_tokenize.
            let mut end = consumed;
            while end < s.len() {
                if td::match_special_at(
                    &s[end..],
                    self.vocab.bos_id,
                    self.vocab.eos_id,
                    self.vocab.user_id,
                    self.vocab.assistant_id,
                    self.vocab.think_start_id,
                    self.vocab.think_end_id,
                    self.vocab.dsml_id,
                )
                .is_some()
                {
                    break;
                }
                let Some(ch) = s[end..].chars().next() else {
                    break;
                };
                end += ch.len_utf8();
            }
            if end == consumed {
                // No char-progress would be made; force-advance one
                // character and emit UNK to avoid an infinite loop.
                out.push(self.vocab.unk_id);
                consumed += s[consumed..]
                    .chars()
                    .next()
                    .map(char::len_utf8)
                    .unwrap_or(1);
                continue;
            }
            let run = &s[consumed..end];
            for piece in td::pre_tokenize(run) {
                self.tokenize_piece(piece, &mut out);
            }
            consumed = end;
        }
        Ok(out)
    }

    /// Tokenize the RAW string WITHOUT applying the chat template â€”
    /// i.e. treat the input as already-rendered text and just BPE it.
    /// Mirrors `ds4.c:22220`.
    pub fn tokenize_rendered_chat(&self, s: &str) -> Ds4Result<Vec<u32>> {
        self.tokenize(s)
    }

    /// Decode token ids through the active vocabulary.
    pub fn detokenize(&self, tokens: &[u32]) -> Ds4Result<String> {
        let mut bytes = Vec::new();
        for &token in tokens {
            if let Some(byte_to_id) = &self.vocab.byte_to_id {
                let Some(byte) = byte_to_id.iter().position(|&id| id == token) else {
                    return Err(Ds4Error::new(
                        Ds4ErrorKind::Tokenizer,
                        format!("token id {token} is not in the byte fallback vocabulary"),
                    ));
                };
                bytes.push(byte as u8);
                continue;
            }

            let Some(piece) = self.vocab.id_to_token.get(token as usize) else {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::Tokenizer,
                    format!(
                        "token id {token} is outside vocabulary size {}",
                        self.vocab_size()
                    ),
                ));
            };
            bytes.extend_from_slice(piece);
        }
        String::from_utf8(bytes).map_err(|e| {
            Ds4Error::new(
                Ds4ErrorKind::Tokenizer,
                format!("decoded token bytes are not valid UTF-8: {e}"),
            )
        })
    }

    // -----------------------------------------------------------------------
    // Internal: turn one pre-tokenize piece into a vector of token ids.
    // -----------------------------------------------------------------------
    fn tokenize_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if self.vocab.merges.is_empty() {
            if self.vocab.byte_to_id.is_none() {
                if let Some(id) = self.vocab.token_to_id.get(piece.as_bytes()) {
                    out.push(*id);
                    return;
                }
            }
            self.tokenize_piece_without_merges(piece, out);
            return;
        }

        let mut symbols: Vec<(u32, Vec<u8>)> = Vec::new();
        for &b in piece.as_bytes() {
            let byte = vec![b];
            let id = if let Some(byte_to_id) = &self.vocab.byte_to_id {
                byte_to_id[b as usize]
            } else {
                self.vocab
                    .token_to_id
                    .get(&byte)
                    .copied()
                    .unwrap_or(self.vocab.unk_id)
            };
            let bytes = self
                .vocab
                .id_to_token
                .get(id as usize)
                .cloned()
                .unwrap_or(byte);
            symbols.push((id, bytes));
        }

        loop {
            let mut best: Option<(usize, u32, Vec<u8>, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].0, symbols[i + 1].0);
                let Some(rank) = self.vocab.merges.get(&pair).copied() else {
                    continue;
                };
                let mut merged = symbols[i].1.clone();
                merged.extend_from_slice(&symbols[i + 1].1);
                let Some(id) = self.vocab.token_to_id.get(&merged).copied() else {
                    continue;
                };
                let replace = best.as_ref().map(|(_, _, _, r)| rank < *r).unwrap_or(true);
                if replace {
                    best = Some((i, id, merged, rank));
                }
            }
            let Some((idx, id, bytes, _rank)) = best else {
                break;
            };
            symbols[idx] = (id, bytes);
            symbols.remove(idx + 1);
        }

        out.extend(symbols.into_iter().map(|(id, _)| id));
    }

    fn tokenize_piece_without_merges(&self, piece: &str, out: &mut Vec<u32>) {
        for &b in piece.as_bytes() {
            if let Some(byte_to_id) = &self.vocab.byte_to_id {
                out.push(byte_to_id[b as usize]);
            } else {
                out.push(
                    self.vocab
                        .token_to_id
                        .get(&vec![b])
                        .copied()
                        .unwrap_or(self.vocab.unk_id),
                );
            }
        }
    }
}

fn gguf_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    gguf.kv_raw(key).and_then(|v| v.as_u32())
}

fn token_string_to_bytes(token: &str) -> Vec<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
            return vec![byte];
        }
    }
    token.as_bytes().to_vec()
}

fn build_merge_ranks(tokens: &[Vec<u8>], merge_strings: &[&str]) -> HashMap<(u32, u32), u32> {
    let token_to_id: HashMap<Vec<u8>, u32> = tokens
        .iter()
        .enumerate()
        .map(|(idx, tok)| (tok.clone(), idx as u32))
        .collect();
    let mut ranks = HashMap::with_capacity(merge_strings.len());
    for (rank, merge) in merge_strings.iter().enumerate() {
        let Some((left, right)) = merge.split_once(' ') else {
            continue;
        };
        let Some(left_id) = token_to_id.get(left.as_bytes()).copied() else {
            continue;
        };
        let Some(right_id) = token_to_id.get(right.as_bytes()).copied() else {
            continue;
        };
        ranks.insert((left_id, right_id), rank as u32);
    }
    ranks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ascii_tokenizer() -> Ds4Tokenizer {
        // Map byte b -> b + 1 so ids 1..=256 are ASCII bytes (with
        // a sentinel "0" left for UNK).
        let mut map = [0u32; 256];
        for b in 0u32..256 {
            map[b as usize] = b + 1;
        }
        Ds4Tokenizer::from_byte_mapping(
            map, /* unk  */ 0, /* bos  */ 1000, /* eos  */ 1001,
            /* user */ 1002, /* asst */ 1003, /* th_s */ 1004, /* th_e */ 1005,
            /* dsml */ 1006,
        )
        .expect("tokenizer")
    }

    #[test]
    fn tokenize_pieces_split_digits() {
        let t = ascii_tokenizer();
        let toks = t.tokenize("1234").expect("tokenize");
        // Each digit byte becomes its ASCII value + 1: '1' = 0x31 -> 50.
        assert_eq!(toks, vec![50u32, 51, 52, 53]);
    }

    #[test]
    fn round_trip_ascii() {
        let t = ascii_tokenizer();
        let s = "hello world";
        let toks = t.tokenize(s).expect("tokenize");
        // 'h' is 0x68 â†’ 105; 'o' is 0x6f â†’ 112; ' ' â†’ 33; 'w' â†’ 120.
        assert_eq!(toks[0], b'h' as u32 + 1);
        assert_eq!(toks[4], b'o' as u32 + 1);
        assert_eq!(toks[5], b' ' as u32 + 1);
        assert_eq!(toks[6], b'w' as u32 + 1);
        assert_eq!(t.detokenize(&toks).expect("detokenize"), s);
    }

    #[test]
    fn bpe_merge_uses_lowest_rank_pair() {
        let mut merges = HashMap::new();
        merges.insert((0, 1), 0);
        let vocab = BpeVocab::new(
            vec![b"h".to_vec(), b"i".to_vec(), b"hi".to_vec()],
            merges,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        .expect("vocab");
        let tokenizer = Ds4Tokenizer::from_vocab(vocab).expect("tokenizer");
        assert_eq!(tokenizer.tokenize("hi").expect("tokenize"), vec![2]);
        assert_eq!(tokenizer.detokenize(&[2]).expect("detokenize"), "hi");
    }

    #[test]
    fn detokenize_rejects_unknown_token_id() {
        let t = ascii_tokenizer();
        let err = t.detokenize(&[300]).err().unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::Tokenizer);
    }

    #[test]
    fn token_string_hex_byte_decodes_to_raw_byte() {
        assert_eq!(token_string_to_bytes("<0x0A>"), vec![b'\n']);
        assert_eq!(token_string_to_bytes("<0xGG>"), b"<0xGG>".to_vec());
    }

    #[test]
    fn special_sentinel_emits_id() {
        let t = ascii_tokenizer();
        let toks = t.tokenize("<\u{ff5c}User\u{ff5c}>hi").expect("tokenize");
        assert_eq!(toks[0], 1002);
        assert_eq!(toks[1], b'h' as u32 + 1);
    }

    #[test]
    fn empty_input_yields_empty() {
        let t = ascii_tokenizer();
        let toks = t.tokenize("").expect("tokenize");
        assert!(toks.is_empty());
    }
}
