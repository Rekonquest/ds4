// DS4 (DwarfStar) — DeepSeek pre-tokenize helpers.
//
// Mirrors the JoyAI BPE pre-tokenizer used by DeepSeek V4 Flash / Pro.
// The split shape matters: different pieces lead to different BPE
// merges even when the final text bytes are identical. See
// `ds4.c:22133..22150` for the upstream docstring and
// `ds4.c:22153..22220` for the per-piece loop.
//
// The piece-segmentation rules below match the upstream C exactly:
//
//   * `\p{N}{1,3}`            — up to 3 ASCII digits in one piece.
//   * `[CJK/Hiragana/Katakana]+` — one piece per run of CJK / kana.
//   * `[P/S][A-Za-z]+`        — punctuation-then-letters (e.g. "_var").
//   * `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` — letters (Latin or non-ASCII).
//   * `?[\p{P}\p{S}]+[\r\n]*` — punctuation run with optional trailing
//                                newlines.
//   * `\s*[\r\n]+`            — whitespace-then-newline(s).
//   * `\s+(?!\S)`             — trailing whitespace.
//   * `\s+`                   — generic whitespace.
//
// `pre_tokenize` emits raw BPE pieces as `&str` slices. The tokenizer
// layer applies model-provided merge ranking when GGUF metadata is
// available, and otherwise byte-encodes each piece for deterministic
// fallback behavior.

// ---------------------------------------------------------------------------
// Character classes (mirror ds4.c).
// ---------------------------------------------------------------------------

#[inline]
pub fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_lowercase()
}

#[inline]
pub fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

#[inline]
pub fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c')
}

#[inline]
pub fn ascii_newline(c: u8) -> bool {
    matches!(c, b'\n' | b'\r')
}

#[inline]
pub fn ascii_punct_symbol(c: u8) -> bool {
    (b'!'..=b'/').contains(&c)
        || (b':'..=b'@').contains(&c)
        || (b'['..=b'`').contains(&c)
        || (b'{'..=b'~').contains(&c)
}

#[inline]
fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0xc0 {
        1
    } else if c < 0xe0 {
        2
    } else if c < 0xf0 {
        3
    } else {
        4
    }
}

#[inline]
fn next_utf8_char(s: &[u8], pos: usize) -> usize {
    pos + utf8_len_from_first_byte(s[pos])
}

#[inline]
fn utf8_peek_one(s: &[u8], pos: usize) -> Option<(u32, usize)> {
    let c0 = s[pos];
    let n = utf8_len_from_first_byte(c0);
    if pos + n > s.len() {
        return None;
    }
    let cp = match n {
        1 => u32::from(c0),
        2 => (u32::from(c0 & 0x1f) << 6) | u32::from(s[pos + 1] & 0x3f),
        3 => {
            (u32::from(c0 & 0x0f) << 12)
                | (u32::from(s[pos + 1] & 0x3f) << 6)
                | u32::from(s[pos + 2] & 0x3f)
        }
        _ => {
            (u32::from(c0 & 0x07) << 18)
                | (u32::from(s[pos + 1] & 0x3f) << 12)
                | (u32::from(s[pos + 2] & 0x3f) << 6)
                | u32::from(s[pos + 3] & 0x3f)
        }
    };
    Some((cp, pos + n))
}

#[inline]
pub fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

#[inline]
fn is_cjk_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 {
        return false;
    }
    matches!(utf8_peek_one(s, pos), Some((cp, _)) if utf8_is_cjk_hira_kata(cp))
}

/// Treat any non-ASCII byte as "letter-like" — JoyAI's regex collapses
/// the unicode-letter alphabet before the pre-tokenizer, so Italian
/// accents and similar non-ASCII letters stay attached to the word
/// they sit in.
#[inline]
fn letter_like_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 {
        return ascii_alpha(s[pos]);
    }
    true
}

fn consume_letters(s: &[u8], mut pos: usize) -> usize {
    while pos < s.len() && letter_like_at(s, pos) {
        pos = next_utf8_char(s, pos);
    }
    pos
}

// ---------------------------------------------------------------------------
// Public pre-tokenizer.
// ---------------------------------------------------------------------------

/// Split `text` into pre-tokenize pieces (BPE "words"). The order of
/// pieces is the order the BPE merger will see them.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let s = text.as_bytes();
    let mut pieces: Vec<&str> = Vec::new();
    let mut pos = 0usize;

    while pos < s.len() {
        let start = pos;
        let c = s[pos];

        if ascii_digit(c) {
            let mut ndigits = 0;
            while pos < s.len() && ascii_digit(s[pos]) && ndigits < 3 {
                pos += 1;
                ndigits += 1;
            }
        } else if is_cjk_at(s, pos) {
            while pos < s.len() && is_cjk_at(s, pos) {
                pos = next_utf8_char(s, pos);
            }
        } else if ascii_punct_symbol(c) && pos + 1 < s.len() && ascii_alpha(s[pos + 1]) {
            // [P/S][A-Za-z]+  — punctuation-then-letters (e.g. "_var").
            pos += 1;
            while pos < s.len() && ascii_alpha(s[pos]) {
                pos += 1;
            }
        } else if letter_like_at(s, pos) {
            pos = consume_letters(s, pos);
        } else if !ascii_newline(c)
            && !ascii_punct_symbol(c)
            && pos + 1 < s.len()
            && letter_like_at(s, pos + 1)
        {
            // Non-letter, non-newline, non-punctuation immediately
            // before a letter run — e.g. digits glued to letters via
            // a separator. The C code joins one leading non-letter
            // into the word.
            pos += 1;
            pos = consume_letters(s, pos);
        } else if c == b' ' && pos + 1 < s.len() && ascii_punct_symbol(s[pos + 1]) {
            // Space then punctuation — let one space join the punct run.
            pos += 1;
            while pos < s.len() && ascii_punct_symbol(s[pos]) {
                pos += 1;
            }
            while pos < s.len() && ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if ascii_punct_symbol(c) {
            while pos < s.len() && ascii_punct_symbol(s[pos]) {
                pos += 1;
            }
            while pos < s.len() && ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            // Walk through whitespace; pick the longest trailing
            // newline run.
            let mut p = pos;
            let mut last_newline_end = pos;
            while p < s.len() && ascii_space(s[p]) {
                if ascii_newline(s[p]) {
                    last_newline_end = p + 1;
                }
                p += 1;
            }
            if last_newline_end > pos {
                pos = last_newline_end;
            } else if p < s.len()
                && p > pos + 1
                && (letter_like_at(s, p) || ascii_punct_symbol(s[p]))
            {
                // JoyAI lets one leading space join the following
                // word or punctuation run.
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(s, pos);
        }

        // Safety net: if none of the branches advanced the cursor,
        // step one byte so the loop terminates on pathological input.
        if pos == start {
            pos = next_utf8_char(s, pos);
        }

        // Map the byte slice back to a `&str` so callers can iterate
        // without re-encoding.
        pieces.push(&text[start..pos]);
    }
    pieces
}

/// Look up the special-token literal at the start of `text`. Returns
/// `(token_id, byte_length)` if `text` starts with one of the
/// DeepSeek chat sentinels. Mirrors `special_token_at` in `ds4.c`.
#[allow(clippy::too_many_arguments)]
pub fn match_special_at(
    text: &str,
    bos: u32,
    eos: u32,
    user: u32,
    assistant: u32,
    think_start: u32,
    think_end: u32,
    dsml: u32,
) -> Option<(u32, usize)> {
    // The lookup is a flat if/else over the known literals with the
    // runtime ids supplied by the caller.
    if text.starts_with("<｜begin▁of▁sentence｜>") {
        return Some((bos, "<｜begin▁of▁sentence｜>".len()));
    }
    if text.starts_with("<｜end▁of▁sentence｜>") {
        return Some((eos, "<｜end▁of▁sentence｜>".len()));
    }
    if text.starts_with("<｜User｜>") {
        return Some((user, "<｜User｜>".len()));
    }
    if text.starts_with("<｜Assistant｜>") {
        return Some((assistant, "<｜Assistant｜>".len()));
    }
    if text.starts_with("<think>") {
        return Some((think_start, "<think>".len()));
    }
    if text.starts_with("</think>") {
        return Some((think_end, "</think>".len()));
    }
    if text.starts_with("｜DSML｜") {
        return Some((dsml, "｜DSML｜".len()));
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_groups_split_into_three_or_less() {
        let pieces = pre_tokenize("1234567");
        assert_eq!(pieces, vec!["123", "456", "7"]);
    }

    #[test]
    fn letter_runs_stay_together() {
        let pieces = pre_tokenize("hello world");
        assert_eq!(pieces, vec!["hello", " world"]);
    }

    #[test]
    fn punctuation_joins_following_letters() {
        let pieces = pre_tokenize("_var x");
        assert_eq!(pieces[0], "_var");
    }

    #[test]
    fn trailing_newlines_attach_to_punct_run() {
        let pieces = pre_tokenize(";\nfoo");
        assert_eq!(pieces[0], ";\n");
    }

    #[test]
    fn whitespace_then_newline_is_one_piece() {
        let pieces = pre_tokenize("   \nfoo");
        // Whitespace prefix collapses with the newline.
        assert_eq!(pieces[0], "   \n");
    }

    #[test]
    fn cjk_runs_stay_together() {
        // "你好" is two CJK code points.
        let pieces = pre_tokenize("你好world");
        assert_eq!(pieces[0], "你好");
        assert_eq!(pieces[1], "world");
    }

    #[test]
    fn special_sentinels_match() {
        assert!(match_special_at("<｜User｜>hello", 1, 2, 3, 4, 5, 6, 7).is_some());
    }
}
