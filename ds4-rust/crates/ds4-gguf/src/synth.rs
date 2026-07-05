//
// DS4 (DwarfStar) -- synthetic GGUF generator for end-to-end tests.
//
// Builds a *minimal* valid GGUF v3 file in memory, suitable for
// driving the engine's load + forward path without needing a real
// DeepSeek V4 model on disk. The schema is deliberately minimal:
//
//   - 1 layer, 8 hidden dim, 1 head (no GQA), 16-token vocab
//   - weight tensors (all F32, named to match the GGUF descriptors
//     the engine reads):
//       token_embd.weight    [vocab, hidden]
//       output.weight        [hidden, vocab]
//       output_norm.weight   [hidden]
//       blk.0.attn_q.weight  [hidden, hidden]
//       blk.0.attn_k.weight  [hidden, hidden]
//       blk.0.attn_v.weight  [hidden, hidden]
//       blk.0.attn_out.weight [hidden, hidden]
//       blk.0.attn_norm.weight  [hidden]
//       blk.0.ffn_gate.weight   [hidden, hidden]
//       blk.0.ffn_up.weight     [hidden, hidden]
//       blk.0.ffn_down.weight   [hidden, hidden]
//       blk.0.ffn_norm.weight   [hidden]
//   - metadata kv pairs: ds4.vocab_size, ds4.embedding_dim, etc.
//
// The weights are deterministic small integers; running the engine
// against this file should produce a deterministic output token.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use ds4_types::Ds4Error;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: u64 = 32;

// ggml.h GGML_TYPE values
const GGML_TYPE_F32: u32 = 0;

// ggml.h gguf_metadata_value_type values
const GGUF_METADATA_VALUE_TYPE_UINT32: u32 = 4;
const GGUF_METADATA_VALUE_TYPE_STRING: u32 = 8;
const GGUF_METADATA_VALUE_TYPE_ARRAY: u32 = 9;

/// Build a tiny synthetic GGUF v3 file at `path` and return the
/// path. The file is suitable for driving the engine's load + forward
/// path in tests.
pub fn write_synthetic_gguf(path: &Path) -> Result<(), Ds4Error> {
    let mut buf: Vec<u8> = Vec::new();

    let vocab_size: u32 = 16;
    let hidden: u32 = 8;
    let n_layers: u32 = 1;
    let n_heads: u32 = 1;
    let head_dim: u32 = hidden / n_heads; // 8
    let context_length: u32 = 64;
    let expert_count: u32 = 0; // no MoE in the synthetic model
    let expert_used_count: u32 = 0;

    // Header: magic, version, n_tensors, n_kv. GGUF v3 stores the
    // two counts as u64; alignment is metadata (`general.alignment`).
    buf.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    buf.write_all(&GGUF_VERSION.to_le_bytes()).unwrap();
    let n_tensors: u64 = 12;
    let n_kv: u64 = 21;
    buf.write_all(&n_tensors.to_le_bytes()).unwrap();
    buf.write_all(&n_kv.to_le_bytes()).unwrap();

    // Metadata kv pairs
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    write_kv_string(&mut buf, "general.name", "ds4-synthetic-v0");
    write_kv_string(&mut buf, "general.architecture", "ds4-synth");
    write_kv_u32(&mut buf, "general.alignment", GGUF_ALIGNMENT as u32);
    write_kv_u32(&mut buf, "ds4.vocab_size", vocab_size);
    write_kv_u32(&mut buf, "ds4.embedding_dim", hidden);
    write_kv_u32(&mut buf, "ds4.block_count", n_layers);
    write_kv_u32(&mut buf, "ds4.attention.head_count", n_heads);
    write_kv_u32(&mut buf, "ds4.feed_forward_length", hidden * 2);
    write_kv_u32(&mut buf, "ds4.expert_count", expert_count);
    write_kv_u32(&mut buf, "ds4.expert_used_count", expert_used_count);
    write_kv_u32(&mut buf, "ds4.context_length", context_length);
    write_kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 1);
    write_kv_u32(&mut buf, "tokenizer.ggml.eos_token_id", 2);
    write_kv_u32(&mut buf, "tokenizer.ggml.unk_token_id", 0);
    write_kv_string_array(
        &mut buf,
        "tokenizer.ggml.tokens",
        &[
            "<unk>",
            "<s>",
            "</s>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think_start>",
            "<think_end>",
            "<dsml>",
            "h",
            "i",
            "hi",
            " ",
            "e",
            "l",
            "o",
            "!",
        ],
    );
    write_kv_string_array(&mut buf, "tokenizer.ggml.merges", &["h i"]);
    write_kv_u32(&mut buf, "ds4.user_token_id", 3);
    write_kv_u32(&mut buf, "ds4.assistant_token_id", 4);
    write_kv_u32(&mut buf, "ds4.think_start_token_id", 5);
    write_kv_u32(&mut buf, "ds4.think_end_token_id", 6);
    write_kv_u32(&mut buf, "ds4.dsml_token_id", 7);

    // Tensor descriptors + data. We must know each tensor's
    // (name, dims, n_bytes) to compute the data-region offset.
    // Aligned to GGUF_ALIGNMENT.
    let tensors: Vec<(&str, Vec<u32>)> = vec![
        ("token_embd.weight", vec![vocab_size, hidden]),
        ("output_norm.weight", vec![hidden]),
        ("output.weight", vec![hidden, vocab_size]),
        ("blk.0.attn_norm.weight", vec![hidden]),
        ("blk.0.attn_q.weight", vec![hidden, hidden]),
        ("blk.0.attn_k.weight", vec![hidden, hidden]),
        ("blk.0.attn_v.weight", vec![hidden, hidden]),
        ("blk.0.attn_out.weight", vec![hidden, hidden]),
        ("blk.0.ffn_norm.weight", vec![hidden]),
        ("blk.0.ffn_gate.weight", vec![hidden, hidden * 2]),
        ("blk.0.ffn_up.weight", vec![hidden, hidden * 2]),
        ("blk.0.ffn_down.weight", vec![hidden * 2, hidden]),
    ];

    // Tensor header region
    let mut tensor_offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut data_offset: u64 = 0;
    for (name, dims) in &tensors {
        write_string(&mut buf, name);
        let n_dims = dims.len() as u32;
        buf.write_all(&n_dims.to_le_bytes()).unwrap();
        for d in dims {
            buf.write_all(&(*d as u64).to_le_bytes()).unwrap();
        }
        buf.write_all(&GGML_TYPE_F32.to_le_bytes()).unwrap();
        tensor_offsets.push(data_offset);
        buf.write_all(&data_offset.to_le_bytes()).unwrap();
        let n_elements: u64 = dims.iter().map(|d| *d as u64).product();
        data_offset += n_elements * 4; // F32 = 4 bytes
                                       // align to GGUF_ALIGNMENT
        data_offset = (data_offset + GGUF_ALIGNMENT - 1) & !(GGUF_ALIGNMENT - 1);
    }

    // Pad to align the data region to GGUF_ALIGNMENT
    while !buf.len().is_multiple_of(GGUF_ALIGNMENT as usize) {
        buf.push(0);
    }
    let data_region_start = buf.len();

    // Tensor data region: deterministic small values, no RNG.
    // Each tensor is filled with a simple pattern that the engine
    // can use to produce a deterministic argmax.
    for (idx, (_name, dims)) in tensors.iter().enumerate() {
        let n_elements: usize = dims.iter().map(|d| *d as usize).product();
        let start = data_region_start + tensor_offsets[idx] as usize;
        // Pad bytes between end of previous and start of this
        while buf.len() < start {
            buf.push(0);
        }
        // Write 4-byte little-endian f32 values
        for elem in 0..n_elements {
            // Diagonal-dominant pattern: bias diagonal up by 0.5,
            // everything else 0.1, so argmax on logits = `output` is
            // deterministic (the token with the highest embedding
            // magnitude wins).
            let row = elem / dims.last().copied().unwrap_or(1) as usize;
            let col = elem % dims.last().copied().unwrap_or(1) as usize;
            let v: f32 = if row == col { 0.5 } else { 0.1 };
            let bytes = (idx as f32 * 0.001 + v).to_le_bytes();
            buf.extend_from_slice(&bytes);
        }
        // pad to alignment
        while !buf.len().is_multiple_of(GGUF_ALIGNMENT as usize) {
            buf.push(0);
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            ds4_types::Ds4Error::new(ds4_types::Ds4ErrorKind::Io, format!("create parent: {e}"))
        })?;
    }
    std::fs::write(path, &buf).map_err(|e| {
        ds4_types::Ds4Error::new(ds4_types::Ds4ErrorKind::Io, format!("write {path:?}: {e}"))
    })?;
    let _ = (
        vocab_size,
        hidden,
        n_layers,
        head_dim,
        context_length,
        HashMap::<&str, ()>::new(),
    );
    Ok(())
}

fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    // key length + key bytes
    buf.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(key.as_bytes()).unwrap();
    // value type
    buf.write_all(&GGUF_METADATA_VALUE_TYPE_UINT32.to_le_bytes())
        .unwrap();
    // value (u32 LE)
    buf.write_all(&value.to_le_bytes()).unwrap();
}

fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    buf.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(key.as_bytes()).unwrap();
    buf.write_all(&GGUF_METADATA_VALUE_TYPE_STRING.to_le_bytes())
        .unwrap();
    buf.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(value.as_bytes()).unwrap();
}

fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
    buf.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(key.as_bytes()).unwrap();
    buf.write_all(&GGUF_METADATA_VALUE_TYPE_ARRAY.to_le_bytes())
        .unwrap();
    buf.write_all(&GGUF_METADATA_VALUE_TYPE_STRING.to_le_bytes())
        .unwrap();
    buf.write_all(&(values.len() as u64).to_le_bytes()).unwrap();
    for value in values {
        buf.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        buf.write_all(value.as_bytes()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_synthetic_gguf_produces_loadable_file() {
        let path = std::env::temp_dir().join("ds4-synth-test.gguf");
        write_synthetic_gguf(&path).expect("write synth gguf");
        let bytes = std::fs::read(&path).expect("read back");
        // magic
        assert_eq!(&bytes[0..4], &GGUF_MAGIC.to_le_bytes());
        // version
        assert_eq!(&bytes[4..8], &GGUF_VERSION.to_le_bytes());
        // must be at least header + 12 tensor headers
        assert!(
            bytes.len() > 1024,
            "synth gguf should be non-trivial: got {} bytes",
            bytes.len()
        );
        let _ = std::fs::remove_file(&path);
    }
}
