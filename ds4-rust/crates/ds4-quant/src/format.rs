//! Per-format block sizes and byte counts.
//!
//! These constants mirror the upstream `ggml-common.h` block layouts:
//!
//! * Q8_0  : 32 elements, 34 bytes  (f16 d + i8 qs\[32\]) - note we
//!   currently store d as f32 in the `q8_0` module's API
//!   because the reference impl uses f32, but the GGML on-disk
//!   layout is f16. We expose the f32 working type because
//!   quantize/dequantize are lossless roundtrips in f32.
//! * Q4_K  : 256 elements, 144 bytes (2 f16 + 12 scales + 128 nibbles)
//! * Q2_K  : 256 elements,  84 bytes (16 scales + 64 quants + 2 f16)
//! * IQ2_XXS: 256 elements, 66 bytes (f16 d + u16 qs\[32\])  (we keep
//!   `d` as `u16` in the typed API to avoid lossy round-trips
//!   with the upstream reference; the on-disk size is 2 + 64.)
//! * F16   : 1 element, 2 bytes
//! * F32   : 1 element, 4 bytes

use super::QuantKind;

/// Number of elements per block for a given quant kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSize(pub usize);

pub const Q8_0_BLOCK: BlockSize = BlockSize(32);
pub const Q4_K_BLOCK: BlockSize = BlockSize(256);
pub const Q2_K_BLOCK: BlockSize = BlockSize(256);
pub const IQ2_XXS_BLOCK: BlockSize = BlockSize(256);
pub const F16_BLOCK: BlockSize = BlockSize(1);
pub const F32_BLOCK: BlockSize = BlockSize(1);

/// Number of elements per block for the given quant kind.
pub fn block_size(k: QuantKind) -> BlockSize {
    match k {
        QuantKind::Q8_0 => Q8_0_BLOCK,
        QuantKind::Q4_K => Q4_K_BLOCK,
        QuantKind::Q2_K => Q2_K_BLOCK,
        QuantKind::Iq2Xxs => IQ2_XXS_BLOCK,
        QuantKind::F16 => F16_BLOCK,
        QuantKind::F32 => F32_BLOCK,
    }
}

/// Number of bytes per block for the given quant kind (matches the
/// upstream C layout).
pub fn block_bytes(k: QuantKind) -> usize {
    match k {
        QuantKind::Q8_0 => 2 + 32,           // ggml_half d + i8 qs\[32\]
        QuantKind::Q4_K => 2 + 2 + 12 + 128, // 2 f16 + K_SCALE_SIZE + QK_K/2
        QuantKind::Q2_K => 16 + 64 + 2 + 2,  // scales[QK_K/16] + qs[QK_K/4] + 2 f16
        QuantKind::Iq2Xxs => 2 + 2 * 32,     // f16 d + u16 qs[QK_K/8]
        QuantKind::F16 => 2,
        QuantKind::F32 => 4,
    }
}

/// Convenience: `block_bytes(k) / core::mem::size_of::<f32>()`.
pub fn block_words_f32(k: QuantKind) -> usize {
    block_bytes(k) / core::mem::size_of::<f32>()
}
