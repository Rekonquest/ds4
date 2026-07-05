//! Q3_K 3-bit super-block quantization.
//!
//! Block layout (matches `block_q3_K` in ggml-common.h):
//!
//! ```text
//! uint8_t hmask[QK_K/8];   // 32 bytes, high bit mask for signed 3-bit values
//! uint8_t qs[QK_K/4];      // 64 bytes, low two bits packed four values per byte
//! uint8_t scales[12];      // sixteen packed 6-bit signed scales, biased by 32
//! ggml_half d;             // shared block scale
//! ```
//!
//! Total bytes per block = 32 + 64 + 12 + 2 = 110 bytes for 256 elements.
//!
//! Weight reconstruction follows `dequantize_row_q3_K` from GGML:
//! `x = d * signed_scale * (low_2_bits - high_bit_bias)`.

use crate::q4_k::Q8KView;

type Q3ScaleBytes = [u8; 12];
type Q3SignedScales = [i8; 16];

/// Q3_K super-block: 256 elements per block.
#[derive(Debug, Clone, PartialEq)]
pub struct Q3_KBlock {
    pub d: f32,
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: Q3ScaleBytes,
}

impl Q3_KBlock {
    pub const BLOCK_SIZE: usize = 256;

    pub fn new_zero() -> Self {
        Self {
            d: 0.0,
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: [0u8; 12],
        }
    }
}

impl Default for Q3_KBlock {
    fn default() -> Self {
        Self::new_zero()
    }
}

#[inline]
fn unpack_scales(scales: &Q3ScaleBytes) -> Q3SignedScales {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let aux0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    let aux1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    let tmp = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);

    let words = [
        (aux0 & KMASK2) | ((tmp & KMASK1) << 4),
        (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];

    let mut out = [0i8; 16];
    for (word_idx, word) in words.iter().enumerate() {
        for (byte_idx, byte) in word.to_le_bytes().iter().enumerate() {
            out[word_idx * 4 + byte_idx] = *byte as i8 - 32;
        }
    }
    out
}

/// Dequantize a single Q3_K block into 256 f32 values. Mirrors
/// `dequantize_row_q3_K` in ggml-quants.c.
pub fn dequantize(b: &Q3_KBlock, out: &mut [f32; 256]) {
    let scales = unpack_scales(&b.scales);
    let mut scale_idx = 0usize;
    let mut out_off = 0usize;
    let mut q_off = 0usize;
    let mut hmask_bit = 1u8;

    for _ in 0..2 {
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = b.d * scales[scale_idx] as f32;
            scale_idx += 1;
            for l in 0..16 {
                let q = ((b.qs[q_off + l] >> shift) & 3) as i8;
                let high_bias = if b.hmask[l] & hmask_bit != 0 { 0 } else { 4 };
                out[out_off + l] = dl * (q - high_bias) as f32;
            }
            out_off += 16;

            let dl = b.d * scales[scale_idx] as f32;
            scale_idx += 1;
            for l in 0..16 {
                let q = ((b.qs[q_off + 16 + l] >> shift) & 3) as i8;
                let high_bias = if b.hmask[16 + l] & hmask_bit != 0 {
                    0
                } else {
                    4
                };
                out[out_off + l] = dl * (q - high_bias) as f32;
            }
            out_off += 16;

            shift += 2;
            hmask_bit <<= 1;
        }
        q_off += 32;
    }
}

/// Dot product of a Q3_K row with a Q8_K row. This uses the reference
/// dequantization path for correctness and shares the `Q8KView` used by
/// the other K-quant kernels.
pub fn vec_dot(n: usize, a: &[Q3_KBlock], b: &[Q8KView<'_>], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "q3_k vec_dot: block slices must match");
    assert!(n <= a.len() * 256, "q3_k vec_dot: n exceeds input length");
    assert!(!out.is_empty(), "q3_k vec_dot: output must be non-empty");

    let nb = n / 256;
    let mut sum = 0.0f32;
    for i in 0..nb {
        sum += dot_block(&a[i], &b[i]);
    }
    out[0] = sum;
}

pub fn dot_block(a: &Q3_KBlock, b: &Q8KView<'_>) -> f32 {
    let mut deq = [0.0f32; 256];
    dequantize(a, &mut deq);
    deq.iter()
        .zip(b.qs.iter())
        .map(|(x, q)| *x * b.d * *q as f32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_scales(scales: Q3SignedScales) -> Q3ScaleBytes {
        let mut out = [0u8; 12];
        for (j, scale) in scales.iter().enumerate() {
            let packed = (*scale + 32) as u8;
            if j < 8 {
                out[j] = packed & 0x0f;
            } else {
                out[j - 8] |= (packed & 0x0f) << 4;
            }
            out[j % 4 + 8] |= (packed >> 4) << (2 * (j / 4));
        }
        out
    }

    #[test]
    fn unpack_scales_roundtrips_reference_packing() {
        let scales = [-32, -17, -1, 0, 1, 7, 15, 31, -8, -3, 2, 6, 10, 14, 20, 30];
        let packed = pack_scales(scales);
        assert_eq!(unpack_scales(&packed), scales);
    }

    #[test]
    fn dequantize_zero_block_is_zero() {
        let b = Q3_KBlock::new_zero();
        let mut out = [99.0f32; 256];
        dequantize(&b, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_hand_computed() {
        let mut scales = [0i8; 16];
        scales[0] = 2;
        let mut b = Q3_KBlock {
            d: 0.5,
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: pack_scales(scales),
        };
        for mask in b.hmask.iter_mut().take(16) {
            *mask = 1;
        }
        b.qs[0] = 3;

        let mut out = [0.0f32; 256];
        dequantize(&b, &mut out);
        assert_eq!(out[0], 3.0);
        assert!(out[1..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn dot_block_matches_dequantized_reference() {
        let mut scales = [0i8; 16];
        scales[0] = 1;
        let mut b = Q3_KBlock {
            d: 1.0,
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: pack_scales(scales),
        };
        for mask in b.hmask.iter_mut().take(16) {
            *mask = 1;
        }
        b.qs[0] = 3;

        let mut qs8 = [0i8; 256];
        qs8[0] = 5;
        let bsums = [0i16; 16];
        let view = Q8KView {
            d: 1.0,
            qs: &qs8,
            bsums: &bsums,
        };

        let mut out = [0.0f32];
        vec_dot(256, &[b], &[view], &mut out);
        assert_eq!(out[0], 15.0);
    }
}
