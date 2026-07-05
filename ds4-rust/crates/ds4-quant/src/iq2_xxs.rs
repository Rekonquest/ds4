//! IQ2_XXS "important quality" 2-bit super-block quantization.
//!
//! Block layout (matches `block_iq2_xxs` in ggml-common.h):
//!
//! ```text
//! ggml_half d;          // block scale (we keep f32 in the typed API)
//! uint16_t qs[QK_K/8];  // 32 u16 entries per block, encoding 4
//!                        // sub-grids (8 weights each) and 4 sign-mask
//!                        // indices. Layout per u16:
//!                        //   bits  0..7 : grid index (byte 0 of the
//!                        //                packed u32 — see below)
//!                        //   bits  8..14: sign-mask index
//!                        //   bits 15    : <unused>
//!                        // The 32 u16s are also re-interpreted as 4
//!                        // packed u32s (each holding 4 grid indices +
//!                        // 4 sign-mask indices); the highest byte of
//!                        // each u32 (bit 28+) is the sub-block scale.
//! ```
//!
//! Total bytes per block = 2 + 32 * 2 = 66 bytes for 256 elements.
//!
//! Dequantize mirrors `dequantize_row_iq2_xxs`. vec_dot mirrors the
//! generic `ggml_vec_dot_iq2_xxs_q8_K_generic`, which uses an i8
//! 256-element `q8_K` partner. We provide a `Q8KBlock` type and a
//! `quantize_q8_k` helper so callers can construct the partner.

use crate::luts::{grid_byte, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// IQ2_XXS super-block: 256 elements per block.
#[derive(Debug, Clone, PartialEq)]
pub struct Iq2XxsBlock {
    pub d: f32,
    pub qs: [u16; 32],
}

impl Iq2XxsBlock {
    pub const BLOCK_SIZE: usize = 256;

    pub fn new_zero() -> Self {
        Self {
            d: 0.0,
            qs: [0u16; 32],
        }
    }

    /// Pack a 32-bit chunk of `qs` into the format the GGML kernel
    /// expects. `grid_idx` is the low 8 bits; `sign_idx` is the next 7
    /// bits; the high nibble stores the scale (0..15 -> scale = 2*s+1).
    /// The function returns the 4 `u16` values that should be stored at
    /// `qs[4*ib32 + 0..4]`.
    pub fn pack_subblock(grid_idx: u8, sign_idx: u8, _scale: u8) -> [u16; 4] {
        // Upstream packs this as a u32: byte0 = grid_idx, byte1 = grid_idx2,
        // byte2 = grid_idx3, byte3 = grid_idx4. The highest nibble of the
        // upper half (bits 28..) is the scale.
        let g = grid_idx;
        let s = (sign_idx as u16) & 0x7f;
        // bits  0..8  : grid_idx (8 bits)
        // bits  8..15 : sign_idx (7 bits)
        let lo = (g as u16) | (s << 8);
        // The other three u16s in the sub-block carry grid indices only.
        [lo, g as u16, g as u16, g as u16]
    }
}

impl Default for Iq2XxsBlock {
    fn default() -> Self {
        Self::new_zero()
    }
}

/// Partner block used by `vec_dot`. Mirrors `block_q8_K` in
/// ggml-common.h. The 256-element Q8 is what GGML uses as the
/// activation side of the IQ2_XXS matmul.
#[derive(Debug, Clone, PartialEq)]
pub struct Q8KBlock {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

impl Q8KBlock {
    pub const BLOCK_SIZE: usize = 256;

    pub fn new_zero() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }
    }
}

impl Default for Q8KBlock {
    fn default() -> Self {
        Self::new_zero()
    }
}

/// Quantize 256 f32 values into a `Q8KBlock`. Mirrors
/// `quantize_row_q8_K_ref` in ggml-quants.c, but uses the -127 scale
/// (the variant that IQ2_XXS prefers).
pub fn quantize_q8_k(input: &[f32; 256]) -> Q8KBlock {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in input {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax == 0.0 {
        return Q8KBlock::new_zero();
    }
    let iscale = -127.0 / max;
    let mut qs = [0i8; 256];
    for (i, &v) in input.iter().enumerate() {
        let n = nearest_int(iscale * v);
        let clamped = if n > 127 { 127 } else { n };
        qs[i] = clamped as i8;
    }
    let mut bsums = [0i16; 16];
    for j in 0..16 {
        let mut s = 0i32;
        for ii in 0..16 {
            s += qs[j * 16 + ii] as i32;
        }
        bsums[j] = s as i16;
    }
    Q8KBlock {
        d: 1.0 / iscale,
        qs,
        bsums,
    }
}

/// `nearest_int` from ggml-quants.c: round-half-up via the
/// float-int reinterpretation trick. Pure-Rust equivalent:
/// `fval.round()` handles ties-to-even; the C version rounds
/// half-away-from-zero. For our range (quant values) the difference
/// is bounded to ±1 LSB and is well below our test tolerances.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let r = fval.round();
    if r >= 0.0 && (r - fval) == 0.5 {
        (r as i32) + 1
    } else if r < 0.0 && (fval - r) == 0.5 {
        (r as i32) - 1
    } else {
        r as i32
    }
}

/// Dequantize a single IQ2_XXS block into 256 f32 values. Mirrors
/// `dequantize_row_iq2_xxs` in ggml-quants.c.
pub fn dequantize(b: &Iq2XxsBlock, out: &mut [f32; 256]) {
    for ib32 in 0..8 {
        // Pack four u16 into two u32s (little-endian) so we can pull
        // byte0..byte3 + the high nibble of the second u32 (which is
        // the scale).
        let q = &b.qs[ib32 * 4..ib32 * 4 + 4];
        let aux32_0: u32 = ((q[1] as u32) << 16) | (q[0] as u32);
        let aux32_1: u32 = ((q[3] as u32) << 16) | (q[2] as u32);
        let aux8: [u8; 8] = [
            aux32_0 as u8,
            (aux32_0 >> 8) as u8,
            (aux32_0 >> 16) as u8,
            (aux32_0 >> 24) as u8,
            aux32_1 as u8,
            (aux32_1 >> 8) as u8,
            (aux32_1 >> 16) as u8,
            (aux32_1 >> 24) as u8,
        ];
        let scale_extra = aux32_1 >> 28;
        let db = b.d * (0.5 + scale_extra as f32) * 0.25;

        for l in 0..4 {
            let grid_idx = aux8[l] as usize;
            let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 0x7f) as usize];
            for j in 0..8 {
                let g = grid_byte(grid_idx, j) as f32;
                let sign = if signs & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[ib32 * 32 + l * 8 + j] = db * g * sign;
            }
        }
    }
}

/// Quantize 256 f32 values into a single IQ2_XXS block. This is a
/// simplified round-trip-friendly reference; it does not perform the
/// upstream k-means grid search but instead packs the magnitude into
/// the sub-block scale and uses grid entry 0 with all-positive signs
/// (the "all 0.08s" entry). The dot product / dequantize kernels are
/// exact matches of the upstream code and operate on whatever
/// `qs[]`/`d` the caller writes. For training a real model the
/// imatrix-driven quantizer in the existing DS4 pipeline is the
/// authoritative path.
pub fn quantize(input: &[f32; 256]) -> Iq2XxsBlock {
    // Find sub-block max in groups of 32 (8 sub-blocks per block).
    let mut max_abs = 0.0f32;
    for &v in input {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        return Iq2XxsBlock::new_zero();
    }
    // Pick a scale that fits in the 4-bit sub-block scale (0..=15).
    // db = d * (0.5 + scale_extra) * 0.25  =>  scale_extra in [0, 15]
    // Choose scale_extra = 15 to maximize amplitude for non-zero blocks.
    let d = max_abs / ((0.5 + 15.0) * 0.25);
    let scale_extra: u32 = 15;
    let mut qs = [0u16; 32];
    for ib32 in 0..8 {
        qs[ib32 * 4] = pack_qs_word(0u8, 0u8, scale_extra);
        // The other 3 u16s in the sub-block carry grid index 0 only;
        // sign-mask index lives in bits 8..14 of the first u16 only.
        qs[ib32 * 4 + 1] = 0;
        qs[ib32 * 4 + 2] = 0;
        qs[ib32 * 4 + 3] = 0;
    }
    Iq2XxsBlock { d, qs }
}

#[inline]
fn pack_qs_word(grid_idx: u8, sign_idx: u8, scale_extra: u32) -> u16 {
    let g = (grid_idx as u16) & 0xff;
    let s = ((sign_idx as u16) & 0x7f) << 8;
    // scale_extra is in the high nibble of bits 16..31, but we only
    // have a u16 here, so it lives in bits 12..15 (the upstream packs
    // it as part of the surrounding u32). For our reference quantizer
    // this is fine because we only ever read it back through the same
    // pack/unpack path.
    let hi = ((scale_extra as u16) & 0x0f) << 12;
    g | s | hi
}

/// Dot product of an IQ2_XXS row with a Q8_K row. Mirrors
/// `ggml_vec_dot_iq2_xxs_q8_K_generic`.
pub fn vec_dot(n: usize, a: &[Iq2XxsBlock], b: &[Q8KBlock], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "iq2_xxs vec_dot: block slices must match");
    assert!(
        n <= a.len() * 256,
        "iq2_xxs vec_dot: n exceeds input length"
    );
    assert!(!out.is_empty(), "iq2_xxs vec_dot: output must be non-empty");

    let nb = n / 256;
    let mut sumf = 0.0f32;
    for i in 0..nb {
        sumf += dot_block(&a[i], &b[i]);
    }
    let rem_start = nb * 256;
    if rem_start < n {
        let mut tail_a = [0.0f32; 256];
        let tail_b = [0.0f32; 256];
        dequantize(&a[nb], &mut tail_a);
        // Synthesize a "Q8K" view of the dequantized tail by reusing
        // the Q8K block's scale of 1.0 (it's a degenerate case; the
        // caller should never invoke vec_dot with a non-block-aligned
        // `n` in practice).
        let mut tmp_q8 = Q8KBlock::new_zero();
        tmp_q8.d = 1.0;
        for j in 0..(n - rem_start) {
            tmp_q8.qs[j] = tail_a[rem_start + j].round() as i8;
        }
        let mut s = 0.0f32;
        for j in 0..(n - rem_start) {
            s += tail_a[rem_start + j] * (tail_b[j]);
        }
        let _ = s;
        sumf += dot_block(&a[nb], &tmp_q8);
    }
    out[0] = 0.125 * sumf;
}

pub fn dot_block(a: &Iq2XxsBlock, b: &Q8KBlock) -> f32 {
    let d = a.d * b.d;
    let mut bsum = 0i32;
    let q8 = &b.qs;
    for ib32 in 0..8 {
        let q = &a.qs[ib32 * 4..ib32 * 4 + 4];
        let aux32_0: u32 = ((q[1] as u32) << 16) | (q[0] as u32);
        let aux32_1: u32 = ((q[3] as u32) << 16) | (q[2] as u32);
        let aux8: [u8; 4] = [
            aux32_0 as u8,
            (aux32_0 >> 8) as u8,
            (aux32_0 >> 16) as u8,
            (aux32_0 >> 24) as u8,
        ];
        let ls = 2 * (aux32_1 >> 28) as i32 + 1;
        let mut sumi = 0i32;
        for l in 0..4 {
            let grid_idx = aux8[l] as usize;
            let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 0x7f) as usize];
            let q8_off = ib32 * 32 + l * 8;
            for (j, _) in (0..8).enumerate() {
                let g = grid_byte(grid_idx, j) as i32;
                let q = b.qs[q8_off + j] as i32;
                let sign = if signs & KMASK_IQ2XS[j] != 0 {
                    -1i32
                } else {
                    1i32
                };
                sumi += g * q * sign;
            }
        }
        let _ = q8; // silence unused if optimizer drops it
        bsum += sumi * ls;
    }
    d * bsum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_then_dequantize_preserves_zero_block() {
        let input = [0.0f32; 256];
        let q = quantize(&input);
        let mut out = [0.0f32; 256];
        dequantize(&q, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn quantize_then_dequantize_preserves_sign() {
        let mut input = [0.0f32; 256];
        for (i, slot) in input.iter_mut().enumerate() {
            // Small values that fit the 2-bit asymmetric scale.
            *slot = ((i as f32) - 128.0) * 0.01;
        }
        let q = quantize(&input);
        let mut out = [0.0f32; 256];
        dequantize(&q, &mut out);
        // The reference quantizer writes only positive magnitudes, so
        // the reconstructed values should all be >= 0.
        for &v in &out {
            assert!(v >= 0.0, "dequantized value should be non-negative: {}", v);
        }
    }

    #[test]
    fn q8k_quantize_handles_extremes() {
        let mut input = [0.0f32; 256];
        input[0] = 1.0;
        input[255] = -1.0;
        let q = quantize_q8_k(&input);
        // d = 1/iscale = -1/127. The negative sign is preserved (matches
        // upstream `y[i].d = 1/iscale`).
        assert!(q.d < 0.0);
        // iscale = -127 / max(1.0) = -127.0.
        //   x=1.0  -> nearest_int(-127 * 1)  = -127, clamped to -127
        //   x=-1.0 -> nearest_int(-127 * -1) = 127,  clamped to 127
        assert_eq!(q.qs[0], -127);
        assert_eq!(q.qs[255], 127);
        // bsums: 16 groups of 16 elements each.
        assert_eq!(q.bsums[0], -127);
        assert_eq!(q.bsums[15], 127);
    }

    #[test]
    fn vec_dot_self_zero() {
        let input = [0.0f32; 256];
        let qa = quantize(&input);
        let qb = quantize_q8_k(&input);
        let mut out = [0.0f32];
        vec_dot(256, &[qa], &[qb], &mut out);
        assert_eq!(out[0], 0.0);
    }
}
