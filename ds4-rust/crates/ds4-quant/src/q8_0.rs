//! Q8_0 quantization format.
//!
//! Block layout (matches `block_q8_0` in ggml-common.h):
//!
//! ```text
//! ggml_half d;       // scale (we keep f32 internally for round-trip
//!                    // safety; on-disk ggml stores f16)
//! int8_t qs[32];     // signed quants
//! ```
//!
//! Total bytes per block = 2 + 32 = 34 bytes for 32 elements.
//!
//! Quantization mirrors `quantize_row_q8_0_reference` /
//! `quantize_row_q8_0` in ggml-quants.c: the scale `d` is the absolute
//! max of the input divided by 127, and each quantized value is
//! `round(x / d)` clamped to `[-127, 127]`. Dequantization is the
//! inverse `d * qs[i]`.

/// Q8_0 super-block: 32 elements per block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q8_0Block {
    pub d: f32,
    pub qs: [i8; 32],
}

impl Q8_0Block {
    pub const BLOCK_SIZE: usize = 32;

    pub fn new_zero() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 32],
        }
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(34);
        out.extend_from_slice(&self.d.to_le_bytes());
        for &q in &self.qs {
            out.push(q as u8);
        }
        out
    }
}

impl Default for Q8_0Block {
    fn default() -> Self {
        Self::new_zero()
    }
}

/// Quantize 32 f32 values into a single Q8_0 block.
pub fn quantize(input: &[f32; 32]) -> Q8_0Block {
    let mut amax = 0.0f32;
    for &v in input {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let d = amax / 127.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0i8; 32];
    for (i, &v) in input.iter().enumerate() {
        let q = (v * id).round();
        // ggml-quants.c clamps to [-127, 127]; the bounds are
        // symmetric so we can use `f32::clamp` directly.
        let clamped = q.clamp(-127.0, 127.0);
        qs[i] = clamped as i8;
    }
    Q8_0Block { d, qs }
}

/// Dequantize a Q8_0 block back into 32 f32 values.
pub fn dequantize(b: &Q8_0Block, out: &mut [f32; 32]) {
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = b.d * b.qs[i] as f32;
    }
}

/// Compute the dot product of two equal-length block arrays. The
/// caller passes the number of elements (`n`) and the slice pair; we
/// iterate over `n / 32` blocks. The output is written into `out[0]`.
pub fn vec_dot(n: usize, a: &[Q8_0Block], b: &[Q8_0Block], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "q8_0 vec_dot: block slices must match");
    assert!(n <= a.len() * 32, "q8_0 vec_dot: n exceeds input length");
    assert!(!out.is_empty(), "q8_0 vec_dot: output must be non-empty");

    let nb = n / 32;
    let mut sum = 0.0f32;
    for i in 0..nb {
        sum += dot_block(&a[i], &b[i]);
    }
    // Remainder (partial block) - ggml always works in whole blocks, but
    // we tolerate partial inputs by treating missing entries as 0.
    let rem_start = nb * 32;
    if rem_start < n {
        for i in 0..(n - rem_start) {
            sum += a[nb].d * (a[nb].qs[i] as f32) * b[nb].d * (b[nb].qs[i] as f32);
        }
    }
    out[0] = sum;
}

/// Dot product of a single 32-element block pair.
pub fn dot_block(a: &Q8_0Block, b: &Q8_0Block) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..32 {
        sum += a.d * (a.qs[i] as f32) * b.d * (b.qs[i] as f32);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_uniform() {
        let input = [0.5f32; 32];
        let q = quantize(&input);
        let mut out = [0.0f32; 32];
        dequantize(&q, &mut out);
        for i in 0..32 {
            assert!((out[i] - input[i]).abs() < 0.01, "i={} got={}", i, out[i]);
        }
    }

    #[test]
    fn roundtrip_hand_computed() {
        // Hand-picked input: range [0, 31]. amax = 31.0 -> d = 31/127.
        // Each value gets quantized to round(x * 127/31), clamped to 127.
        let input: [f32; 32] = std::array::from_fn(|i| i as f32);
        let q = quantize(&input);
        let mut out = [0.0f32; 32];
        dequantize(&q, &mut out);
        let d_expected = 31.0 / 127.0;
        assert!((q.d - d_expected).abs() < 1e-6);
        // x = 1 -> q = round(127/31) = 4 -> out = 4 * 31/127 = 124/127
        assert!((out[1] - (4.0 * d_expected)).abs() < 1e-5);
        // x = 4 -> q = round(508/31) = 16 -> out = 16 * 31/127 = 496/127
        assert!((out[4] - (16.0 * d_expected)).abs() < 1e-5);
        // x = 31 -> q = clamp(127) -> out = 31.0 exactly
        assert!((out[31] - 31.0).abs() < 1e-5);
    }

    #[test]
    fn dot_self_orthonormal_basis() {
        // Construct an identity-like set: for each i in [0, n), the i-th
        // block has only the i-th entry non-zero.
        let mut a = Q8_0Block::new_zero();
        let mut b = Q8_0Block::new_zero();
        a.d = 1.0;
        b.d = 1.0;
        a.qs[7] = 100;
        b.qs[7] = 100;
        let dot = dot_block(&a, &b);
        assert!((dot - 10000.0).abs() < 1.0);
    }
}
