//! F32 (native IEEE 754 binary32) format pass-through.
//!
//! Quantize/dequantize are identity casts. The interesting kernel is
//! `vec_dot`, which the rest of DS4 uses for f32 reference matmul.

/// Identity: copy f32 into a working buffer.
#[inline]
pub fn quantize(input: &[f32]) -> Vec<f32> {
    input.to_vec()
}

/// Identity: copy f32 into the output slice.
#[inline]
pub fn dequantize(input: &[f32], out: &mut [f32]) {
    assert_eq!(input.len(), out.len());
    out.copy_from_slice(input);
}

/// Plain dot product of two equal-length slices.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "f32 dot: slices must match");
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// Inner-product of one row of block-quantized data (a sequence of f32
/// values) with another sequence of f32 values. Same shape as the
/// vec_dot APIs for the other formats so callers can dispatch uniformly.
pub fn vec_dot(n: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    assert!(n <= a.len() && n <= b.len(), "f32 vec_dot: n out of range");
    assert!(!out.is_empty(), "f32 vec_dot: output must be non-empty");
    out[0] = dot(&a[..n], &b[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_works() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn vec_dot_writes_to_out() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32];
        vec_dot(4, &a, &b, &mut out);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((out[0] - 70.0).abs() < 1e-5);
    }
}
