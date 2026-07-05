//! F16 (IEEE 754 binary16) format pass-through.
//!
//! GGML stores F16 as a `ggml_half` which is just a 16-bit IEEE half.
//! The `half` crate provides the type and conversions. This module
//! exists primarily to give callers a stable DS4-facing API and a
//! place to host any SIMD-friendly conversions.

pub use half::f16 as F16;

/// Convert f32 -> f16 (lossy on denormals / overflow).
#[inline]
pub fn from_f32(v: f32) -> F16 {
    F16::from_f32(v)
}

/// Convert f16 -> f32.
#[inline]
pub fn to_f32(v: F16) -> f32 {
    v.to_f32()
}

/// Quantize (passthrough: just cast).
#[inline]
pub fn quantize(input: &[f32]) -> Vec<F16> {
    input.iter().copied().map(F16::from_f32).collect()
}

/// Dequantize (passthrough: just cast back to f32).
#[inline]
pub fn dequantize(input: &[F16], out: &mut [f32]) {
    assert_eq!(input.len(), out.len());
    for (o, i) in out.iter_mut().zip(input.iter()) {
        *o = i.to_f32();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_known_values() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 2.0, 65504.0, -65504.0, 0.1];
        let mut out = vec![0.0f32; vals.len()];
        let q = quantize(&vals);
        dequantize(&q, &mut out);
        for (orig, deq) in vals.iter().zip(out.iter()) {
            // f16 has 11-bit mantissa, so allow ~1e-3 relative error.
            let tol = orig.abs().max(1.0) * 1e-3;
            assert!((orig - deq).abs() <= tol, "orig={} deq={}", orig, deq);
        }
    }
}
