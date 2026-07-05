// DS4 (DwarfStar) — RMSNorm reference kernel.
//
// Standard RMSNorm from the Llama-style architectures:
//
//   rms = sqrt(mean(x^2) + eps)
//   x'[i] = (x[i] / rms) * weight[i]
//
// We do the reduction in f32 and apply the multiplicative gain
// element-wise. The mean is computed by summing `x[i] * x[i]`
// across the slice and dividing by the slice length; we
// precompute `inv_n = 1 / n` so the only sqrt we pay is on the
// summed value.

/// Apply RMSNorm in-place.
///
/// * `x`      — the activation vector to normalize. Modified in
///   place.
/// * `weight` — per-channel gain. Must have the same length as
///   `x`.
/// * `eps`    — the stabilizer added inside the square root.
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm: x and weight must have the same length"
    );
    assert!(
        eps > 0.0,
        "rms_norm: eps must be > 0 (got {eps}); 0-divisor protection"
    );
    let n = x.len();
    if n == 0 {
        return;
    }
    let mut sum_sq = 0.0f64;
    for &v in x.iter() {
        sum_sq += f64::from(v) * f64::from(v);
    }
    let mean_sq = (sum_sq / n as f64) + f64::from(eps);
    let inv_rms = 1.0f32 / (mean_sq.sqrt() as f32);
    for (slot, w) in x.iter_mut().zip(weight.iter()) {
        *slot *= inv_rms * *w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn rmsnorm_hand_computed_uniform_weights() {
        // x = [1, 2, 3, 4], w = [1, 1, 1, 1], eps = 1e-6.
        // mean(x^2) = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5.
        // rms = sqrt(7.5 + 1e-6) ≈ 2.7386...
        // x' = x / rms.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        rms_norm(&mut x, &w, 1e-6);
        let rms = (7.5f32 + 1e-6).sqrt();
        let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
        for (g, e) in x.iter().zip(expected.iter()) {
            assert!(approx_eq(*g, *e, 1e-5));
        }
    }

    #[test]
    fn rmsnorm_applies_per_channel_gain() {
        // With weights [0.5, 2.0, 1.0, 1.0] the post-norm vector
        // must be exactly `(x / rms) * w`.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [0.5f32, 2.0, 1.0, 1.0];
        rms_norm(&mut x, &w, 1e-6);
        let rms = (7.5f32 + 1e-6).sqrt();
        let expected = [
            0.5 * 1.0 / rms,
            2.0 * 2.0 / rms,
            1.0 * 3.0 / rms,
            1.0 * 4.0 / rms,
        ];
        for (g, e) in x.iter().zip(expected.iter()) {
            assert!(approx_eq(*g, *e, 1e-5));
        }
    }

    #[test]
    fn rmsnorm_unit_gain_preserves_norm() {
        // For unit weights and a well-conditioned input, the
        // RMS of the output must equal 1.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        rms_norm(&mut x, &w, 1e-6);
        let sq_sum: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sq_sum / x.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rmsnorm_zero_input_produces_zero() {
        // x = 0 -> sum_sq = 0 -> rms = sqrt(eps) -> x' = 0.
        let mut x = [0.0f32, 0.0, 0.0, 0.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        rms_norm(&mut x, &w, 1e-6);
        for &v in x.iter() {
            assert!(v.abs() < 1e-7);
        }
    }
}
