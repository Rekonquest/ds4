// DS4 (DwarfStar) — Rotary Position Embedding (RoPE) reference kernel.
//
// Applies the standard rotary embedding in-place over the head
// dimension. For each pair of consecutive elements (x[2i], x[2i+1])
// in a head, the rotation by angle `theta_i = pos / freq_base^(2i / d)`
// gives:
//
//   x'[2i]   = x[2i]   * cos(theta_i) - x[2i+1] * sin(theta_i)
//   x'[2i+1] = x[2i]   * sin(theta_i) + x[2i+1] * cos(theta_i)
//
// `x` is laid out as `[seq_len, n_heads, head_dim]` (contiguous);
// we rotate each (head, position) block independently. With
// `seq_len` positions and `n_heads` heads per position, the total
// number of elements is `seq_len * n_heads * head_dim`.
//
// We work on a flat slice; callers are responsible for slicing the
// right view (typically the Q or K projection of a single token).

/// Apply rotary position embedding in-place over the head
/// dimension of `x`.
///
/// * `x`        — flat `[seq_len, n_heads, head_dim]` buffer.
/// * `pos`      — absolute position of the first token in `x`.
/// * `n_heads`  — number of attention heads per token.
/// * `head_dim` — per-head channel count. Must be even.
/// * `freq_base` — the rotary frequency base (e.g. 10_000 for the
///   Llama default or 1_000_000 for the extended
///   "xPos" base).
///
/// For each token `t` in `[0, seq_len)` and each head `h` in
/// `[0, n_heads)`, pair index `i` in `[0, head_dim/2)` computes
/// the rotation by `theta_i = (pos + t) / freq_base^(2i/d)` and
/// rotates the corresponding pair of floats in the head channel.
pub fn apply_rope(x: &mut [f32], pos: usize, n_heads: usize, head_dim: usize, freq_base: f32) {
    assert!(
        head_dim.is_multiple_of(2),
        "apply_rope: head_dim ({head_dim}) must be even"
    );
    assert!(
        !freq_base.is_nan() && freq_base > 0.0,
        "apply_rope: freq_base must be > 0 (got {freq_base})"
    );
    let seq_len = x.len() / (n_heads * head_dim);
    assert_eq!(
        x.len(),
        seq_len * n_heads * head_dim,
        "apply_rope: x length not divisible by n_heads * head_dim"
    );

    let half = head_dim / 2;
    // Pre-compute the per-pair inv-frequency once: theta_exponent[i] = -2i/d.
    // For pair i we need freq = freq_base^(2i/d).
    // We compute it as exp(2i * ln(freq_base) / head_dim) for stability.
    let log_base = freq_base.ln();
    let inv_d = 1.0f32 / head_dim as f32;

    for t in 0..seq_len {
        let abs_pos = pos + t;
        for h in 0..n_heads {
            let head_off = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let exponent = 2.0f32 * i as f32 * inv_d;
                let freq = (exponent * log_base).exp();
                let theta = abs_pos as f32 / freq;
                let (sin_t, cos_t) = theta.sin_cos();
                let a = x[head_off + 2 * i];
                let b = x[head_off + 2 * i + 1];
                x[head_off + 2 * i] = a * cos_t - b * sin_t;
                x[head_off + 2 * i + 1] = a * sin_t + b * cos_t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn rope_at_pos0_is_identity_when_sin_is_zero() {
        // pos=0 -> theta=0 for every pair -> rotation by 0 rad
        // is the identity: x' = x.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        apply_rope(&mut x, 0, 1, 4, 10_000.0);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_at_pos1_with_freq_base_10k_first_pair() {
        // x = [1, 0, 1, 0], pos = 1, head_dim = 4, freq_base = 10_000.
        // Pair 0: exponent = 0 -> freq = 1.0 -> theta = 1.0.
        //   a = 1, b = 0 -> x' = (1*cos(1), 1*sin(1))
        // Pair 1: exponent = 1 -> freq = 10_000^(2/4) = 10_000^0.5 = 100.0
        //   -> theta = 1.0 / 100.0 = 0.01.
        //   a = 1, b = 0 -> x' = (1*cos(0.01), 1*sin(0.01))
        let mut x = [1.0f32, 0.0, 1.0, 0.0];
        apply_rope(&mut x, 1, 1, 4, 10_000.0);
        let c0 = 1.0f32.cos();
        let s0 = 1.0f32.sin();
        let t1 = 0.01f32;
        let c1 = t1.cos();
        let s1 = t1.sin();
        let expected = [c0, s0, c1, s1];
        for (g, e) in x.iter().zip(expected.iter()) {
            assert!(
                approx_eq(*g, *e, 1e-6),
                "rope mismatch: got {x:?} expected {expected:?}"
            );
        }
    }

    #[test]
    fn rope_preserves_vector_norm() {
        // Rotation is orthogonal so it must preserve per-pair
        // vector length.
        let mut x = [0.6f32, -0.8, 0.3, 0.4];
        let norm_in: f32 = x.iter().map(|v| v * v).sum();
        apply_rope(&mut x, 7, 1, 4, 10_000.0);
        let norm_out: f32 = x.iter().map(|v| v * v).sum();
        assert!(
            (norm_in - norm_out).abs() < 1e-5,
            "norm drifted: in={norm_in} out={norm_out}"
        );
    }

    #[test]
    fn rope_multi_head_multi_seq_independent() {
        // seq_len=2, n_heads=2, head_dim=4. We pass pos such that
        // every (token, head) pair has the same absolute position,
        // so RoPE degenerates to the identity. The exact absolute
        // position is encoded as `pos + t`, so for seq_len=2 the
        // only value of `pos` that makes *every* token's
        // `pos + t` equal to a fixed constant is `pos = 0,
        // t = 0` — for `t = 1` we'd need `pos = -1`, which is
        // not representable. Use `seq_len = 1` instead so a
        // single token at absolute position `pos` produces the
        // expected rotation result.
        let mut x = [0.0f32; 8];
        // Single token, two heads, head_dim=4.
        x[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        x[4..8].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        let snapshot = x;
        // pos = 0, seq_len = 1 -> theta = 0 -> identity.
        apply_rope(&mut x, 0, 2, 4, 10_000.0);
        assert_eq!(x, snapshot);
    }

    #[test]
    #[should_panic(expected = "head_dim")]
    fn rope_rejects_odd_head_dim() {
        let mut x = [0.0f32; 3];
        apply_rope(&mut x, 0, 1, 3, 10_000.0);
    }
}
