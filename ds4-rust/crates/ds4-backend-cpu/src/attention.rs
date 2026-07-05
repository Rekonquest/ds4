// DS4 (DwarfStar) — CPU reference attention block.
//
// This is the *correctness path* for attention. It implements the
// standard scaled dot-product attention used during decode-time:
//
//   scores[t] = sum_d q[h, d] * k_cache[t, h, d] / sqrt(head_dim)
//   weights   = softmax(scores)        // causal over t in [0, seq_len)
//   out[h,d]  = sum_t weights[t] * v_cache[t, h, d]
//
// Memory layouts (row-major, contiguous):
// * `q`        — `[n_heads, head_dim]`.
// * `k_cache`  — `[seq_len, n_heads, head_dim]` (i.e. K for every
//                past position, every head).
// * `v_cache`  — `[seq_len, n_heads, head_dim]`, same layout as K.
// * `out`      — `[n_heads, head_dim]`, written in place.
//
// Paged attention (mistralrs-paged-attn) lives in
// `ds4-backend-paged`; this kernel does not page.
//
// Causal masking: every past position is allowed (the standard
// decode-time setup where each newly generated token attends to
// everything before it). The `seq_len` passed in already
// represents the prefix the active query can attend to.

use crate::softmax::softmax;

/// Scaled dot-product attention with a causal mask over
/// `[0, seq_len)`.
///
/// Returns `Err(InvalidArgument)` when the buffer lengths don't
/// match the declared dimensions; otherwise writes the result
/// into `out`.
pub fn attention_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), &'static str> {
    if head_dim == 0 || n_heads == 0 {
        return Err("attention_decode: head_dim and n_heads must be > 0");
    }
    if q.len() != n_heads * head_dim {
        return Err("attention_decode: q length must equal n_heads * head_dim");
    }
    if k_cache.len() != seq_len * n_heads * head_dim {
        return Err("attention_decode: k_cache length must equal seq_len * n_heads * head_dim");
    }
    if v_cache.len() != seq_len * n_heads * head_dim {
        return Err("attention_decode: v_cache length must equal seq_len * n_heads * head_dim");
    }
    if out.len() != n_heads * head_dim {
        return Err("attention_decode: out length must equal n_heads * head_dim");
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Per-head scratch space for the scores.
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..n_heads {
        let q_head = &q[h * head_dim..h * head_dim + head_dim];
        for t in 0..seq_len {
            let k_head = &k_cache[t * (n_heads * head_dim) + h * head_dim..];
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q_head[d] * k_head[d];
            }
            scores[t] = s * scale;
        }
        softmax(&mut scores);
        // Weighted sum of V.
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..seq_len {
                let v_head = &v_cache[t * (n_heads * head_dim) + h * head_dim..];
                acc += scores[t] * v_head[d];
            }
            out[h * head_dim + d] = acc;
        }
    }

    Ok(())
}

/// Scaled dot-product attention allowing caller-supplied
/// per-head scratch space (useful in benchmarks; the
/// `attention_decode` wrapper above is the public entry point).
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_with_scratch(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    scratch: &mut [f32],
) -> Result<(), &'static str> {
    if scratch.len() < seq_len {
        return Err("attention_decode_with_scratch: scratch shorter than seq_len");
    }
    if head_dim == 0 || n_heads == 0 {
        return Err("attention_decode_with_scratch: head_dim and n_heads must be > 0");
    }
    if q.len() != n_heads * head_dim {
        return Err("attention_decode_with_scratch: q length must equal n_heads * head_dim");
    }
    if k_cache.len() != seq_len * n_heads * head_dim {
        return Err("attention_decode_with_scratch: k_cache length mismatch");
    }
    if v_cache.len() != seq_len * n_heads * head_dim {
        return Err("attention_decode_with_scratch: v_cache length mismatch");
    }
    if out.len() != n_heads * head_dim {
        return Err("attention_decode_with_scratch: out length mismatch");
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let q_head = &q[h * head_dim..h * head_dim + head_dim];
        for t in 0..seq_len {
            let k_head = &k_cache[t * (n_heads * head_dim) + h * head_dim..];
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q_head[d] * k_head[d];
            }
            scratch[t] = s * scale;
        }
        softmax(&mut scratch[..seq_len]);
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..seq_len {
                let v_head = &v_cache[t * (n_heads * head_dim) + h * head_dim..];
                acc += scratch[t] * v_head[d];
            }
            out[h * head_dim + d] = acc;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn attention_hand_computed_single_head_seq2() {
        // n_heads=1, head_dim=2, seq_len=2.
        // q = [1, 1]
        // k cache (t, head, d): t=0 -> [[1,0]], t=1 -> [[0,1]]
        // v cache (t, head, d): t=0 -> [[1,0]], t=1 -> [[0,1]]
        //
        // scores:
        //   t=0: dot(q, k0) * 1/sqrt(2) = (1*1+1*0) / sqrt(2) = 1/sqrt(2)
        //   t=1: dot(q, k1) * 1/sqrt(2) = (1*0+1*1) / sqrt(2) = 1/sqrt(2)
        // softmax([1/sqrt(2), 1/sqrt(2)]) = [0.5, 0.5].
        // out = 0.5 * v0 + 0.5 * v1 = 0.5 * [1,0] + 0.5 * [0,1] = [0.5, 0.5].
        let q = [1.0f32, 1.0];
        let k_cache = [1.0f32, 0.0, 0.0, 1.0];
        let v_cache = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 2];
        attention_decode(&q, &k_cache, &v_cache, &mut out, 2, 1, 2).unwrap();
        assert!(approx_eq(out[0], 0.5, 1e-6));
        assert!(approx_eq(out[1], 0.5, 1e-6));
    }

    #[test]
    fn attention_singleton_seq_picks_first_token() {
        // seq_len=1: only one past token; no softmax ambiguity.
        // q = [1, 0]; k = v = [[1, 0]]; dot=1; score = 1/sqrt(2);
        // softmax([1/sqrt(2)]) = [1]. out = v0 = [1, 0].
        let q = [1.0f32, 0.0];
        let k_cache = [1.0f32, 0.0];
        let v_cache = [1.0f32, 0.0];
        let mut out = [0.0f32; 2];
        attention_decode(&q, &k_cache, &v_cache, &mut out, 1, 1, 2).unwrap();
        assert!(approx_eq(out[0], 1.0, 1e-6));
        assert!(approx_eq(out[1], 0.0, 1e-6));
    }

    #[test]
    fn attention_multi_head_orthogonal_inputs() {
        // Two heads; identical data so we can cross-check with the
        // single-head version.
        let q = [1.0f32, 1.0, 1.0, 1.0];
        let k_cache = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let v_cache = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        attention_decode(&q, &k_cache, &v_cache, &mut out, 2, 2, 2).unwrap();
        // Each head sees the same data and should produce [0.5, 0.5].
        assert!(approx_eq(out[0], 0.5, 1e-6));
        assert!(approx_eq(out[1], 0.5, 1e-6));
        assert!(approx_eq(out[2], 0.5, 1e-6));
        assert!(approx_eq(out[3], 0.5, 1e-6));
    }

    #[test]
    fn attention_rejects_dimension_mismatch() {
        let q = [1.0f32, 1.0];
        let k_cache = [1.0f32, 0.0];
        let v_cache = [1.0f32, 0.0];
        // n_heads=2 with a length-2 q buffer -> mismatch.
        let mut out = [0.0f32; 2];
        let r = attention_decode(&q, &k_cache, &v_cache, &mut out, 1, 2, 2);
        assert!(r.is_err());
    }

    #[test]
    fn attention_with_scratch_matches_default() {
        let q = [1.0f32, 1.0];
        let k_cache = [1.0f32, 0.0, 0.0, 1.0];
        let v_cache = [1.0f32, 0.0, 0.0, 1.0];
        let mut out_a = [0.0f32; 2];
        let mut out_b = [0.0f32; 2];
        let mut scratch = [0.0f32; 8];
        attention_decode(&q, &k_cache, &v_cache, &mut out_a, 2, 1, 2).unwrap();
        attention_decode_with_scratch(&q, &k_cache, &v_cache, &mut out_b, 2, 1, 2, &mut scratch)
            .unwrap();
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!(approx_eq(*a, *b, 1e-7));
        }
    }
}
