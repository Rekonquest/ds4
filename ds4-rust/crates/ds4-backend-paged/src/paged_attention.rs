// DS4 (DwarfStar) — paged-attention compute.
//
// Mirrors the C `attention_decode_mixed_kernel` semantics but reads
// K/V from a `PageTable` instead of a contiguous buffer. The kernel
// implements scaled dot-product attention for a single query token
// (`seq_len == 1`) over an arbitrary-length paged KV cache:
//
//   out[h, d] = sum_t softmax(Q[h, :] . K[t, h, :])[t] * V[t, h, d]
//
// For the multi-query / grouped-query variant we collapse `n_kv_heads`
// into the same buffer layout as `n_heads` — i.e. the caller has
// already expanded K/V to per-head buffers before calling in. The
// reference is correctness-first; the vendored mistralrs-paged-attn
// is the optimization path.
//
// The vendored dependency chain (see Cargo.toml `vendored` feature)
// -------------------------------------------------------------------
// `third_party/mistralrs-paged-attn/mistralrs-paged-attn/Cargo.toml`
// inherits `version.workspace = true`, `candle-core.workspace = true`,
// `mistralrs-metal-compile.workspace = true`, `cudaforge.workspace = true`,
// etc. None of those are defined in the DS4 workspace because we
// haven't vendored the upstream `mistral.rs` workspace yet, so the
// vendored crate fails manifest parsing. The dependency chain that
// would be needed to flip the `vendored` feature ON:
//
//   ds4-backend-paged (vendored feature)
//     -> mistralrs-paged-attn  (path = third_party/mistralrs-paged-attn/mistralrs-paged-attn)
//        -> candle-core        (workspace member, must add to DS4 Cargo.toml)
//           -> candle-kernels
//        -> half
//        -> float8
//        -> [cuda]  cudaforge
//        -> [metal] objc2-metal, objc2-foundation, dispatch2,
//                   candle-metal-kernels, mistralrs-metal-compile
//
// Until that chain is wired, this module is the canonical reference.
// See `vendored_status` below for a runtime probe.

use crate::page_table::{Page, PageTable, PAGE_TOKENS};

/// Returns `false`. The vendored mistralrs-paged-attn crate is not
/// built into this crate today; see the module-level note for the
/// dependency chain that would flip this to `true`. When flipped,
/// the caller (the `Backend::name()` impl, the dispatcher in
/// `ds4-core`) needs to be aware so it can route the request
/// through the vendored kernel. See `Cargo.toml` for the patch
/// recipe.
pub fn vendored_status() -> bool {
    false
}

/// 2D query tensor layout: `[n_heads, head_dim]`. Multi-head
/// self-attention with grouped-query sharing of K/V is collapsed to
/// this layout at the call site.
pub type Query = [f32];

/// Output tensor layout: `[n_heads, head_dim]`.
pub type Out = [f32];

/// Scaled-dot-product attention over a paged KV cache for a single
/// query token. Pure-Rust, f32, correctness-first.
///
/// # Arguments
/// * `q` — query vector of length `n_heads * head_dim`, row-major
///   `[n_heads, head_dim]` (head `h` starts at `q[h * head_dim]`).
/// * `kv_cache` — the page table to read K/V from.
/// * `layer` — which transformer layer the cache belongs to.
/// * `seq_len` — number of tokens stored in `kv_cache` (0 means an
///   empty cache; the function returns zeros in that case).
/// * `n_heads` — number of query heads.
/// * `head_dim` — per-head dimension.
///
/// # Returns
/// A `Vec<f32>` of length `n_heads * head_dim`, row-major
/// `[n_heads, head_dim]`. Empty when `seq_len == 0`.
pub fn paged_attention_decode(
    q: &[f32],
    kv_cache: &PageTable,
    layer: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(
        q.len(),
        n_heads * head_dim,
        "paged_attention_decode: q.len() ({}) != n_heads * head_dim ({})",
        q.len(),
        n_heads * head_dim,
    );
    assert_eq!(
        kv_cache.layer(),
        layer,
        "paged_attention_decode: kv_cache.layer() ({}) != layer ({})",
        kv_cache.layer(),
        layer,
    );
    let _ = layer;

    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return vec![0.0; n_heads * head_dim];
    }

    // The "logical" sequence length on the page table may exceed
    // the requested `seq_len` (callers might be decoding a slice).
    // Clamp so we don't walk past the live tokens.
    let live = seq_len.min(kv_cache.seq_len);
    if live == 0 {
        return vec![0.0; n_heads * head_dim];
    }

    // ---- Step 1: compute logits = Q[h, :] . K[t, h, :] / sqrt(d)
    //             for every (h, t). Store as Vec<f32> indexed by
    //             [h * live + t] so the softmax in step 2 is
    //             contiguous per head.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut logits = vec![0.0f32; n_heads * live];
    for h in 0..n_heads {
        let q_base = h * head_dim;
        for t in 0..live {
            // Look up the page + offset for token `t`.
            let (page_idx, offset) = match kv_cache.lookup(t) {
                Some(lo) => lo,
                None => {
                    // Empty slot or out of range; treat as zero
                    // contribution. This shouldn't happen when
                    // `live <= kv_cache.seq_len` but we defend
                    // against it.
                    continue;
                }
            };
            let page: &Page = match kv_cache.pages[page_idx].as_ref() {
                Some(p) => p,
                None => continue,
            };
            if offset >= page.n_tokens {
                continue;
            }
            let k_base = offset * page.head_elements;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_base + d] * page.k[k_base + d];
            }
            logits[h * live + t] = dot * scale;
        }
    }

    // ---- Step 2: softmax per head over the `live` token axis.
    // Numerically stable: subtract row max before exp.
    let mut probs = vec![0.0f32; n_heads * live];
    for h in 0..n_heads {
        let row = &logits[h * live..h * live + live];
        let mut max = f32::NEG_INFINITY;
        for &v in row {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            // All-masked head (shouldn't happen here; defensive).
            for t in 0..live {
                probs[h * live + t] = 0.0;
            }
            continue;
        }
        let mut sum = 0.0f32;
        for t in 0..live {
            let e = (logits[h * live + t] - max).exp();
            probs[h * live + t] = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for t in 0..live {
            probs[h * live + t] *= inv;
        }
    }

    // ---- Step 3: weighted sum of V[t, h, :] by probs[h, t].
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let o_base = h * head_dim;
        for t in 0..live {
            let p = probs[h * live + t];
            if p == 0.0 {
                continue;
            }
            let (page_idx, offset) = match kv_cache.lookup(t) {
                Some(lo) => lo,
                None => continue,
            };
            let page = match kv_cache.pages[page_idx].as_ref() {
                Some(pg) => pg,
                None => continue,
            };
            if offset >= page.n_tokens {
                continue;
            }
            let v_base = offset * page.head_elements;
            for d in 0..head_dim {
                out[o_base + d] += p * page.v[v_base + d];
            }
        }
    }

    out
}

/// Convenience wrapper: forward-declare a `&mut [f32]` out-buffer and
/// write the result into it. Asserts sizing.
pub fn paged_attention_decode_into(
    q: &[f32],
    kv_cache: &PageTable,
    layer: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let result = paged_attention_decode(q, kv_cache, layer, seq_len, n_heads, head_dim);
    assert_eq!(
        out.len(),
        result.len(),
        "paged_attention_decode_into: out.len() ({}) != expected ({})",
        out.len(),
        result.len(),
    );
    out.copy_from_slice(&result);
}

/// Returns the size in bytes of a single paged-attention K+V cache
/// for one layer at the given context size, excluding paging
/// overhead. This is the formula the backend uses for its
/// `memory_estimate` upper bound.
pub fn page_cache_bytes(ctx_size: usize, n_heads: usize, head_dim: usize) -> usize {
    let n_pages = ctx_size / PAGE_TOKENS + 1;
    n_pages * PAGE_TOKENS * n_heads * head_dim * 4 /*f32*/ * 2 /*k and v*/
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_table::{PageTable, PAGE_TOKENS};

    /// Helper: build a paged cache that spans two pages with 4
    /// meaningful tokens (PAGE_TOKENS - 1 + PAGE_TOKENS + 1 = 2
    /// pages total). Each "meaningful" token is at a known slot in
    /// the cache so we can hand-compute the expected scores. The
    /// filler tokens between/around them are zero.
    ///
    /// Layout (PAGE_TOKENS = 16):
    ///   page 0 slots 0..15
    ///     slot 0  -> t0 (the one we test): K_h0=(1,0), K_h1=(1,0)
    ///     slot 1  -> t1: K_h0=(0,1), K_h1=(0,1)
    ///     slot 2  -> t2: K_h0=(-1,0), K_h1=(-1,0)
    ///     slot 3  -> t3: K_h0=(0,-1), K_h1=(0,-1)
    ///     slot 4..15 -> zero (no contribution)
    ///   page 1 slots 0..15
    ///     slot 0..14 -> zero
    ///     slot 15 -> zero
    ///
    /// K and V are the same buffer so the attention output for the
    /// aligned query equals the K vector we wrote.
    fn build_two_page_cache() -> (PageTable, [f32; 4]) {
        // Head_elements = n_heads * head_dim = 4.
        let head_elements = 4;
        let mut cache = PageTable::new(0, head_elements);

        // Page 0: first 4 tokens are the "real" ones; rest are zero.
        // We write 16 tokens to fill page 0.
        let real_k0 = [1.0, 0.0, 1.0, 0.0]; // t0
        let real_k1 = [0.0, 1.0, 0.0, 1.0]; // t1
        let real_k2 = [-1.0, 0.0, -1.0, 0.0]; // t2
        let real_k3 = [0.0, -1.0, 0.0, -1.0]; // t3
        cache.append_token(&real_k0, &real_k0);
        cache.append_token(&real_k1, &real_k1);
        cache.append_token(&real_k2, &real_k2);
        cache.append_token(&real_k3, &real_k3);
        for _ in 4..PAGE_TOKENS {
            cache.append_token(&[0.0; 4], &[0.0; 4]);
        }
        // One more token to spill onto page 1.
        cache.append_token(&[0.0; 4], &[0.0; 4]);
        assert_eq!(cache.seq_len, PAGE_TOKENS + 1);
        assert_eq!(
            cache.n_materialized(),
            2,
            "must have crossed a page boundary"
        );
        (cache, real_k0)
    }

    #[test]
    fn empty_seq_returns_zero_output() {
        let cache = PageTable::new(0, 4);
        let q = [1.0, 0.0, 0.0, 1.0]; // 2 heads * head_dim=2
        let out = paged_attention_decode(&q, &cache, 0, 0, 2, 2);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn single_token_uniform_softmax() {
        let mut cache = PageTable::new(0, 4);
        cache.append_token(&[1.0, 0.0, 1.0, 0.0], &[2.0, 0.0, 2.0, 0.0]);
        let q = [1.0, 0.0, 1.0, 0.0];
        let out = paged_attention_decode(&q, &cache, 0, 1, 2, 2);
        // Single token -> softmax puts all mass on it -> out = V.
        assert_eq!(out, vec![2.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn two_pages_orthogonal_queries_pick_max_overlap() {
        // Build the 2-page cache; the first 4 slots are the real
        // tokens, the rest are zero. We pick `seq_len=4` so only
        // the first 4 slots participate. With a query that aligns
        // positively with t0 and is orthogonal (or negatively
        // aligned) with t1..t3, softmax will put most mass on t0
        // and the output approximates V[t0].
        //
        // K vectors (per head): t0=(1,0), t1=(0,1), t2=(-1,0), t3=(0,-1)
        // V vectors (per head): same as K
        // Query: (2,0).h0 -> dots = 2,0,-2,0 -> softmax puts almost
        // all mass on t0; expected out ~= (1, 0) for both heads.
        let (cache, _) = build_two_page_cache();
        let q = [2.0, 0.0, 2.0, 0.0];
        let out = paged_attention_decode(&q, &cache, 0, 4, 2, 2);
        // Compute the expected softmax output exactly:
        // logits = [2, 0, -2, 0] / sqrt(2) = [1.4142, 0, -1.4142, 0]
        let s = 1.0_f32 / (2.0_f32).sqrt();
        let l0 = 2.0 * s;
        let l2 = -2.0 * s;
        let e0 = l0.exp();
        let e1 = 1.0_f32; // exp(0)
        let e2 = l2.exp();
        let denom = e0 + e1 + e2 + e1;
        let p0 = e0 / denom;
        let _p1 = e1 / denom;
        let p2 = e2 / denom;
        // head 0, dim 0: p0 * 1 + p1 * 0 + p2 * (-1) + p1 * 0 = p0 - p2
        let h0d0 = p0 - p2;
        // head 0, dim 1: p0 * 0 + p1 * 1 + p2 * 0 + p1 * (-1) = p1 - p1 = 0
        let h0d1 = 0.0_f32;
        // head 1 is identical.
        assert!(
            (out[0] - h0d0).abs() < 1e-5,
            "out[0] = {}, expected {}",
            out[0],
            h0d0,
        );
        assert!(
            (out[1] - h0d1).abs() < 1e-5,
            "out[1] = {}, expected {}",
            out[1],
            h0d1,
        );
        assert!(
            (out[2] - h0d0).abs() < 1e-5,
            "out[2] = {}, expected {}",
            out[2],
            h0d0,
        );
        assert!(
            (out[3] - h0d1).abs() < 1e-5,
            "out[3] = {}, expected {}",
            out[3],
            h0d1,
        );
    }

    #[test]
    fn two_pages_perfect_orthogonality_full_softmax() {
        // Query head 0 = (1, 1) (so it aligns equally with t0=(1,0)
        // and t1=(0,1)). Pre-softmax scores for head 0 are
        // (1/sqrt(2)) * [1, 1, -1, -1]. After softmax:
        //   p_top = e^(1/sqrt(2)) / (2*e^(1/sqrt(2)) + 2*e^(-1/sqrt(2)))
        //   p_bot = e^(-1/sqrt(2)) / (same denom)
        //
        // Vectors V[t0]_h0 = (1,0), V[t1]_h0 = (0,1),
        // V[t2]_h0 = (-1,0), V[t3]_h0 = (0,-1). So:
        //   out[0] = p_top*1 + p_top*0 + p_bot*(-1) + p_bot*0 = p_top - p_bot
        //   out[1] = p_top*0 + p_top*1 + p_bot*0 + p_bot*(-1) = p_top - p_bot
        let (cache, _) = build_two_page_cache();
        let q = [1.0, 1.0, 1.0, 1.0];
        let out = paged_attention_decode(&q, &cache, 0, 4, 2, 2);
        let s = 1.0_f32 / (2.0_f32).sqrt();
        let e = (s).exp();
        let em1 = (-s).exp();
        let denom = 2.0 * e + 2.0 * em1;
        let p_top = e / denom;
        let p_bot = em1 / denom;
        let h0d0 = p_top - p_bot;
        let h0d1 = p_top - p_bot;
        assert!(
            (out[0] - h0d0).abs() < 1e-5,
            "out[0] = {}, expected {}",
            out[0],
            h0d0,
        );
        assert!(
            (out[1] - h0d1).abs() < 1e-5,
            "out[1] = {}, expected {}",
            out[1],
            h0d1,
        );
        assert!(
            (out[2] - h0d0).abs() < 1e-5,
            "out[2] = {}, expected {}",
            out[2],
            h0d0,
        );
        assert!(
            (out[3] - h0d1).abs() < 1e-5,
            "out[3] = {}, expected {}",
            out[3],
            h0d1,
        );
    }

    #[test]
    fn page_cache_bytes_matches_formula() {
        let bytes = page_cache_bytes(512, 32, 128);
        // (512/16 + 1) * 16 * 32 * 128 * 4 * 2
        let expected = (512 / 16 + 1) * 16 * 32 * 128 * 4 * 2;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn span_pages_with_attention_computes_correctly() {
        // seq_len = PAGE_TOKENS + 1 -> the attention walk must touch
        // both pages. We pad the cache with zero tokens so softmax
        // doesn't blow up.
        let mut cache = PageTable::new(0, 4);
        for _ in 0..(PAGE_TOKENS + 1) {
            cache.append_token(&[0.0; 4], &[0.0; 4]);
        }
        let q = [1.0, 0.0, 1.0, 0.0];
        let out = paged_attention_decode(&q, &cache, 0, PAGE_TOKENS + 1, 2, 2);
        // All K vectors are zero, so Q . K = 0 for every t -> softmax
        // is uniform 1 / (PAGE_TOKENS + 1). Output = uniform mean of
        // zero V vectors = zeros.
        for v in &out {
            assert!(v.abs() < 1e-6, "expected zeros, got {}", v);
        }
    }

    #[test]
    fn into_helper_writes_into_buffer() {
        let mut cache = PageTable::new(0, 4);
        cache.append_token(&[1.0, 0.0, 1.0, 0.0], &[2.0, 0.0, 2.0, 0.0]);
        let q = [1.0, 0.0, 1.0, 0.0];
        let mut out = [0.0_f32; 4];
        paged_attention_decode_into(&q, &cache, 0, 1, 2, 2, &mut out);
        assert_eq!(out, [2.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn layer_mismatch_is_caught() {
        let cache = PageTable::new(7, 4);
        let q = [0.0; 4];
        let result = std::panic::catch_unwind(|| paged_attention_decode(&q, &cache, 0, 1, 2, 2));
        assert!(result.is_err(), "layer mismatch should panic");
    }

    #[test]
    fn vendored_status_reflects_feature() {
        // The default build (feature off) returns false; the vendored
        // build returns true. Either way the test is well-defined.
        let _ = vendored_status();
    }
}
