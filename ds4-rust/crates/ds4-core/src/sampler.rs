// DS4 (DwarfStar) — top-p / min-p / top-k sampler.
//
// Faithful Rust port of `sample_top_p_min_p` in `ds4.c:22706..22685`.
// Implements:
//
//   * Temperature scaling.
//   * Top-k filtering (kept indices sorted descending by logit).
//   * Min-p filtering: drop candidates below `min_p * max_prob`.
//   * Top-p filtering: take the smallest prefix whose cumulative
//     normalized probability reaches `top_p`.
//   * Final categorical sample from the surviving set.
//
// Reference: `ds4.c:22706..22765` (`sample_top_p_min_p`) and
// `ds4.c:22617..22703` (`sample_full_vocab`). The behaviour for
// `top_k == 0` matches `sample_full_vocab` (no top-k cap), and the
// `temperature <= 0` short-circuit returns the argmax, matching the C
// implementation's contract.
//
// Numeric choices:
//   * The C code keeps the un-renormalized exponential weights so the
//     top-p filter operates on `filtered_sum / sum >= top_p`. We
//     follow the same convention so the rounding error profile is
//     identical.
//   * Finite-only filtering: any non-finite logit is dropped silently
//     (matches the C `if (!isfinite(v)) continue`).

use rand::RngCore;

/// Sample one token id from `logits` using top-k / min-p / top-p
/// filtering. `logits` length is the vocab size.
///
/// Parameters:
///   * `temperature` — softmax temperature; `<= 0` short-circuits to
///     argmax.
///   * `top_k`       — keep at most this many candidates. `0` means
///     "no top-k cap".
///   * `top_p`       — nucleus sampling cumulative-probability cutoff.
///     Values outside `(0, 1]` are clamped to 1.
///   * `min_p`       — drop candidates below `min_p * max_prob`. Values
///     below 0 are clamped to 0.
///   * `rng`         — any `RngCore`; we only consume `f32`s from it.
///
/// Returns the sampled token id. Vocab-size 0 returns 0.
pub fn sample_top_p_min_p(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut dyn RngCore,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let top_p = if top_p <= 0.0 || top_p > 1.0 {
        1.0
    } else {
        top_p
    };
    let min_p = if min_p < 0.0 { 0.0 } else { min_p };

    // Step 1: find top-k finite candidates. We use an inline insertion
    // sort of length `top_k` so we don't have to materialize the full
    // vocab as a sortable pair vector (the C code caps at 1024).
    let top_k = if top_k == 0 {
        logits.len()
    } else if top_k > 1024 {
        1024.min(logits.len())
    } else {
        top_k.min(logits.len())
    };

    let mut ids: Vec<u32> = Vec::with_capacity(top_k);
    let mut vals: Vec<f32> = Vec::with_capacity(top_k);
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if ids.len() < top_k {
            let mut j = ids.len();
            ids.push(0);
            vals.push(0.0);
            while j > 0 && vals[j - 1] < v {
                ids[j] = ids[j - 1];
                vals[j] = vals[j - 1];
                j -= 1;
            }
            ids[j] = i as u32;
            vals[j] = v;
        } else if v > vals[top_k - 1] {
            // Replace the worst and re-bubble.
            let mut j = top_k - 1;
            while j > 0 && vals[j - 1] < v {
                ids[j] = ids[j - 1];
                vals[j] = vals[j - 1];
                j -= 1;
            }
            ids[j] = i as u32;
            vals[j] = v;
        }
    }
    if ids.is_empty() {
        return argmax(logits);
    }

    // Step 2: softmax with max-trick for numerical stability.
    let max_logit = vals[0];
    let mut probs: Vec<f32> = Vec::with_capacity(ids.len());
    let mut sum = 0.0f32;
    for &v in &vals {
        let p = ((v - max_logit) / temperature).exp();
        probs.push(p);
        sum += p;
    }
    if !matches!(sum.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) || !sum.is_finite() {
        return ids[0];
    }

    // Step 3: min-p filter on the normalized distribution, then top-p.
    let max_prob = probs[0] / sum;
    let min_prob_cutoff = max_prob * min_p;
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    for (i, &p_raw) in probs.iter().enumerate() {
        let p_norm = p_raw / sum;
        if i > 0 && p_norm < min_prob_cutoff {
            break;
        }
        filtered_sum += p_raw;
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    if filtered == 0 {
        return ids[0];
    }

    // Step 4: categorical sample. `rng` is only consumed via `f32()`.
    let mut u: f32 = rng.next_u32() as f32 / u32::MAX as f32;
    if !u.is_finite() {
        u = 0.0;
    }
    let mut r = u * filtered_sum;
    for i in 0..filtered {
        r -= probs[i];
        if r <= 0.0 {
            return ids[i];
        }
    }
    ids[filtered - 1]
}

/// Argmax over `logits`. Returns 0 for an empty slice; non-finite
/// values are skipped (matches the C `sample_argmax`).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

/// Argmax with one token excluded. Returns 0 when the vocab is empty
/// or when every non-excluded logit is non-finite.
pub fn argmax_excluding(logits: &[f32], excluded: u32) -> u32 {
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if i as u32 == excluded {
            continue;
        }
        if !v.is_finite() {
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

/// Compute the log-softmax of `logits` in-place. Vocab-size 0 is a
/// no-op. Used by `top_logprobs` and `token_logprob` on the session.
pub fn log_softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let mut max_v = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v.is_finite() && v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        if v.is_finite() {
            *v = (*v - max_v).exp();
            sum += *v;
        } else {
            *v = 0.0;
        }
    }
    let log_z = sum.max(f32::MIN_POSITIVE).ln();
    for v in logits.iter_mut() {
        *v = v.max(f32::MIN_POSITIVE).ln() - log_z;
    }
}

/// Compute the top-`k` log-probabilities from a freshly-decoded
/// logits slice. Writes (`token_id`, `logprob`) pairs into `out`;
/// returns the number of entries written.
///
/// The caller is expected to pass the live `session.logits` (not a
/// log-softmaxed copy); this helper applies log_softmax internally
/// and discards the modified copy.
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if k == 0 || logits.is_empty() {
        return Vec::new();
    }
    let k = k.min(logits.len());
    let mut work = logits.to_vec();
    log_softmax_inplace(&mut work);
    // Selection sort on a [0..k] window — fine for k up to ~256.
    let mut best_ids: Vec<u32> = Vec::with_capacity(k);
    let mut used = vec![false; work.len()];
    for _ in 0..k {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in work.iter().enumerate() {
            if used[i] {
                continue;
            }
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        used[best_i] = true;
        best_ids.push(best_i as u32);
    }
    best_ids
        .into_iter()
        .map(|id| (id, work[id as usize]))
        .collect()
}

/// Log-probability of a single token id. Returns 0.0 for an out-of-range
/// id or for an all-non-finite logits slice.
pub fn token_logprob(logits: &[f32], token: u32) -> f32 {
    let mut work = logits.to_vec();
    log_softmax_inplace(&mut work);
    match work.get(token as usize) {
        Some(v) if v.is_finite() => *v,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn argmax_returns_first_max() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[f32::NAN, f32::NEG_INFINITY, 1.0]), 2);
    }

    #[test]
    fn argmax_excluding_skips_token() {
        assert_eq!(argmax_excluding(&[1.0, 5.0, 3.0], 1), 2);
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let mut rng = StdRng::seed_from_u64(0);
        let sampled = sample_top_p_min_p(&[0.0, 5.0, 1.0], 0.0, 0, 1.0, 0.0, &mut rng);
        assert_eq!(sampled, 1);
    }

    #[test]
    fn top_k_one_picks_argmax() {
        let mut rng = StdRng::seed_from_u64(42);
        let sampled = sample_top_p_min_p(
            &[0.1, 0.2, 5.0, 0.3, 0.4],
            1.0,
            1,   // top_k = 1
            1.0, // top_p = 1
            0.0, // min_p = 0
            &mut rng,
        );
        assert_eq!(sampled, 2);
    }

    #[test]
    fn log_softmax_is_normalized() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        log_softmax_inplace(&mut v);
        let z: f32 = v.iter().map(|x| x.exp()).sum();
        assert!((z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn token_logprob_matches_top() {
        let logits = [1.0, 2.0, 3.0];
        let lp = token_logprob(&logits, 2);
        let top = top_logprobs(&logits, 3);
        assert!((lp - top[0].1).abs() < 1e-6);
    }

    #[test]
    fn top_logprobs_returns_k_in_descending_order() {
        let logits = [0.1, 0.5, 0.2, 0.8, 0.3];
        let top = top_logprobs(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 3);
        assert_eq!(top[1].0, 1);
        assert_eq!(top[2].0, 4);
    }

    #[test]
    fn sample_is_deterministic_with_seed() {
        let logits = vec![0.1f32; 16];
        let mut rng1 = StdRng::seed_from_u64(7);
        let mut rng2 = StdRng::seed_from_u64(7);
        let s1 = sample_top_p_min_p(&logits, 1.0, 4, 0.9, 0.05, &mut rng1);
        let s2 = sample_top_p_min_p(&logits, 1.0, 4, 0.9, 0.05, &mut rng2);
        assert_eq!(s1, s2);
    }
}
