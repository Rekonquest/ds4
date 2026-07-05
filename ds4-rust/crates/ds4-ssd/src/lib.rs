//
// Faithful port of upstream `ds4_ssd.c` (181 LoC) + the static
// hotlist (`ds4_streaming_hotlist.inc`). The hotlist is a
// `[layer, expert]` array baked at compile time; we parse it at
// runtime into a histogram that drives prefetch order.
//
// Layout:
//   * `hotlist`     â€” `.inc` parser + histogram + top-N helpers.
//   * `expert_cache`â€” per-expert LRU keyed by `(layer, expert)` with
//                     byte-budget and entry-count eviction.
//   * `stream`      â€” `SsdStreamer` that records preloads and drives
//                     eviction and cache residency bookkeeping.

pub const CRATE_NAME: &str = "ds4-ssd";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod expert_cache;
pub mod hotlist;
pub mod stream;

pub use expert_cache::{ExpertCache, ExpertKey};
pub use hotlist::{Hotlist, Pair as HotlistPair};
pub use stream::{SsdOptions, SsdStreamer};

/// Error type for hotlist parsing. Lives at the crate root so
/// `hotlist.rs` can import it via `use super::HotlistError;`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotlistError {
    Parse { line: usize, message: String },
}

impl std::fmt::Display for HotlistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HotlistError::Parse { line, message } => {
                write!(f, "hotlist parse error at line {}: {}", line, message)
            }
        }
    }
}

impl std::error::Error for HotlistError {}

/// Parse a "0:42" style `layer:expert` argument.
pub fn parse_layer_expert(s: &str) -> Result<(u32, u32), String> {
    let (l, e) = s
        .split_once(':')
        .ok_or_else(|| format!("expected LAYER:EXPERT, got {s:?}"))?;
    let layer: u32 = l.parse().map_err(|e| format!("bad layer {l:?}: {e}"))?;
    let expert: u32 = e.parse().map_err(|e| format!("bad expert {e:?}: {e}"))?;
    Ok((layer, expert))
}

/// Parse a `1g` / `512m` / `12345` byte-budget argument (matches the
/// upstream `ds4_parse_gib_arg` + `ds4_parse_streaming_cache_experts_arg`).
/// If the string ends in `g`/`gb`/`G`/`GB`, the leading digits are
/// interpreted as gibibytes. Otherwise the whole string is a plain
/// non-negative integer count of experts.
pub fn parse_cache_budget(s: &str) -> Result<CacheBudget, String> {
    if s.is_empty() {
        return Err("empty budget argument".to_string());
    }
    let lower = s.to_ascii_lowercase();
    if lower.ends_with("gb") || lower.ends_with('g') {
        let end = if lower.ends_with("gb") {
            s.len() - 2
        } else {
            s.len() - 1
        };
        if end == 0 {
            return Err(format!("missing digits before 'g' in {s:?}"));
        }
        if !s[..end].chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("non-digit in {s:?}"));
        }
        let digits: u64 = s[..end].parse().map_err(|e| format!("{s:?}: {e}"))?;
        if digits == 0 {
            return Err(format!("zero bytes in {s:?}"));
        }
        let bytes = digits * 1024 * 1024 * 1024;
        Ok(CacheBudget::Bytes(bytes))
    } else if s.chars().all(|c| c.is_ascii_digit()) {
        let v: u64 = s.parse().map_err(|e| format!("{s:?}: {e}"))?;
        Ok(CacheBudget::ExpertCount(v as u32))
    } else {
        Err(format!("unrecognised budget {s:?}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBudget {
    ExpertCount(u32),
    Bytes(u64),
}

/// Compute the auto-plan, mirroring
/// `ds4_ssd_auto_cache_plan(recommended_bytes, non_routed_bytes,
/// per_expert_bytes, max_model_experts)`.
pub fn auto_cache_plan(
    recommended_bytes: u64,
    non_routed_bytes: u64,
    per_expert_bytes: u64,
    max_model_experts: u64,
) -> CachePlan {
    if recommended_bytes == 0 || per_expert_bytes == 0 {
        return CachePlan::default();
    }
    let model_target_bytes = (recommended_bytes / 5).saturating_mul(4);
    let cache_bytes = model_target_bytes.saturating_sub(non_routed_bytes);
    let mut cache_experts = cache_bytes / per_expert_bytes;
    if cache_experts == 0 {
        cache_experts = 1;
    }
    if max_model_experts != 0 && cache_experts > max_model_experts {
        cache_experts = max_model_experts;
    }
    let effective_cache_bytes = cache_experts.saturating_mul(per_expert_bytes);
    CachePlan {
        model_target_bytes,
        cache_bytes,
        effective_cache_bytes,
        cache_experts: cache_experts.min(u32::MAX as u64) as u32,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CachePlan {
    pub model_target_bytes: u64,
    pub cache_bytes: u64,
    pub effective_cache_bytes: u64,
    pub cache_experts: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-ssd");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn hotlist_get_returns_zero_for_missing() {
        let h = Hotlist::empty(4, 8);
        assert_eq!(h.get(0, 0), 0);
        assert_eq!(h.get(99, 99), 0);
    }

    #[test]
    fn parse_cache_budget_bytes() {
        assert_eq!(
            parse_cache_budget("1g").unwrap(),
            CacheBudget::Bytes(1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_cache_budget("2GB").unwrap(),
            CacheBudget::Bytes(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_cache_budget("8g").unwrap(),
            CacheBudget::Bytes(8 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn parse_cache_budget_experts() {
        assert_eq!(
            parse_cache_budget("32").unwrap(),
            CacheBudget::ExpertCount(32)
        );
    }

    #[test]
    fn parse_layer_expert_strings() {
        assert_eq!(parse_layer_expert("0:42").unwrap(), (0, 42));
        assert!(parse_layer_expert("foo").is_err());
        assert!(parse_layer_expert("0:bar").is_err());
    }

    #[test]
    fn auto_cache_plan_matches_upstream_heuristic() {
        // 24 GiB recommended, 8 GiB non-routed, 16 MiB per expert,
        // 256 max experts.
        let plan = auto_cache_plan(
            24 * 1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024,
            256,
        );
        assert_eq!(plan.model_target_bytes, (24 * 1024 * 1024 * 1024 / 5) * 4);
        assert!(plan.cache_bytes > 0);
        assert_eq!(
            plan.effective_cache_bytes,
            plan.cache_experts as u64 * 16 * 1024 * 1024
        );
        assert!(plan.cache_experts > 0);
    }
}
