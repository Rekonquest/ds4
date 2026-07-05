// DS4 (DwarfStar) â€” SSD streamer.
//
// Schedules expert weight preloads in response to the
// `(layer, expert)` stream emitted by the inference graph. The
// upstream code lives in `ds4_ssd.c` + the auto-plan helper, with
// the runtime loop inlined in the C engine. Here we expose:
//
//   * `SsdStreamer::preload(layer, experts)` - record what the
//     runtime wants resident and drive the LRU eviction policy.
//
// Preload records cache residency intent and applies the same LRU
// budget rules the runtime needs before issuing backend-specific disk
// reads.

use std::path::PathBuf;

use crate::expert_cache::ExpertCache;

#[derive(Debug, Clone)]
pub struct SsdOptions {
    pub model_root: PathBuf,
    pub cache_experts: usize,
    pub cache_bytes: usize,
    pub preload_experts: usize,
    pub cold: bool,
}

impl Default for SsdOptions {
    fn default() -> Self {
        Self {
            model_root: PathBuf::new(),
            cache_experts: 0,
            cache_bytes: 0,
            preload_experts: 0,
            cold: false,
        }
    }
}

#[derive(Debug)]
pub struct SsdStreamer {
    opts: SsdOptions,
    cache: ExpertCache,
    /// Last expert id touched per layer, used as a tiebreaker for
    /// cold-mode preloads.
    last_expert_by_layer: std::collections::HashMap<u32, u32>,
    /// Last `preload` call's plan (for diagnostics / testing).
    last_plan: Vec<(u32, u32)>,
}

impl SsdStreamer {
    pub fn new(opts: SsdOptions) -> Self {
        let cache = if opts.cache_bytes > 0 {
            ExpertCache::with_caps(opts.cache_experts.max(1), opts.cache_bytes, 0)
        } else if opts.cache_experts > 0 {
            ExpertCache::new(opts.cache_experts)
        } else {
            ExpertCache::new(0)
        };
        Self {
            opts,
            cache,
            last_expert_by_layer: std::collections::HashMap::new(),
            last_plan: Vec::new(),
        }
    }

    pub fn options(&self) -> &SsdOptions {
        &self.opts
    }

    pub fn cache(&self) -> &ExpertCache {
        &self.cache
    }

    /// Mark experts `(layer, experts...)` as the next ones we want
    /// resident. Updates the cache LRU and returns the eviction
    /// candidates the cache just dropped (so the caller can drop
    /// their weight slabs).
    pub fn preload(&mut self, layer: u32, experts: &[u32]) -> Vec<(u32, u32)> {
        let mut plan: Vec<(u32, u32)> = Vec::with_capacity(experts.len());
        for &expert in experts {
            self.cache.touch(layer, expert, 0);
            self.last_expert_by_layer.insert(layer, expert);
            plan.push((layer, expert));
        }
        self.last_plan = plan.clone();
        // Eviction happens synchronously after every preload to keep
        // the in-memory LRU honest. Callers wanting back-pressure can
        // inspect the returned eviction list.
        self.cache.evict_to_fit()
    }

    /// Convenience for the CLI / server hot paths: preload a slice
    /// of `(layer, expert)` pairs directly.
    pub fn preload_pairs(&mut self, pairs: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let mut evicted = Vec::new();
        let by_layer = pair_buckets(pairs);
        for (layer, experts) in by_layer {
            evicted.extend(self.preload(layer, &experts));
        }
        evicted
    }

    /// The most recent preload plan.
    pub fn last_plan(&self) -> &[(u32, u32)] {
        &self.last_plan
    }

    /// True when streamer was constructed with `cold: true`.
    pub fn is_cold(&self) -> bool {
        self.opts.cold
    }

    /// `preload_experts` knob from the options.
    pub fn preload_window(&self) -> usize {
        self.opts.preload_experts
    }

    /// Drop everything in the LRU.
    pub fn clear_cache(&mut self) {
        while self.cache.evict_one().is_some() {}
    }
}

fn pair_buckets(pairs: &[(u32, u32)]) -> std::collections::BTreeMap<u32, Vec<u32>> {
    let mut out: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for &(l, e) in pairs {
        out.entry(l).or_default().push(e);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preload_updates_lru() {
        let mut s = SsdStreamer::new(SsdOptions {
            cache_experts: 4,
            ..SsdOptions::default()
        });
        let evicted = s.preload(0, &[1, 2, 3]);
        assert!(evicted.is_empty());
        assert_eq!(s.cache().current_count(), 3);
    }

    #[test]
    fn preload_pairs_groups_by_layer() {
        let mut s = SsdStreamer::new(SsdOptions {
            cache_experts: 8,
            ..SsdOptions::default()
        });
        s.preload_pairs(&[(0, 1), (1, 2), (0, 3), (1, 4)]);
        assert_eq!(s.cache().current_count(), 4);
        // The last `preload` call was for layer 1.
        assert_eq!(s.last_plan(), &[(1, 2), (1, 4)]);
    }

    #[test]
    fn cache_evicts_when_full() {
        let mut s = SsdStreamer::new(SsdOptions {
            cache_experts: 2,
            ..SsdOptions::default()
        });
        s.preload(0, &[1, 2]);
        let evicted = s.preload(0, &[3]);
        assert_eq!(evicted, vec![(0, 1)]);
    }
}
