// DS4 (DwarfStar) — per-expert LRU.
//
// The expert cache tracks `(layer, expert)` weight slabs loaded from
// SSD. Eviction is keyed by `(layer, expert)` and bounded by a
// configured byte budget plus an optional entry-count cap. Access
// count is tracked per entry to support profile-aware preloading.
//
// The C implementation lives in `ds4_ssd.c` (`ds4_ssd_cache_plan`
// and the auto-plan heuristic). The Rust port keeps the same
// byte-budget + entry-count semantics but exposes them on a per-entry
// `touch/evict` API instead of the C `ds4_ssd_memory_lock` flow.

use std::collections::HashMap;

use parking_lot::Mutex;

/// One expert slot key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    bytes: u64,
    access_count: u64,
    /// Position in the LRU `VecDeque` for O(1) removal on touch.
    pos: usize,
}

/// LRU keyed by `(layer, expert)`. Eviction respects both a
/// configured byte budget (over the whole cache) and an entry-count
/// cap (default `usize::MAX` when the caller doesn't care).
#[derive(Debug)]
pub struct ExpertCache {
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    cap_entries: usize,
    cap_bytes: u64,
    approx_bytes: u64,
    by_key: HashMap<ExpertKey, Entry>,
    order: Vec<ExpertKey>,
    live_bytes: u64,
}

impl ExpertCache {
    /// Build a cache with only an entry-count cap.
    pub fn new(cap_entries: usize) -> Self {
        Self::with_caps(cap_entries, usize::MAX, 0)
    }

    /// Build a cache bounded by a byte budget. If
    /// `approx_bytes_per_expert > 0`, we use it as a fallback when the
    /// caller doesn't know an entry's exact size.
    pub fn with_byte_budget(cap_bytes: usize, approx_bytes_per_expert: usize) -> Self {
        Self::with_caps(usize::MAX, cap_bytes, approx_bytes_per_expert)
    }

    /// Build a cache with both budgets active.
    pub fn with_caps(cap_entries: usize, cap_bytes: usize, approx_bytes_per_expert: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                cap_entries,
                cap_bytes: cap_bytes as u64,
                approx_bytes: approx_bytes_per_expert as u64,
                by_key: HashMap::new(),
                order: Vec::new(),
                live_bytes: 0,
            }),
        }
    }

    /// Touch `(layer, expert)`, refreshing its position at the back
    /// of the LRU. `bytes` is the entry's on-memory size (or 0 to use
    /// the cache-wide approximation).
    pub fn touch(&self, layer: u32, expert: u32, bytes: u64) {
        let key = ExpertKey::new(layer, expert);
        let mut g = self.inner.lock();
        let approx = g.approx_bytes;
        let effective_bytes = if bytes == 0 { approx } else { bytes };
        if let Some(existing) = g.by_key.remove(&key) {
            // Subtract old bytes, fix LRU order, then re-insert with
            // refreshed metadata at the back.
            g.live_bytes = g.live_bytes.saturating_sub(existing.bytes);
            let pos = existing.pos;
            // Remove `key` from `g.order` at `pos`. Splice-out is
            // cheaper than VecDeque when the order vector is plain.
            g.order.remove(pos);
            // Subsequent entries shift down by one. Collect keys
            // first so we can take fresh mutable borrows.
            let shifted: Vec<ExpertKey> = g.order.iter().skip(pos).copied().collect();
            for k in shifted {
                if let Some(e) = g.by_key.get_mut(&k) {
                    e.pos -= 1;
                }
            }
            let new_pos = g.order.len();
            g.order.push(key);
            g.by_key.insert(
                key,
                Entry {
                    bytes: effective_bytes,
                    access_count: existing.access_count.saturating_add(1),
                    pos: new_pos,
                },
            );
        } else {
            let pos = g.order.len();
            g.order.push(key);
            g.by_key.insert(
                key,
                Entry {
                    bytes: effective_bytes,
                    access_count: 1,
                    pos,
                },
            );
        }
        g.live_bytes = g.live_bytes.saturating_add(effective_bytes);
    }

    /// Pop the oldest entry, returning its `(layer, expert)`. Returns
    /// `None` when the cache is empty.
    pub fn evict_one(&self) -> Option<(u32, u32)> {
        let mut g = self.inner.lock();
        let key = g.order.first().copied()?;
        if let Some(entry) = g.by_key.remove(&key) {
            g.live_bytes = g.live_bytes.saturating_sub(entry.bytes);
        }
        g.order.remove(0);
        // Fix up positions: every remaining entry's `pos` shifts down
        // by one. Rebuild the map in one shot to avoid overlapping
        // borrows of `g.order` and `g.by_key`.
        let entries: Vec<(ExpertKey, Entry)> = g
            .by_key
            .drain()
            .map(|(k, mut e)| {
                // `pos` no longer meaningful; we'll reassign below.
                e.pos = 0;
                (k, e)
            })
            .collect();
        for (k, mut e) in entries {
            if let Some(idx) = g.order.iter().position(|kk| *kk == k) {
                e.pos = idx;
            }
            g.by_key.insert(k, e);
        }
        Some((key.layer, key.expert))
    }

    /// Drop entries until the cache fits in both budgets.
    pub fn evict_to_fit(&self) -> Vec<(u32, u32)> {
        let mut evicted = Vec::new();
        loop {
            let (over_entries, over_bytes) = {
                let g = self.inner.lock();
                (g.order.len() > g.cap_entries, g.live_bytes > g.cap_bytes)
            };
            if !(over_entries || over_bytes) {
                break;
            }
            match self.evict_one() {
                Some(k) => evicted.push(k),
                None => break,
            }
        }
        evicted
    }

    pub fn len(&self) -> usize {
        self.inner.lock().order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().order.is_empty()
    }

    /// Total live bytes across all entries.
    pub fn live_bytes(&self) -> u64 {
        self.inner.lock().live_bytes
    }

    /// Configured entry-count cap.
    pub fn configured_count(&self) -> usize {
        self.inner.lock().cap_entries
    }

    /// Currently resident entry count.
    pub fn current_count(&self) -> usize {
        self.inner.lock().order.len()
    }

    /// Configured byte budget.
    pub fn configured_bytes(&self) -> u64 {
        self.inner.lock().cap_bytes
    }

    /// Number of times `(layer, expert)` has been touched.
    pub fn access_count(&self, layer: u32, expert: u32) -> Option<u64> {
        let key = ExpertKey::new(layer, expert);
        self.inner.lock().by_key.get(&key).map(|e| e.access_count)
    }

    /// Convenience: parse `<layer>:<expert>` strings ("0:42") and
    /// touch them in order.
    pub fn touch_many<I>(&self, items: I)
    where
        I: IntoIterator<Item = (u32, u32)>,
    {
        for (l, e) in items {
            self.touch(l, e, 0);
        }
    }

    /// Returns true when a `touch(bytes)` for an unseen entry would
    /// push us over either budget. Useful as a fast precheck before
    /// disk I/O.
    pub fn is_over_budget(&self, extra_bytes: u64) -> bool {
        let g = self.inner.lock();
        if g.order.len() + 1 > g.cap_entries {
            return true;
        }
        g.live_bytes.saturating_add(extra_bytes) > g.cap_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_reports_zero() {
        let c = ExpertCache::new(4);
        assert_eq!(c.len(), 0);
        assert_eq!(c.configured_count(), 4);
        assert_eq!(c.current_count(), 0);
        assert!(c.evict_one().is_none());
    }

    #[test]
    fn evicts_in_insertion_order() {
        let c = ExpertCache::new(2);
        c.touch(0, 1, 0);
        c.touch(0, 2, 0);
        c.touch(0, 3, 0);
        // Cap is 2, so the third touch forced an evict in our policy.
        // We do not auto-evict here; the caller is expected to call
        // `evict_to_fit`. So len() is still 3 after three touches.
        assert_eq!(c.len(), 3);
        let mut evicted = c.evict_to_fit();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted.pop(), Some((0, 1)));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn refresh_promotes_to_back() {
        let c = ExpertCache::new(3);
        c.touch(0, 1, 0);
        c.touch(0, 2, 0);
        c.touch(0, 3, 0);
        c.touch(0, 1, 0); // refresh
        let evicted = c.evict_to_fit();
        // After refresh, (0,1) is at the back. Eviction only happens
        // when over budget, and we are still at 3/3 — nothing should
        // happen.
        assert!(evicted.is_empty());
        c.touch(0, 4, 0); // now over cap by 1
        let evicted = c.evict_to_fit();
        assert_eq!(evicted, vec![(0, 2)]);
    }

    #[test]
    fn byte_budget_evicts_oldest() {
        let c = ExpertCache::with_byte_budget(100, 0);
        c.touch(0, 1, 40);
        c.touch(0, 2, 40);
        assert!(!c.is_over_budget(20));
        assert!(c.is_over_budget(30));
        // Adding a third 30-byte expert pushes us over the byte cap;
        // the LRU drops the oldest, leaving one 40-byte expert.
        c.touch(0, 3, 30);
        let evicted = c.evict_to_fit();
        assert_eq!(evicted, vec![(0, 1)]);
        assert_eq!(c.live_bytes(), 70);
    }

    #[test]
    fn access_count_increments() {
        let c = ExpertCache::new(8);
        c.touch(0, 1, 0);
        c.touch(0, 1, 0);
        c.touch(0, 1, 0);
        assert_eq!(c.access_count(0, 1), Some(3));
        assert_eq!(c.access_count(9, 9), None);
    }
}
