// DS4 (DwarfStar) — disk LRU for the kv-store.
//
// The C side uses a `ds4_kvstore` that tracks an entry list ordered by
// "first evict the entry with the lowest score". The eviction policy
// itself (score formula, continued-prefix waypoints, anchor reasons)
// lives in `ds4_kvstore.c`; here we only model the byte-budgeted
// "pop the oldest entry until we fit" bookkeeping that the Rust side
// needs in front of that policy.

use std::collections::VecDeque;

/// An entry in the LRU. The key is the SHA1 hex string naming the file
/// on disk; `bytes` is its on-disk size used for byte-budget accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: String,
    pub bytes: u64,
}

#[derive(Debug)]
pub struct Lru {
    cap_entries: usize,
    cap_bytes: u64,
    approx_entry_bytes: u64,
    inner: VecDeque<Entry>,
    live_bytes: u64,
}

impl Lru {
    /// Count-based LRU with the previous semantics (`cap` entries).
    pub fn new(cap: usize) -> Self {
        Self {
            cap_entries: cap,
            cap_bytes: u64::MAX,
            approx_entry_bytes: 0,
            inner: VecDeque::with_capacity(cap),
            live_bytes: 0,
        }
    }

    /// Byte-budgeted LRU. `cap_bytes` is the on-disk byte budget for the
    /// whole store; `approx_entry_bytes` is used as an entry-count
    /// fallback when callers don't know exact file sizes yet.
    ///
    /// If `approx_entry_bytes == 0`, only the byte budget applies.
    pub fn bytes_capacity(cap_bytes: usize, approx_entry_bytes: usize) -> Self {
        Self {
            cap_entries: usize::MAX,
            cap_bytes: cap_bytes as u64,
            approx_entry_bytes: approx_entry_bytes as u64,
            inner: VecDeque::new(),
            live_bytes: 0,
        }
    }

    /// Both budgets together. Useful when the caller wants a hard cap
    /// on the entry count plus a byte budget.
    pub fn with_caps(cap_entries: usize, cap_bytes: usize, approx_entry_bytes: usize) -> Self {
        Self {
            cap_entries,
            cap_bytes: cap_bytes as u64,
            approx_entry_bytes: approx_entry_bytes as u64,
            inner: VecDeque::with_capacity(cap_entries.min(1024)),
            live_bytes: 0,
        }
    }

    /// Record a touch of `key`. If `key` was already present, refreshes
    /// it (moves to the back) and updates its byte count.
    pub fn touch(&mut self, key: String, bytes: u64) {
        let bytes = self.effective_bytes(bytes);
        if let Some(pos) = self.inner.iter().position(|e| e.key == key) {
            if let Some(old) = self.inner.remove(pos) {
                self.live_bytes = self.live_bytes.saturating_sub(old.bytes);
            }
        }
        self.inner.push_back(Entry { key, bytes });
        self.live_bytes = self.live_bytes.saturating_add(bytes);
    }

    /// Pop the oldest entry, returning its `(key, bytes)`. Updates the
    /// live byte counter accordingly. Returns `None` when the LRU is
    /// empty.
    pub fn evict_next(&mut self) -> Option<Entry> {
        let entry = self.inner.pop_front()?;
        self.live_bytes = self.live_bytes.saturating_sub(entry.bytes);
        Some(entry)
    }

    /// Current number of tracked entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Current live byte total.
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    pub fn cap_entries(&self) -> usize {
        self.cap_entries
    }

    fn effective_bytes(&self, bytes: u64) -> u64 {
        if bytes == 0 && self.approx_entry_bytes != 0 {
            self.approx_entry_bytes
        } else {
            bytes
        }
    }

    /// Returns true if accepting `extra_bytes` would push us over either
    /// the entry-count or byte budget.
    pub fn is_over_budget(&self, extra_bytes: u64) -> bool {
        let extra_bytes = self.effective_bytes(extra_bytes);
        if self.inner.len() + 1 > self.cap_entries {
            return true;
        }
        self.live_bytes.saturating_add(extra_bytes) > self.cap_bytes
    }

    /// Returns the oldest entry that should be evicted to make room for
    /// `extra_bytes`. If we're already within budget, returns `None`.
    pub fn next_eviction_for(&self, extra_bytes: u64) -> Option<&Entry> {
        if self.inner.len() + 1 > self.cap_entries {
            return self.inner.front();
        }
        if self.live_bytes.saturating_add(extra_bytes) > self.cap_bytes {
            return self.inner.front();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_lru_evicts_in_order() {
        let mut lru = Lru::new(3);
        lru.touch("a".into(), 1);
        lru.touch("b".into(), 1);
        lru.touch("c".into(), 1);
        lru.touch("a".into(), 1); // refresh "a"
        assert_eq!(lru.evict_next().unwrap().key, "b");
        assert_eq!(lru.evict_next().unwrap().key, "c");
        assert_eq!(lru.evict_next().unwrap().key, "a");
        assert_eq!(lru.evict_next(), None);
    }

    #[test]
    fn byte_budget_evicts_oldest() {
        // cap 100 bytes, no entry-count cap.
        let mut lru = Lru::bytes_capacity(100, 0);
        lru.touch("a".into(), 40);
        lru.touch("b".into(), 40);
        // Both fit; trying to add 30 should evict "a".
        assert!(lru.is_over_budget(30));
        let next = lru.next_eviction_for(30).unwrap().clone();
        assert_eq!(next.key, "a");
        lru.evict_next();
        // Now we have 40 bytes live; 30 more fits.
        assert!(!lru.is_over_budget(30));
    }

    #[test]
    fn touch_refresh_replaces_old_bytes() {
        let mut lru = Lru::new(8);
        lru.touch("a".into(), 100);
        lru.touch("a".into(), 200);
        assert_eq!(lru.live_bytes(), 200);
        assert_eq!(lru.len(), 1);
    }
}
