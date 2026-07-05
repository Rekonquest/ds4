// DS4 (DwarfStar) — disk KV prefix index.
//
// The C side (`ds4_kvstore.c`) uses SHA1 hex strings for cache-file
// names and for "did we already see this prefix?" membership tests.
// We keep the same linear-scan `HashSet<String>` shape rather than a
// rax tree: rax is reserved for the agent path (next revision), and the kv-store
// path explicitly does not use it upstream.

use std::collections::HashSet;

use sha1::{Digest, Sha1};

/// Lowercase hex SHA1 of `bytes`, length 40, no allocation of an
/// intermediate `Vec`. Matches `ds4_kvstore_sha1_bytes_hex` upstream.
pub fn sha1_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(40);
    for b in digest {
        // {:02x} on a u8 always emits exactly two lowercase hex chars.
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

/// Linear-scan prefix index. Same shape as the C side (a flat list of
/// hex SHA1 strings scanned on lookup); we just back it with
/// `HashSet<String>` for O(1) membership in the common case while
/// keeping the public surface small.
#[derive(Debug, Default)]
pub struct PrefixIndex {
    entries: HashSet<String>,
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self {
            entries: HashSet::new(),
        }
    }

    /// Insert a key. Returns `true` if the key was newly added.
    pub fn insert(&mut self, key: String) -> bool {
        self.entries.insert(key)
    }

    /// Membership test.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains(key)
    }

    /// Number of tracked prefixes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate keys (insertion order is unspecified).
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.entries.iter()
    }

    /// Remove a key. Returns `true` if it was present.
    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key)
    }

    /// Clear all tracked prefixes.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_hex_known_vector() {
        // "abc" -> a9993e364706816aba3e25717850c26c9cd0d89d
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        // "" -> da39a3ee5e6b4b0d3255bfef95601890afd80709
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn sha1_hex_length_is_40() {
        let h = sha1_hex(b"some longer payload that hashes to something");
        assert_eq!(h.len(), 40);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn prefix_index_insert_contains() {
        let mut idx = PrefixIndex::new();
        assert!(!idx.contains("deadbeef"));
        assert!(idx.insert("deadbeef".to_string()));
        assert!(idx.contains("deadbeef"));
        assert!(!idx.insert("deadbeef".to_string()));
        assert_eq!(idx.len(), 1);
        assert!(idx.remove("deadbeef"));
        assert!(!idx.contains("deadbeef"));
    }
}
