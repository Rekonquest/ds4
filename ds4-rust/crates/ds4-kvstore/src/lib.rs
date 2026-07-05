// DS4 (DwarfStar) — disk KV cache.
//
// Faithful port of upstream ds4_kvstore. The on-disk payload formats
// (DSV4 / DSVL) are preserved byte-for-byte so Rust-written and
// C-written payloads interoperate.

pub const CRATE_NAME: &str = "ds4-kvstore";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// On-disk payload constants. MUST match upstream ds4.h exactly.
pub const DS4_SESSION_PAYLOAD_MAGIC: u32 = payload::DS4_SESSION_PAYLOAD_MAGIC;
pub const DS4_SESSION_PAYLOAD_VERSION: u32 = payload::DS4_SESSION_PAYLOAD_VERSION;
pub const DS4_SESSION_PAYLOAD_U32_FIELDS: u32 = payload::DS4_SESSION_PAYLOAD_U32_FIELDS;
pub const DS4_SESSION_LAYER_PAYLOAD_MAGIC: u32 = payload::DS4_SESSION_LAYER_PAYLOAD_MAGIC;
pub const DS4_SESSION_LAYER_PAYLOAD_VERSION: u32 = payload::DS4_SESSION_LAYER_PAYLOAD_VERSION;
pub const DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS: u32 = payload::DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS;

pub mod lru;
pub mod payload;
pub mod prefix;
pub mod store;

pub use payload::{LayerPayload, SessionPayload};
pub use prefix::{sha1_hex, PrefixIndex};
pub use store::{KvError, KvStore, StagedLayer, StagedSession};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn payload_magic_values_are_locked() {
        assert_eq!(DS4_SESSION_PAYLOAD_MAGIC, 0x3456_5344);
        assert_eq!(DS4_SESSION_LAYER_PAYLOAD_MAGIC, 0x4c56_5344);
        assert_eq!(DS4_SESSION_PAYLOAD_VERSION, 2);
        assert_eq!(DS4_SESSION_LAYER_PAYLOAD_VERSION, 1);
        assert_eq!(DS4_SESSION_PAYLOAD_U32_FIELDS, 13);
        assert_eq!(DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS, 14);
    }

    #[test]
    fn session_payload_roundtrip() {
        let p = SessionPayload::new(b"hello world".to_vec());
        let bytes = p.to_bytes();
        let q = SessionPayload::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(p.bytes, q.bytes);
        assert_eq!(p.magic, q.magic);
        assert_eq!(p.version, q.version);
    }

    #[test]
    fn layer_payload_rejects_bad_magic() {
        let mut bytes = LayerPayload::new(vec![0; 16]).to_bytes();
        bytes[0] = 0xff; // clobber magic
        let err = LayerPayload::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, KvError::BadMagic(_)));
    }

    #[test]
    fn lru_evicts_in_order() {
        let mut lru = lru::Lru::new(3);
        lru.touch("a".into(), 1);
        lru.touch("b".into(), 1);
        lru.touch("c".into(), 1);
        lru.touch("a".into(), 1); // refresh
        assert_eq!(lru.evict_next().unwrap().key, "b");
        assert_eq!(lru.evict_next().unwrap().key, "c");
        assert_eq!(lru.evict_next().unwrap().key, "a");
        assert_eq!(lru.evict_next(), None);
    }

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-kvstore");
        assert!(!VERSION.is_empty());
    }

    /// Build a 1 MiB random byte payload, save it to a tempdir, read
    /// it back via the streaming reader, and assert byte-equal. This
    /// is the real-world round-trip: large payload, on disk, no
    /// double-buffering through `Vec<u8>` before decoding.
    #[test]
    fn roundtrip_1mib_random_payload() {
        // Simple deterministic pseudo-random so the test is hermetic.
        let mut state: u64 = 0x1234_5678_dead_beef;
        let mut next = move || -> u8 {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u8
        };
        let mut payload_bytes = vec![0u8; 1024 * 1024];
        for b in payload_bytes.iter_mut() {
            *b = next();
        }

        let dir = unique_tmpdir("roundtrip-1mib");
        let mut store = KvStore::open(&dir).expect("open store");
        let payload = SessionPayload::new(payload_bytes.clone());
        let staged = store.stage_payload_session(&payload);
        let path = store.save_session(&payload).expect("save_session");
        assert!(path.exists());

        // Header bytes are 16; the on-disk file is header + payload.
        let file_size = fs::metadata(&path).expect("metadata").len();
        assert_eq!(
            file_size as usize,
            SessionPayload::header_bytes() + payload_bytes.len()
        );

        // Read back via the streaming decoder.
        let loaded = store.load_session(&staged.sha).expect("load_session");
        assert_eq!(loaded.bytes.len(), payload_bytes.len());
        assert_eq!(loaded.bytes, payload_bytes, "payload bytes must match");

        // The file should also be decodable byte-for-byte from the raw
        // disk bytes (matches `from_bytes` slice path used elsewhere).
        let raw = fs::read(&path).expect("read");
        let slice_decoded = SessionPayload::from_bytes(&raw).expect("from_bytes");
        assert_eq!(slice_decoded.bytes, payload_bytes);

        // Cleanup best-effort.
        let _ = fs::remove_dir_all(&dir);
    }

    fn unique_tmpdir(label: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("ds4-kvstore-{}-{}-{}", label, pid, nanos));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create tmpdir");
        dir
    }
}
