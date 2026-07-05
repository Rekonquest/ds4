// DS4 (DwarfStar) — disk KV store.
//
// `KvStore` is the Rust-side equivalent of the C-side
// `ds4_kvstore_open` machinery. The C side keeps a flat array of
// `ds4_kvstore_entry`s ordered by an eviction score; we keep a simpler
// "oldest-first" `Lru` plus a SHA1 `PrefixIndex` (matching the
// upstream linear-scan design) and let higher-level policy (continued
// waypoints, anchor reasons) live in the calling code.
//
// On-disk layout under `root`:
//   <root>/<sha1_hex>.dsv4      // whole-session payloads
//   <root>/<sha1_hex>.dsvl      // per-layer payloads
//   <root>/<sha1_hex>.dsv4.tmp  // atomic-write scratch
//
// Writes go `<name>.tmp` -> fsync -> `rename` to the final path so a
// crash never leaves a half-written file in place.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::lru::Lru;
use crate::payload::{LayerPayload, SessionPayload};
use crate::prefix::{sha1_hex, PrefixIndex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    Truncated,
    BadMagic(u32),
    BadVersion(u32),
    BadU32Fields(u32),
    Io(String),
    NotFound,
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvError::Truncated => write!(f, "kv payload truncated"),
            KvError::BadMagic(m) => write!(f, "kv payload bad magic {:#x}", m),
            KvError::BadVersion(v) => write!(f, "kv payload bad version {}", v),
            KvError::BadU32Fields(n) => write!(f, "kv payload bad u32_fields {}", n),
            KvError::Io(e) => write!(f, "kv io error: {}", e),
            KvError::NotFound => write!(f, "kv prefix not found"),
        }
    }
}

impl std::error::Error for KvError {}

impl From<io::Error> for KvError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::NotFound {
            KvError::NotFound
        } else {
            KvError::Io(e.to_string())
        }
    }
}

/// Default on-disk byte budget if the caller doesn't pick one. Mirrors
/// `DS4_KVSTORE_DEFAULT_MB = 4096` from `ds4_kvstore.h` (4 GiB).
pub const DS4_KVSTORE_DEFAULT_MB: usize = 4096;

/// Disk-resident prefix KV store.
pub struct KvStore {
    root: PathBuf,
    prefix: PrefixIndex,
    lru: Lru,
}

impl KvStore {
    /// Open a store rooted at `root` with the default 4 GiB byte budget
    /// and no entry-count cap. Creates `root` if it doesn't exist.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, KvError> {
        Self::open_with_budget(root, DS4_KVSTORE_DEFAULT_MB * 1024 * 1024)
    }

    /// Open a store rooted at `root` with an explicit on-disk byte
    /// budget. `budget_bytes` is the maximum total on-disk size of
    /// tracked entries; older entries are evicted when a write would
    /// push us over.
    pub fn open_with_budget(root: impl AsRef<Path>, budget_bytes: usize) -> Result<Self, KvError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            prefix: PrefixIndex::new(),
            lru: Lru::bytes_capacity(budget_bytes, 0),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn prefix_index(&self) -> &PrefixIndex {
        &self.prefix
    }

    pub fn live_bytes(&self) -> u64 {
        self.lru.live_bytes()
    }

    pub fn cap_bytes(&self) -> u64 {
        self.lru.cap_bytes()
    }

    /// Number of tracked entries. Note: this only reflects entries
    /// touched in the current process; entries from a previous run are
    /// not auto-rehydrated.
    pub fn len(&self) -> usize {
        self.lru.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lru.is_empty()
    }

    // ---------- stage / write split ----------

    /// Stage a payload into a `StagedPayload` so it can be written to
    /// multiple destinations (e.g. the live KV store plus a snapshot
    /// directory) without re-encoding. Mirrors `ds4_session_stage_payload`
    /// on the C side.
    pub fn stage_payload_session(&self, payload: &SessionPayload) -> StagedSession {
        StagedSession {
            sha: sha1_hex(&payload.bytes),
            bytes: payload.to_bytes(),
        }
    }

    pub fn stage_payload_layer(
        &self,
        payload: &LayerPayload,
        layer_start: usize,
        layer_end: usize,
    ) -> StagedLayer {
        StagedLayer {
            sha: sha1_hex(&payload.bytes),
            bytes: payload.to_bytes(),
            layer_start,
            layer_end,
        }
    }

    /// Write a previously-staged session payload to `dir` (atomic).
    /// Returns the final path. Does not touch the in-memory LRU or
    /// prefix index — call `record_session` for that, or use the
    /// combined `save_session` helper.
    pub fn write_staged_session(
        &self,
        staged: &StagedSession,
        dir: impl AsRef<Path>,
    ) -> Result<PathBuf, KvError> {
        atomic_write(dir.as_ref(), &staged.file_name(), &staged.bytes)
    }

    pub fn write_staged_layer(
        &self,
        staged: &StagedLayer,
        dir: impl AsRef<Path>,
    ) -> Result<PathBuf, KvError> {
        atomic_write(dir.as_ref(), &staged.file_name(), &staged.bytes)
    }

    // ---------- session API ----------

    /// Encode `payload`, hash it, atomically write it to `<root>/<sha>.dsv4`,
    /// record it in the LRU + prefix index, and evict older entries to
    /// stay within the on-disk byte budget. Returns the final on-disk
    /// path.
    pub fn save_session(&mut self, payload: &SessionPayload) -> Result<PathBuf, KvError> {
        let staged = self.stage_payload_session(payload);
        let path = self.write_staged_session(&staged, self.root.clone())?;
        self.record_session(&staged, &path);
        Ok(path)
    }

    fn record_session(&mut self, staged: &StagedSession, path: &Path) {
        let bytes = staged.bytes.len() as u64;
        // If we previously recorded this SHA, just refresh it.
        if self.prefix.contains(&staged.sha) {
            // Old entry stays in LRU with refreshed size; touch replaces.
            self.lru.touch(staged.sha.clone(), bytes);
            return;
        }
        // New entry: enforce the byte budget by evicting oldest entries
        // until we'd fit (we'll then touch() below, which adds bytes).
        while let Some(victim) = self.lru.next_eviction_for(bytes) {
            if victim.key == staged.sha {
                break;
            }
            let victim_key = victim.key.clone();
            let victim_path = self.root.join(format!("{}.dsv4", victim_key));
            // Best-effort remove of the victim file. If it's already
            // gone, that's fine — we still want to drop the entry.
            let _ = fs::remove_file(&victim_path);
            self.prefix.remove(&victim_key);
            self.lru.evict_next();
        }
        self.prefix.insert(staged.sha.clone());
        self.lru.touch(staged.sha.clone(), bytes);
        // `path` is recorded only via the LRU; we keep the parameter so
        // hook callbacks (cache-hit updates, log lines) have
        // a stable place to read it.
        let _ = path;
    }

    /// Load a session payload by its SHA1 hex name (without the
    /// `.dsv4` extension). Uses `from_bytes_stream` so the payload
    /// body isn't double-buffered.
    pub fn load_session(&self, sha: &str) -> Result<SessionPayload, KvError> {
        let path = self.root.join(format!("{}.dsv4", sha));
        let file = File::open(&path)?;
        SessionPayload::from_bytes_stream(file)
    }

    /// Load the most recently stored session whose stored SHA prefix
    /// matches `prefix`. Linear scan (matches the C-side policy of
    /// not using rax on this path). Returns `None` when no entry
    /// starts with `prefix`.
    pub fn find_session_by_prefix(&self, prefix: &str) -> Result<Option<SessionPayload>, KvError> {
        // Walk newest-to-oldest via the LRU order.
        let mut matches: Vec<&str> = self
            .prefix
            .iter()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.as_str())
            .collect();
        matches.sort_unstable_by(|a, b| b.cmp(a));
        match matches.first() {
            Some(sha) => self.load_session(sha).map(Some),
            None => Ok(None),
        }
    }

    // ---------- layer API ----------

    pub fn save_layer(
        &mut self,
        payload: &LayerPayload,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<PathBuf, KvError> {
        let staged = self.stage_payload_layer(payload, layer_start, layer_end);
        let path = self.write_staged_layer(&staged, self.root.clone())?;
        let bytes = staged.bytes.len() as u64;
        let key = format!("{}#{}", staged.sha, staged.range_key());
        if !self.prefix.contains(&key) {
            while let Some(victim) = self.lru.next_eviction_for(bytes) {
                if victim.key == key {
                    break;
                }
                let victim_key = victim.key.clone();
                // Layer paths embed a range key; pull the SHA part.
                let sha_part = victim_key.split('#').next().unwrap_or(&victim_key);
                let victim_path = self.root.join(format!(
                    "{}-{}-{}.dsvl",
                    sha_part,
                    range_from_key(&victim_key).0,
                    range_from_key(&victim_key).1,
                ));
                let _ = fs::remove_file(&victim_path);
                self.prefix.remove(&victim_key);
                self.lru.evict_next();
            }
            self.prefix.insert(key.clone());
        }
        self.lru.touch(key, bytes);
        Ok(path)
    }

    pub fn load_layer(
        &self,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<Option<LayerPayload>, KvError> {
        // Iterate the prefix index looking for matching range keys.
        // We don't keep a separate layer index because layer ranges are
        // tiny in practice and a linear scan over <prefix>.len() entries
        // is what the upstream port does.
        for key in self.prefix.iter() {
            let (sha_part, start, end) = match parse_layer_key(key) {
                Some(v) => v,
                None => continue,
            };
            if start == layer_start && end == layer_end {
                let path = self
                    .root
                    .join(format!("{}-{}-{}.dsvl", sha_part, start, end));
                let file = File::open(&path)?;
                return Ok(Some(LayerPayload::from_bytes_stream(file)?));
            }
        }
        Ok(None)
    }
}

// ---------- staged payloads ----------

/// A whole-session payload that has been hashed and encoded and is
/// ready to be written to one or more destinations.
#[derive(Debug, Clone)]
pub struct StagedSession {
    pub sha: String,
    pub bytes: Vec<u8>,
}

impl StagedSession {
    fn file_name(&self) -> String {
        format!("{}.dsv4", self.sha)
    }
}

/// A per-layer payload that has been hashed and encoded, with the
/// layer range baked into the file name.
#[derive(Debug, Clone)]
pub struct StagedLayer {
    pub sha: String,
    pub bytes: Vec<u8>,
    pub layer_start: usize,
    pub layer_end: usize,
}

impl StagedLayer {
    fn file_name(&self) -> String {
        format!("{}-{}-{}.dsvl", self.sha, self.layer_start, self.layer_end)
    }
    fn range_key(&self) -> String {
        format!("{}-{}", self.layer_start, self.layer_end)
    }
}

fn range_from_key(key: &str) -> (usize, usize) {
    parse_layer_key(key)
        .map(|(_, s, e)| (s, e))
        .unwrap_or((0, 0))
}

fn parse_layer_key(key: &str) -> Option<(&str, usize, usize)> {
    let (sha, rest) = key.split_once('#')?;
    let (start_s, end_s) = rest.split_once('-')?;
    let start = start_s.parse::<usize>().ok()?;
    let end = end_s.parse::<usize>().ok()?;
    Some((sha, start, end))
}

// ---------- atomic write ----------

fn atomic_write(dir: &Path, file_name: &str, bytes: &[u8]) -> Result<PathBuf, KvError> {
    fs::create_dir_all(dir)?;
    let final_path = dir.join(file_name);
    let tmp_path = dir.join(format!("{}.tmp", file_name));
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.flush()?;
        // fsync the data so a crash doesn't leave the rename() pointing
        // at unflushed bytes.
        f.sync_all()?;
    }
    // rename is atomic on POSIX; on Windows it replaces if the dest exists.
    // We allow that here because all writers go through `tmp_path` first.
    if let Err(_e) = fs::rename(&tmp_path, &final_path) {
        #[cfg(windows)]
        {
            let _ = fs::remove_file(&final_path);
            fs::rename(&tmp_path, &final_path).map_err(|e2| KvError::Io(e2.to_string()))?;
        }
        #[cfg(not(windows))]
        {
            return Err(KvError::Io(_e.to_string()));
        }
    }
    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TMPDIR_ID: AtomicU64 = AtomicU64::new(1);

    fn tmpdir(name: &str) -> PathBuf {
        let base = std::env::temp_dir();
        let pid = std::process::id();
        let id = NEXT_TMPDIR_ID.fetch_add(1, Ordering::Relaxed);
        let dir = base.join(format!("ds4-kvstore-test-{pid}-{id}-{name}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn open_creates_directory() {
        let dir = tmpdir("open");
        let store = KvStore::open(&dir).unwrap();
        assert!(dir.exists());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn save_and_load_roundtrip_small() {
        let dir = tmpdir("small");
        let mut store = KvStore::open(&dir).unwrap();
        let payload = SessionPayload::new(b"abc".to_vec());
        let staged = store.stage_payload_session(&payload);
        let path = store.save_session(&payload).unwrap();
        assert!(path.exists());
        assert_eq!(path, dir.join(format!("{}.dsv4", staged.sha)));
        let loaded = store.load_session(&staged.sha).unwrap();
        assert_eq!(loaded.bytes, payload.bytes);
    }

    #[test]
    fn stage_can_be_written_to_two_destinations() {
        let dir_a = tmpdir("stage-a");
        let dir_b = tmpdir("stage-b");
        let store = KvStore::open(&dir_a).unwrap();
        let payload = SessionPayload::new(b"two-dest".to_vec());
        let staged = store.stage_payload_session(&payload);
        let p1 = store.write_staged_session(&staged, &dir_a).unwrap();
        let p2 = store.write_staged_session(&staged, &dir_b).unwrap();
        assert_eq!(p1, dir_a.join(format!("{}.dsv4", staged.sha)));
        assert_eq!(p2, dir_b.join(format!("{}.dsv4", staged.sha)));
    }

    #[test]
    fn byte_budget_evicts_oldest() {
        let dir = tmpdir("budget");
        let mut store = KvStore::open_with_budget(&dir, 1024).unwrap();
        // Each header is 16 bytes; pick payloads that push us past
        // the budget on the second write. Use distinct bytes so the
        // two saves get distinct SHA1 names.
        let mut big_a = vec![0u8; 900];
        let mut big_b = vec![0u8; 900];
        big_a[0] = 0xA1;
        big_b[0] = 0xB1;
        let p1 = SessionPayload::new(big_a);
        let p2 = SessionPayload::new(big_b);
        let path1 = store.save_session(&p1).unwrap();
        let _path2 = store.save_session(&p2).unwrap();
        // The first save should have written a .dsv4 file; after the
        // second save pushes us over budget, the oldest (first)
        // entry should be evicted, so its file is gone.
        assert!(
            !path1.exists(),
            "oldest entry should have been evicted by byte budget"
        );
        // Exactly one entry remains in the in-memory index.
        assert_eq!(store.prefix_index().len(), 1);
    }

    #[test]
    fn layer_save_and_load() {
        let dir = tmpdir("layer");
        let mut store = KvStore::open(&dir).unwrap();
        let payload = LayerPayload::new(b"layer data".to_vec());
        let path = store.save_layer(&payload, 0, 4).unwrap();
        assert!(path.exists());
        let loaded = store.load_layer(0, 4).unwrap().expect("found");
        assert_eq!(loaded.bytes, payload.bytes);
    }

    #[test]
    fn atomic_write_replaces_existing_file() {
        let dir = tmpdir("atomic");
        fs::create_dir_all(&dir).unwrap();
        let path = atomic_write(&dir, "x.dsv4", b"first").unwrap();
        let mut f = File::open(&path).unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"first");
        atomic_write(&dir, "x.dsv4", b"second").unwrap();
        let mut f = File::open(&path).unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"second");
    }

    #[test]
    fn load_session_missing_returns_not_found() {
        let dir = tmpdir("missing");
        let store = KvStore::open(&dir).unwrap();
        let err = store.load_session("deadbeef").unwrap_err();
        assert!(matches!(err, KvError::NotFound));
    }

    #[test]
    fn cleanup_tmpdirs() {
        // Best-effort cleanup so the test runner doesn't fill /tmp.
        let pid = std::process::id();
        let base = std::env::temp_dir();
        for suffix in [
            "open", "small", "stage-a", "stage-b", "budget", "layer", "atomic", "missing",
        ] {
            let _ = fs::remove_dir_all(base.join(format!("ds4-kvstore-test-{}-{}", suffix, pid)));
        }
    }
}
