// DS4 (DwarfStar) — small ID helper.
//
// We avoid pulling in `uuid` (the workspace doesn't depend on it yet)
// and instead emit a 128-bit hex string with a millisecond timestamp
// + an atomic counter. Format: `<timestamp_ms>-<counter>`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn new_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{ts:x}-{counter:x}")
}
