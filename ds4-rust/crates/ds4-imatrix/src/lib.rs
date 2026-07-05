// DS4 (DwarfStar) â€” imatrix collection.
//
// First-class engine API for collecting importance-matrix
// statistics. Port of `ds4_engine_collect_imatrix(e, dataset,
// output, ctx, max_prompts, max_tokens)` and the supporting format
// writer (`imatrix_collector_save`).
//
// Output is llama.cpp's legacy `.dat` format â€” see `format.rs` for
// the binary layout.
//
// The collector accepts a caller-supplied logits function so it can be used`r`n// with live engines, recorded logits, or deterministic tests.

pub const CRATE_NAME: &str = "ds4-imatrix";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod collector;
pub mod format;

pub use collector::{CollectorConfig, CollectorError, ImatrixCollector, LogitsFn, TensorSlot};
pub use format::{parse, parse_trailer, to_bytes, ImatrixEntry, ImatrixFormatError};

/// Convenience: collect from `dataset` and write the `.dat` file at
/// `output`. The `logits_fn` closure supplies per-prompt logits.
///
/// The `logits_fn` closure is the integration boundary: callers decide`r`n/// whether logits come from the Rust engine, a fixture, or another backend.
pub fn run_collect(
    config: CollectorConfig,
    dataset: &str,
    output: &str,
    logits_fn: LogitsFn<'_>,
) -> ds4_types::Ds4Result<()> {
    let mut collector = ImatrixCollector::new(config)?;
    collector.collect(dataset, logits_fn)?;
    collector.finalize(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-imatrix");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn round_trip_format_module() {
        let e = vec![ImatrixEntry {
            tensor_name: "blk.0.ffn_gate_exps.weight".to_string(),
            values: vec![0.5; 8],
        }];
        let bytes = to_bytes(&e, 1, "calib");
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, e);
    }
}
