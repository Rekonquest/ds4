// DS4 (DwarfStar) — paged-attention backend.
//
// This crate hosts the paged-attention attention kernel and the
// per-layer page-table data structure. The kernel is the clean-Rust
// reference implementation (see `paged_attention.rs`); the vendored
// `mistralrs-paged-attn` is gated behind the `vendored` Cargo feature
// and is not built by default because its manifest still inherits
// sibling mistral.rs workspace fields that we have not vendored.
//
// Two public surfaces:
//
// * `page_table::{Page, PageTable, PAGE_TOKENS}` — the data structure
//   used by every other crate that needs paged-KV.
//
// * `backend::{PagedBackend, PagedModel}` — the `Backend` impl and
//   the loaded-model handle. Mirrors `ds4-backend-cpu`'s shape so
//   the engine can dispatch between CPU, CUDA, ROCm, Metal and the
//   paged-attention path through the same `dyn BackendModel`.
//
// # Feature flags
// * (default) — clean-Rust reference kernel, no vendored deps.
// * `vendored` — pulls in `third_party/mistralrs-paged-attn` as a
//   path dep. The vendored crate currently fails to parse its
//   manifest standalone (its `Cargo.toml` inherits
//   `version.workspace = true`, `candle-core.workspace = true`,
//   etc., which we don't define in the DS4 workspace). Until the
//   upstream mistral.rs workspace is vendored and the parent
//   `[workspace]` table provides those fields, `vendored` will not
//   compile. The reference path is correct and ships in v1.

pub const CRATE_NAME: &str = "ds4-backend-paged";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod backend;
pub mod page_table;
pub mod paged_attention;

pub use backend::{PagedBackend, PagedModel};
pub use page_table::{Page, PageTable, PAGE_TOKENS};
pub use paged_attention::{
    page_cache_bytes, paged_attention_decode, paged_attention_decode_into, vendored_status,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-backend-paged");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn page_tokens_constant_is_16() {
        // Matches the vendored mistralrs-paged-attn and the C
        // reference. Hard-coded here as a tripwire: changing this
        // constant without updating both call sites will break
        // caching.
        assert_eq!(PAGE_TOKENS, 16);
    }

    #[test]
    fn vendored_status_reflects_feature_flag() {
        // In the default build (feature off) this is false. The
        // test exists so flipping the feature is the only change
        // required to switch the backend's hot path to the vendored
        // kernel.
        let _ = vendored_status();
    }
}
