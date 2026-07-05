//
//
// Load-bearing crate: hosts the Ds4Engine / Ds4Session logic,
// the sync -> rewrite_from_common -> RebuildNeeded state machine,
// the three-tier KV cache, the DeepSeek chat template, the sampler,
// the speculative-decoding MTP path, and the GGUF loader.
//
// All shared handle / option / error types live in `ds4-types` so
// this crate can be depended on by `ds4-dist` (and backends) without
// forming a cycle.

pub use ds4_types as types;

pub const CRATE_NAME: &str = "ds4-core";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod chat;
pub mod engine;
pub mod gguf;
pub mod gguf_synth;
pub mod kv;
pub mod mtp;
pub mod sampler;
pub mod session;
pub mod tokenizer;
pub mod tokenizer_data;

pub use chat::{Ds4ChatTemplate, Ds4Role};
pub use engine::{
    ds4_backend_name, ds4_context_memory_estimate, ds4_context_memory_estimate_with_prefill,
    ds4_think_max_min_context, ds4_think_max_prefix, ds4_think_mode_enabled,
    ds4_think_mode_for_context, ds4_think_mode_name, ContextMemory, Ds4Engine,
    DS4_THINK_MAX_MIN_CONTEXT,
};
pub use gguf::{
    GgufDType, GgufFile, GgufMetadata, GgufValueType, KvRaw, QuantizedTensor, TensorDescriptor,
    GGUF_MAGIC, GGUF_VERSION_V3,
};
pub use gguf_synth::write_synthetic_gguf;
pub use kv::{KvCache, DEFAULT_COMPRESSED_DIM, DEFAULT_HEAD_DIM, DEFAULT_INDEXER_DIM};
pub use sampler::{
    argmax, argmax_excluding, log_softmax_inplace, sample_top_p_min_p, top_logprobs,
};
pub use session::Ds4Session;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-core");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn types_re_exports_resolve() {
        let opts = types::Ds4EngineOptions::default();
        assert_eq!(opts.backend, types::Ds4Backend::Cpu);
    }
}
