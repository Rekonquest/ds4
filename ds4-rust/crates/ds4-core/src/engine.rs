//
//
// Mirrors the public surface of `ds4_engine_open` /
// `ds4_engine_close` / `ds4_engine_summary` /
// `ds4_engine_layer_compress_ratio` / etc. in `ds4.c:23120..`.
use std::path::Path;
use std::sync::Arc;

use ds4_types::{
    Backend, BackendModel, Ds4Backend, Ds4EngineOptions, Ds4Error, Ds4ErrorKind, Ds4Result,
    Ds4ThinkMode,
};

use crate::chat::{ChatSentinels, Ds4ChatTemplate};
use crate::gguf::GgufFile;
use crate::gguf_synth;
use crate::mtp::Ds4Mtp;
use crate::tokenizer::Ds4Tokenizer;

/// Context memory estimate (mirrors `ds4_context_memory`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextMemory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

/// Engine-wide state Ã¢â‚¬â€ file path, GGUF handle, backend, sentinels, and
/// shared tokenizers + MTP module.
pub struct Ds4Engine {
    opts: Ds4EngineOptions,
    gguf: Option<GgufFile>,
    backend: Box<dyn Backend>,
    /// Loaded model from the backend's `load_model` call. None if the
    /// model path was missing or invalid.
    model: Option<Box<dyn BackendModel>>,
    tokenizer: Ds4Tokenizer,
    chat: Ds4ChatTemplate,
    mtp: Ds4Mtp,
    power_percent: u8,
    model_name: String,
    model_id: String,
    n_layers: usize,
    n_vocab: usize,
    compress_ratio: u32,
    layer_compress_ratios: Vec<u32>,
    hidden_f32_values: usize,
    routed_quant_bits: u8,
    has_output_head: bool,
    has_mtp: bool,
    mtp_draft_tokens: usize,
    sentinels: ChatSentinels,
}

impl Ds4Engine {
    /// Open the model: validate options, load the GGUF, choose backend,
    /// build the tokenizer + chat template, and ask the backend to
    /// `load_model` the file. The returned `Ds4Engine` is ready to
    /// drive inference once a session is opened.
    pub fn open(opts: Ds4EngineOptions) -> Ds4Result<Self> {
        if opts.model_path.as_os_str().is_empty() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "model_path is empty",
            ));
        }

        // Pick the requested backend. Device-specific crates expose
        // host-independent model loading, while kernel compilation stays explicit.
        let mut backend: Box<dyn Backend> = match opts.backend {
            Ds4Backend::Cpu => Box::new(ds4_backend_cpu::CpuBackend::new()),
            Ds4Backend::Cuda => Box::new(ds4_backend_cuda::CudaBackend::new()),
            Ds4Backend::Arc => Box::new(ds4_backend_arc::ArcBackend::new()),
            Ds4Backend::Vulkan => Box::new(ds4_backend_vulkan::VulkanBackend::new()),
            Ds4Backend::Rocm => Box::new(ds4_backend_rocm::RocmBackend::new()),
            Ds4Backend::Metal => Box::new(ds4_backend_metal::MetalBackend::new()),
        };

        // Best effort GGUF open; if the file is missing or invalid we
        // still synthesize an engine so callers can do
        // `engine.tokenize(...)` on the empty vocab. The C engine has
        // the same shape for invalid model paths: it still produces an
        // opaque handle that lights up when the file becomes valid.
        let gguf = GgufFile::open(&opts.model_path).ok();

        // Try to actually load the model. Backend crates own their selected
        // device path and may use a host-backed loader for GGUF parsing.
        let model = if gguf.is_some() {
            match backend.load_model(&opts.model_path) {
                Ok(m) => Some(m),
                Err(e)
                    if e.kind == Ds4ErrorKind::NotImplemented
                        && opts.backend != Ds4Backend::Cpu =>
                {
                    let cpu: Box<dyn Backend> = Box::new(ds4_backend_cpu::CpuBackend::new());
                    match cpu.load_model(&opts.model_path) {
                        Ok(m) => {
                            backend = cpu;
                            Some(m)
                        }
                        Err(cpu_e) if cpu_e.kind == Ds4ErrorKind::NotImplemented => None,
                        Err(cpu_e) => return Err(cpu_e),
                    }
                }
                Err(e) if e.kind == Ds4ErrorKind::NotImplemented => None,
                Err(e) => return Err(e),
            }
        } else {
            None
        };

        let (n_layers, n_vocab, hidden_f32_values, has_mtp) = if let Some(g) = gguf.as_ref() {
            if let Ok(spec) = ds4_gguf::ModelSpec::from_gguf(g) {
                (
                    spec.dims.layers,
                    spec.dims.vocab,
                    spec.dims.hidden,
                    opts.mtp_path.is_some() || opts.mtp_draft_tokens > 0,
                )
            } else {
                let nl = g
                    .metadata
                    .layer_count
                    .unwrap_or_else(|| g.n_tensors().max(1) as u32)
                    as usize;
                let nv = g.metadata.vocab_size.unwrap_or(129_280) as usize;
                (
                    nl,
                    nv,
                    1,
                    opts.mtp_path.is_some() || opts.mtp_draft_tokens > 0,
                )
            }
        } else {
            (0, 0, 0, opts.mtp_draft_tokens > 0)
        };
        let model_name = gguf
            .as_ref()
            .map(|g| g.path().to_string_lossy().to_string())
            .unwrap_or_default();
        let model_id = match gguf.as_ref() {
            Some(_) if n_layers >= 61 => "ds4-pro".to_string(),
            Some(_) => "ds4-flash".to_string(),
            None => String::new(),
        };
        let layer_compress_ratios: Vec<u32> = (0..n_layers)
            .map(|i| if i % 4 == 0 && i > 0 { 4 } else { 0 })
            .collect();
        let compress_ratio = layer_compress_ratios.iter().copied().max().unwrap_or(0);
        let sentinels = gguf
            .as_ref()
            .map(|g| ChatSentinels {
                bos: g.metadata.bos_token_id.unwrap_or(1),
                user: g.metadata.user_token_id.unwrap_or(3),
                assistant: g.metadata.assistant_token_id.unwrap_or(4),
                think_start: g.metadata.think_start_token_id.unwrap_or(5),
                think_end: g.metadata.think_end_token_id.unwrap_or(6),
                dsml: g.metadata.dsml_token_id.unwrap_or(7),
            })
            .unwrap_or(ChatSentinels {
                bos: 1,
                user: 3,
                assistant: 4,
                think_start: 5,
                think_end: 6,
                dsml: 7,
            });
        let tokenizer = if let Some(g) = gguf.as_ref() {
            if let Some(tokenizer) = Ds4Tokenizer::from_gguf(g)? {
                tokenizer
            } else {
                byte_fallback_tokenizer(sentinels)?
            }
        } else {
            byte_fallback_tokenizer(sentinels)?
        };
        let chat = Ds4ChatTemplate::new(tokenizer.clone()).with_sentinels(sentinels);
        let mtp = Ds4Mtp::with_config(crate::mtp::Ds4MtpConfig {
            draft_tokens: opts.mtp_draft_tokens,
            margin: opts.mtp_margin,
        });
        let power_percent = opts.power_percent.min(100);
        let mtp_draft_tokens = opts.mtp_draft_tokens;
        Ok(Ds4Engine {
            opts,
            gguf,
            backend,
            model,
            tokenizer,
            chat,
            mtp,
            power_percent,
            model_name,
            model_id,
            n_layers,
            n_vocab,
            compress_ratio,
            layer_compress_ratios,
            hidden_f32_values,
            routed_quant_bits: 0,
            has_output_head: true,
            has_mtp,
            mtp_draft_tokens,
            sentinels,
        })
    }

    /// Borrow the loaded backend model, if any. Returns `None` if
    /// `open()` succeeded but no backend model was loaded or
    /// the model file was missing/invalid.
    pub fn model(&self) -> Option<&dyn BackendModel> {
        self.model.as_deref()
    }

    pub fn eval_token_logits(&self, token: u32, out: &mut [f32]) -> Ds4Result<()> {
        match self.model.as_deref() {
            Some(model) => model.eval_token_logits(token, out),
            None => Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "engine has no loaded model for token logits",
            )),
        }
    }

    pub fn eval_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        match self.model.as_deref() {
            Some(model) => model.eval_sequence_logits(tokens, out),
            None => Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "engine has no loaded model for sequence logits",
            )),
        }
    }

    /// Borrow the engine options (read-only). Session code uses this
    /// to access prefill_chunk, distributed config, etc.
    pub fn options(&self) -> &Ds4EngineOptions {
        &self.opts
    }

    /// Convenience for tests: write a synthetic GGUF to `path` and
    /// return Ok(()) on success. Re-export of the underlying
    /// `gguf_synth::write_synthetic_gguf` so tests in `ds4-core/tests/`
    /// don't need a separate dependency on the `gguf_synth` module.
    pub fn write_synthetic_gguf(path: &Path) -> Ds4Result<()> {
        gguf_synth::write_synthetic_gguf(path)
    }

    pub fn close(self) {
        // Components (Mmap, etc.) drop on their own; nothing else is
        // required to release.
        let _ = Arc::strong_count(&Arc::new(()));
    }

    pub fn summary(&self) -> String {
        format!(
            "DS4 engine: backend={} layers={} vocab={} mtp={} model_loaded={}",
            self.backend.name(),
            self.n_layers,
            self.n_vocab,
            if self.has_mtp { "on" } else { "off" },
            if self.model.is_some() { "yes" } else { "no" },
        )
    }

    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }
    pub fn power(&self) -> u8 {
        self.power_percent
    }
    pub fn set_power(&mut self, p: u8) {
        self.power_percent = p.min(100);
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    pub fn layer_count(&self) -> usize {
        self.n_layers
    }
    pub fn hidden_f32_values(&self) -> usize {
        self.hidden_f32_values
    }
    pub fn routed_quant_bits(&self) -> u8 {
        self.routed_quant_bits
    }
    pub fn compress_ratio(&self) -> u32 {
        self.compress_ratio
    }
    pub fn has_output_head(&self) -> bool {
        self.has_output_head
    }
    pub fn has_mtp(&self) -> bool {
        self.has_mtp
    }
    pub fn mtp_draft_tokens(&self) -> usize {
        self.mtp_draft_tokens
    }

    /// Start a chat session: returns the BOS-prefixed empty token
    /// stream that callers prepend to the first user turn. Delegates
    /// to the chat template.
    pub fn chat_begin(&self) -> Vec<u32> {
        self.chat.begin()
    }

    pub fn layer_compress_ratio(&self, layer: usize) -> u32 {
        self.layer_compress_ratios.get(layer).copied().unwrap_or(0)
    }

    pub fn gguf(&self) -> Option<&GgufFile> {
        self.gguf.as_ref()
    }
    pub fn tokenizer(&self) -> &Ds4Tokenizer {
        &self.tokenizer
    }
    pub fn chat(&self) -> &Ds4ChatTemplate {
        &self.chat
    }
    pub fn sentinels(&self) -> ChatSentinels {
        self.sentinels
    }
    pub fn mtp(&self) -> &Ds4Mtp {
        &self.mtp
    }
}

fn byte_fallback_tokenizer(sentinels: ChatSentinels) -> Ds4Result<Ds4Tokenizer> {
    let mut bytes_map = [0u32; 256];
    for b in 0u32..256 {
        bytes_map[b as usize] = b + 1;
    }
    Ds4Tokenizer::from_byte_mapping(
        bytes_map,
        0,
        sentinels.bos,
        2,
        sentinels.user,
        sentinels.assistant,
        sentinels.think_start,
        sentinels.think_end,
        sentinels.dsml,
    )
}
// ---------------------------------------------------------------------------
// Backend selector.
// ---------------------------------------------------------------------------

pub fn ds4_backend_name(b: Ds4Backend) -> &'static str {
    match b {
        Ds4Backend::Metal => "metal",
        Ds4Backend::Cuda => "cuda",
        Ds4Backend::Arc => "arc",
        Ds4Backend::Vulkan => "vulkan",
        Ds4Backend::Rocm => "rocm",
        Ds4Backend::Cpu => "cpu",
    }
}

pub fn ds4_think_mode_enabled(mode: Ds4ThinkMode) -> bool {
    matches!(mode, Ds4ThinkMode::High | Ds4ThinkMode::Max)
}

pub fn ds4_think_mode_name(mode: Ds4ThinkMode) -> &'static str {
    match mode {
        Ds4ThinkMode::None => "none",
        Ds4ThinkMode::High => "high",
        Ds4ThinkMode::Max => "max",
    }
}

pub fn ds4_think_max_prefix() -> &'static str {
    "<Ã¯Â½Å“DSMLÃ¯Â½Å“think_max_modeÃ¯Â½Å“enabledÃ¯Â½Å“>\n"
}

pub const DS4_THINK_MAX_MIN_CONTEXT: u32 = 393_216;

pub fn ds4_think_max_min_context() -> u32 {
    DS4_THINK_MAX_MIN_CONTEXT
}

pub fn ds4_think_mode_for_context(mode: Ds4ThinkMode, ctx_size: i32) -> Ds4ThinkMode {
    if mode == Ds4ThinkMode::Max && (ctx_size as u32) < DS4_THINK_MAX_MIN_CONTEXT {
        Ds4ThinkMode::High
    } else {
        mode
    }
}

pub fn ds4_context_memory_estimate(backend: Ds4Backend, ctx_size: usize) -> ContextMemory {
    ds4_context_memory_estimate_with_prefill(backend, ctx_size, 0)
}

pub fn ds4_context_memory_estimate_with_prefill(
    backend: Ds4Backend,
    ctx_size: usize,
    prefill_chunk: u32,
) -> ContextMemory {
    let raw_factor: u64 = match backend {
        Ds4Backend::Cpu => 256,
        _ => 512,
    };
    let scratch_factor: u64 = match backend {
        Ds4Backend::Cpu => 128,
        _ => 256,
    };
    let ctx = ctx_size.max(1) as u64;
    let raw_bytes = ctx * raw_factor;
    let scratch_bytes = (prefill_chunk.max(1) as u64) * scratch_factor;
    let total = raw_bytes + scratch_bytes + 4096;
    ContextMemory {
        total_bytes: total,
        raw_bytes,
        compressed_bytes: 0,
        scratch_bytes,
        prefill_cap: prefill_chunk,
        raw_cap: ctx as u32,
        comp_cap: ctx as u32,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn engine_open_with_empty_path_errors() {
        let result = Ds4Engine::open(Ds4EngineOptions::default());
        let err = result.err().expect("expected error for empty path");
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn engine_open_synthesizes_handle_for_missing_path() {
        let opts = Ds4EngineOptions {
            model_path: std::path::PathBuf::from("nope/does-not-exist.gguf"),
            ..Ds4EngineOptions::default()
        };
        let e = Ds4Engine::open(opts).expect("open");
        assert_eq!(e.power(), 100);
        assert!(e.has_output_head());
        assert!(e.model().is_none(), "no model when path is invalid");
    }

    #[test]
    fn options_accessor_returns_engine_options() {
        let opts = Ds4EngineOptions {
            prefill_chunk: 1024,
            model_path: std::path::PathBuf::from("nope.gguf"),
            ..Ds4EngineOptions::default()
        };
        let e = Ds4Engine::open(opts).expect("open");
        assert_eq!(e.options().prefill_chunk, 1024);
    }

    #[test]
    fn chat_begin_delegates_to_chat_template() {
        let opts = Ds4EngineOptions {
            model_path: PathBuf::from("missing.gguf"),
            ..Ds4EngineOptions::default()
        };
        let e = Ds4Engine::open(opts).expect("open");
        // Chat begin returns BOS + Assistant sentinel. Sentinels are
        // 1 (bos) and 4 (assistant) per ChatSentinels::default.
        assert_eq!(e.chat_begin(), vec![1, 4]);
    }

    #[test]
    fn backend_name_match_helper() {
        assert_eq!(ds4_backend_name(Ds4Backend::Cpu), "cpu");
        assert_eq!(ds4_backend_name(Ds4Backend::Metal), "metal");
    }

    #[test]
    fn think_mode_helpers() {
        assert!(ds4_think_mode_enabled(Ds4ThinkMode::High));
        assert!(!ds4_think_mode_enabled(Ds4ThinkMode::None));
        assert_eq!(ds4_think_max_min_context(), 393_216);
        let m = ds4_think_mode_for_context(Ds4ThinkMode::Max, 1024);
        assert_eq!(m, Ds4ThinkMode::High);
    }

    #[test]
    fn memory_estimate_grows_with_context() {
        let small = ds4_context_memory_estimate(Ds4Backend::Cpu, 1024);
        let big = ds4_context_memory_estimate(Ds4Backend::Cpu, 8192);
        assert!(big.total_bytes > small.total_bytes);
    }

    #[test]
    fn synthetic_gguf_round_trip_loads_and_produces_model() {
        // Build a synthetic GGUF in a tempdir.
        let dir = std::env::temp_dir().join("ds4-synth-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("synth.gguf");
        Ds4Engine::write_synthetic_gguf(&path).expect("write synth gguf");
        // Sanity: file is a real GGUF.
        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(&bytes[0..4], &crate::gguf::GGUF_MAGIC.to_le_bytes());
        // Open via the engine; expect GGUF metadata to be parsed.
        let opts = Ds4EngineOptions {
            model_path: path.clone(),
            ..Ds4EngineOptions::default()
        };
        let e = Ds4Engine::open(opts).expect("open");
        assert_eq!(e.n_vocab, 16);
        assert_eq!(e.n_layers, 1);
        assert_eq!(e.tokenizer().vocab_size(), 16);
        assert_eq!(e.tokenizer().tokenize("hi").expect("tokenize"), vec![10]);
        assert!(e.summary().contains("backend=cpu"));
        assert!(
            e.model().is_some(),
            "CPU backend should load a valid GGUF model handle"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_model_rejects_token_outside_vocab() {
        let dir = std::env::temp_dir().join("ds4-synth-token-bounds-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("synth.gguf");
        Ds4Engine::write_synthetic_gguf(&path).expect("write synth gguf");
        let e = Ds4Engine::open(Ds4EngineOptions {
            model_path: path.clone(),
            ..Ds4EngineOptions::default()
        })
        .expect("open");
        let mut logits = vec![0.0f32; e.vocab_size()];
        let err = e
            .eval_token_logits(e.vocab_size() as u32, &mut logits)
            .err()
            .unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cuda_backend_request_falls_back_to_cpu_name_when_device_unavailable() {
        let dir = std::env::temp_dir().join("ds4-cuda-load-model-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("synth.gguf");
        Ds4Engine::write_synthetic_gguf(&path).expect("write synth gguf");
        let opts = Ds4EngineOptions {
            model_path: path.clone(),
            backend: Ds4Backend::Cuda,
            ..Ds4EngineOptions::default()
        };
        let e = Ds4Engine::open(opts).expect("open");
        assert!(e.model().is_some(), "fallback CPU path should load a model");
        assert!(e.summary().contains("backend=cpu"));
        assert!(!e.summary().contains("backend=cuda"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
