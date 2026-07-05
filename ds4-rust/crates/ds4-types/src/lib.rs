// DS4 (DwarfStar) -- shared leaf types.
//
// Holds the handle types, error/option enums, the Backend trait,
// and any other types that multiple high-level crates need to share
// WITHOUT creating a dependency cycle between them. This crate must
// remain leaf-level: no Eds4-*E deps allowed.
//
// All handle / option / error types live in Eds4-typesE so
// Eds4-coreE and Eds4-distE (and backends) can depend on it
// without forming a cycle.

pub const CRATE_NAME: &str = "ds4-types";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub type Ds4Result<T> = Result<T, Ds4Error>;

#[derive(Debug, Clone)]
pub struct Ds4Error {
    pub kind: Ds4ErrorKind,
    pub message: String,
}

impl Ds4Error {
    pub fn new(kind: Ds4ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Ds4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Ds4Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4ErrorKind {
    InvalidArgument,
    Io,
    Model,
    Tokenizer,
    Backend,
    KvStore,
    OutOfMemory,
    NotImplemented,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4Backend {
    Metal,
    Cuda,
    Rocm,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4ThinkMode {
    None,
    High,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4DistributedRole {
    None,
    Coordinator,
    Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum Ds4QuantKind {
    Q8_0,
    Q4_K,
    Q2_K,
    Iq2Xxs,
    F16,
    F32,
}

#[derive(Debug, Clone)]
pub struct Ds4LayerSlice {
    pub start: usize,
    pub end: usize,
    pub has_output: bool,
    pub set: bool,
}

#[derive(Debug, Clone)]
pub struct Ds4DistributedOptions {
    pub role: Ds4DistributedRole,
    pub layers: Ds4LayerSlice,
    pub listen_host: String,
    pub listen_port: u16,
    pub coordinator_host: String,
    pub coordinator_port: u16,
    pub prefill_chunk: usize,
    pub prefill_window: usize,
    pub activation_bits: u8,
    pub replay_check: bool,
    pub debug: bool,
}

impl Default for Ds4DistributedOptions {
    fn default() -> Self {
        Self {
            role: Ds4DistributedRole::None,
            layers: Ds4LayerSlice {
                start: 0,
                end: 0,
                has_output: false,
                set: false,
            },
            listen_host: String::new(),
            listen_port: 0,
            coordinator_host: String::new(),
            coordinator_port: 0,
            prefill_chunk: 0,
            prefill_window: 0,
            activation_bits: 32,
            replay_check: false,
            debug: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ds4EngineOptions {
    pub model_path: std::path::PathBuf,
    pub mtp_path: Option<std::path::PathBuf>,
    pub backend: Ds4Backend,
    pub n_threads: usize,
    pub prefill_chunk: usize,
    pub mtp_draft_tokens: usize,
    pub mtp_margin: f32,
    pub directional_steering_file: Option<std::path::PathBuf>,
    pub expert_profile_path: Option<std::path::PathBuf>,
    pub directional_steering_attn: f32,
    pub directional_steering_ffn: f32,
    pub power_percent: u8,
    pub ssd_streaming: bool,
    pub ssd_streaming_cache_experts: usize,
    pub ssd_streaming_cache_bytes: usize,
    pub ssd_streaming_preload_experts: usize,
    pub ssd_streaming_cold: bool,
    pub simulate_used_memory_bytes: Option<u64>,
    pub warm_weights: bool,
    pub quality: bool,
    pub inspect_only: bool,
    pub load_slice: Option<Ds4LayerSlice>,
    pub distributed: Option<Ds4DistributedOptions>,
}

impl Default for Ds4EngineOptions {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::new(),
            mtp_path: None,
            backend: Ds4Backend::Cpu,
            n_threads: 1,
            prefill_chunk: 512,
            mtp_draft_tokens: 0,
            mtp_margin: 0.0,
            directional_steering_file: None,
            expert_profile_path: None,
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            power_percent: 100,
            ssd_streaming: false,
            ssd_streaming_cache_experts: 0,
            ssd_streaming_cache_bytes: 0,
            ssd_streaming_preload_experts: 0,
            ssd_streaming_cold: false,
            simulate_used_memory_bytes: None,
            warm_weights: false,
            quality: false,
            inspect_only: false,
            load_slice: None,
            distributed: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4RewriteStatus {
    Ok,
    RewriteError,
    RebuildNeeded,
}

/// Opaque engine handle (mirrors the C engine handle).
#[derive(Debug)]
pub struct Ds4EngineHandle {
    _priv: (),
}

/// Opaque session handle (mirrors the C session handle).
#[derive(Debug)]
pub struct Ds4SessionHandle {
    _priv: (),
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn memory_estimate(ctx_size: usize, prefill_chunk: usize) -> u64
    where
        Self: Sized;

    /// Load model weights from a GGUF file at `path`.
    ///
    /// Returns a boxed `BackendModel` that the engine can drive through
    /// token logits, layer-slice, or output-head calls. Backends may choose a
    /// host-backed execution path when device-specific runtimes are absent.
    fn load_model(&self, path: &std::path::Path) -> Ds4Result<Box<dyn BackendModel>>;
}

pub trait BackendModel: Send + Sync {
    fn quant_kind(&self) -> Ds4QuantKind;

    fn eval_layer_slice(
        &self,
        tokens: &[u32],
        pos0: usize,
        layer_start: usize,
        layer_end: usize,
        input_hc: &[f32],
        output_hc: &mut [f32],
    ) -> Ds4Result<()> {
        let _ = (tokens, pos0);
        if layer_start == layer_end {
            if output_hc.len() < input_hc.len() {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!(
                        "output_hc too small: {} < {}",
                        output_hc.len(),
                        input_hc.len()
                    ),
                ));
            }
            output_hc[..input_hc.len()].copy_from_slice(input_hc);
            return Ok(());
        }
        Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "backend model does not implement layer-slice evaluation",
        ))
    }

    fn eval_output_head_from_hc(
        &self,
        hidden_hc: &[f32],
        n_tokens: usize,
        logits: &mut [f32],
    ) -> Ds4Result<()> {
        let _ = (hidden_hc, n_tokens, logits);
        Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "backend model does not implement output-head evaluation",
        ))
    }

    fn eval_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        let Some(&token) = tokens.last() else {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "token sequence is empty",
            ));
        };
        self.eval_token_logits(token, out)
    }

    fn eval_token_logits(&self, token: u32, out: &mut [f32]) -> Ds4Result<()> {
        let _ = (token, out);
        Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "backend model does not implement token logits",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-types");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn default_engine_options_are_sane() {
        let opts = Ds4EngineOptions::default();
        assert_eq!(opts.backend, Ds4Backend::Cpu);
    }

    #[test]
    fn default_distributed_options_are_sane() {
        let d = Ds4DistributedOptions::default();
        assert_eq!(d.role, Ds4DistributedRole::None);
        assert_eq!(d.activation_bits, 32);
    }
}
