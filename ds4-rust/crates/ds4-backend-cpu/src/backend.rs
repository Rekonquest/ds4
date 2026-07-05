//
//
// `CpuBackend` implements the workspace-wide [`Backend`] trait
// from `ds4-types`. [`CpuModel`] is a thin in-memory cache of the
// tensors the CPU backend needs to run an inference pass.
//
// The CPU backend is the *correctness oracle* for the GPU
// backends. Every CPU kernel in this crate must produce numerically
// identical results to the C reference; the GPU implementations
// are validated against these outputs in their own test suites.

#![allow(clippy::module_name_repetitions, non_camel_case_types)]

use std::collections::HashMap;
use std::path::Path;

use crate::attention::attention_decode;
use crate::matmul::matmul_f32;
use crate::rmsnorm::rms_norm;
use crate::rope::apply_rope;
use ds4_gguf::{
    FfnTensorNames, GgufDType, GgufFile, LayerTensorNames, ModelSpec, TensorDescriptor,
};
use ds4_quant::f16::F16;
use ds4_quant::iq2_xxs::{self, Iq2XxsBlock};
use ds4_quant::q2_k::{self, Q2_KBlock};
use ds4_quant::q3_k::{self, Q3_KBlock};
use ds4_quant::q4_k::{self, Q4_KBlock};
use ds4_quant::q8_0::{self, Q8_0Block};
use ds4_types::{Backend, BackendModel, Ds4Error, Ds4ErrorKind, Ds4QuantKind, Ds4Result};

/// In-memory CPU-side model. Each named tensor is stored as a
/// contiguous `Vec<f32>` (the per-format F32 working type) keyed
/// by the GGUF tensor name. Tensors that are not F32 on disk are
/// stored as the raw quantized bytes so the matmul kernels can
/// route them through the quantized paths in `matmul`.
#[derive(Debug, Clone, Default)]
pub struct CpuModel {
    tensors: HashMap<String, TensorData>,
    descriptor_index: HashMap<String, TensorDescriptor>,
    metadata: ds4_gguf::GgufMetadata,
    spec: Option<ModelSpec>,
    n_elements: usize,
}

#[derive(Debug, Clone, Copy)]
struct ModelDims {
    vocab: usize,
    hidden: usize,
    layers: usize,
    n_heads: usize,
    head_dim: usize,
    ffn: usize,
}

/// In-RAM representation of a single tensor.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// f32 (dequantized or natively F32).
    F32(Vec<f32>),
    /// Q8_0 raw bytes (one block per 32 elements). The kernels in
    /// `matmul` decode these on the fly.
    Q8_0(Vec<u8>),
    /// Q4_K raw bytes (one block per 256 elements). Decoded
    /// on-the-fly by the Q4_K matmul kernel.
    Q4_K(Vec<u8>),
    /// IQ2_XXS raw bytes (passed through; no kernel in this
    /// backend understands the format yet).
    Iq2Xxs(Vec<u8>),
    /// f16 raw bytes (currently routed through dequant â†’ f32 on
    /// load; left here for SIMD-specialized paths).
    F16(Vec<u8>),
    /// Catch-all for GGUF dtypes the CPU backend does not currently
    /// understand. The bytes are stored verbatim so they can be
    /// re-examined by an offline tool without re-opening the file.
    Unknown(Vec<u8>),
}

impl TensorData {
    /// Total element count (after dequantization for non-f32
    /// formats) stored in this tensor.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Q8_0(_) => 0, // unknown without block count
            Self::Q4_K(_) => 0,
            Self::Iq2Xxs(_) => 0,
            Self::F16(_) => 0,
            Self::Unknown(_) => 0,
        }
    }

    /// Borrow the underlying byte buffer.
    ///
    /// For non-f32 tensors this returns the cached raw bytes
    /// verbatim. For f32 tensors the bytes are not exposed
    /// through this entry point because the workspace denies
    /// unsafe at the crate level; callers should use the
    /// [`Self::as_f32`] helper instead.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::F32(_) => &[],
            Self::Q8_0(v) | Self::Q4_K(v) | Self::Iq2Xxs(v) | Self::F16(v) | Self::Unknown(v) => {
                v.as_slice()
            }
        }
    }

    /// Borrow the underlying f32 data. Returns `Some(&[f32])`
    /// when the tensor was stored as F32 (either because the
    /// GGUF dtype was F32, or because it was decoded eagerly).
    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

impl CpuModel {
    /// Empty model. Useful for tests.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a tensor by name.
    pub fn insert(&mut self, name: impl Into<String>, data: TensorData) {
        self.tensors.insert(name.into(), data);
    }

    /// Borrow a tensor by name, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.tensors.get(name)
    }

    /// Mutably borrow a tensor by name, if present.
    #[must_use]
    pub fn get_mut(&mut self, name: &str) -> Option<&mut TensorData> {
        self.tensors.get_mut(name)
    }

    /// All tensor names currently cached.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Total number of tensors cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when no tensors have been loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total number of f32 elements across all tensors, including
    /// only the ones that have known f32 representations.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// Lookup tensor descriptor by name.
    #[must_use]
    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.descriptor_index.get(name)
    }

    fn f32_tensor(&self, name: &str) -> Ds4Result<&[f32]> {
        match self.get(name) {
            Some(TensorData::F32(values)) => Ok(values.as_slice()),
            Some(_) => Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!("tensor {name} is not available as f32"),
            )),
            None => Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("missing tensor {name}"),
            )),
        }
    }

    fn descriptor_required(&self, name: &str) -> Ds4Result<&TensorDescriptor> {
        self.descriptor(name)
            .ok_or_else(|| Ds4Error::new(Ds4ErrorKind::Model, format!("missing tensor {name}")))
    }

    fn infer_layer_count(&self) -> usize {
        self.descriptor_index
            .keys()
            .filter_map(|name| {
                let rest = name.strip_prefix("blk.")?;
                let (idx, _) = rest.split_once('.')?;
                idx.parse::<usize>().ok()
            })
            .max()
            .map_or(0, |idx| idx + 1)
    }

    fn dims(&self) -> Ds4Result<ModelDims> {
        if let Some(spec) = &self.spec {
            let dims = spec.dims;
            return Ok(ModelDims {
                vocab: dims.vocab,
                hidden: dims.hidden,
                layers: dims.layers,
                n_heads: dims.n_heads,
                head_dim: dims.head_dim,
                ffn: dims.ffn,
            });
        }
        let token_desc = self.descriptor_required("token_embd.weight")?;
        if token_desc.dims.len() != 2 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("invalid token_embd.weight shape {:?}", token_desc.dims),
            ));
        }
        let vocab = token_desc.dims[0] as usize;
        let hidden = token_desc.dims[1] as usize;
        if vocab == 0 || hidden == 0 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                "invalid token embedding shape",
            ));
        }
        let layers = self
            .metadata
            .layer_count
            .map_or_else(|| self.infer_layer_count(), |v| v as usize);
        let n_heads = self.metadata.head_count.unwrap_or(1) as usize;
        if n_heads == 0 {
            return Err(Ds4Error::new(Ds4ErrorKind::Model, "head count is zero"));
        }
        let head_dim = self
            .metadata
            .head_dim
            .map_or(hidden / n_heads, |v| v as usize);
        if layers != 0 && (head_dim == 0 || head_dim * n_heads != hidden) {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "attention dimensions do not cover hidden size: heads={n_heads} head_dim={head_dim} hidden={hidden}"
                ),
            ));
        }
        if layers != 0 && !head_dim.is_multiple_of(2) {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("head_dim must be even for RoPE, got {head_dim}"),
            ));
        }
        let ffn = self
            .descriptor("blk.0.ffn_gate.weight")
            .and_then(|d| d.dims.get(1).copied())
            .map_or(hidden * 4, |v| v as usize);
        Ok(ModelDims {
            vocab,
            hidden,
            layers,
            n_heads,
            head_dim,
            ffn,
        })
    }

    fn layer_tensor_names(&self, layer: usize) -> Ds4Result<LayerTensorNames> {
        if let Some(spec) = &self.spec {
            return spec.layer_tensors(layer);
        }
        let prefix = format!("blk.{layer}");
        Ok(LayerTensorNames {
            attn_norm: format!("{prefix}.attn_norm.weight"),
            attn_q: format!("{prefix}.attn_q.weight"),
            attn_k: format!("{prefix}.attn_k.weight"),
            attn_v: format!("{prefix}.attn_v.weight"),
            attn_output: format!("{prefix}.attn_out.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            ffn: FfnTensorNames::Dense {
                gate: format!("{prefix}.ffn_gate.weight"),
                up: format!("{prefix}.ffn_up.weight"),
                down: format!("{prefix}.ffn_down.weight"),
            },
        })
    }

    fn matvec_f32(&self, input: &[f32], tensor_name: &str, out: &mut [f32]) -> Ds4Result<()> {
        let desc = self.descriptor_required(tensor_name)?;
        if desc.dims.len() != 2 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("invalid {tensor_name} shape {:?}", desc.dims),
            ));
        }
        let in_dim = desc.dims[0] as usize;
        let out_dim = desc.dims[1] as usize;
        if input.len() != in_dim || out.len() != out_dim {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "{tensor_name} matvec shape mismatch: input={} expected={in_dim}, out={} expected={out_dim}",
                    input.len(),
                    out.len()
                ),
            ));
        }
        let weights = self.f32_tensor(tensor_name)?;
        let needed = in_dim * out_dim;
        if weights.len() < needed {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("{tensor_name} has {} values, needs {needed}", weights.len()),
            ));
        }
        matmul_f32(input, &weights[..needed], out, 1, out_dim, in_dim);
        Ok(())
    }

    fn embedding(&self, token: u32, dims: ModelDims) -> Ds4Result<Vec<f32>> {
        let token_embd = self.f32_tensor("token_embd.weight")?;
        let needed = dims.vocab * dims.hidden;
        if token_embd.len() < needed {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "token_embd.weight has {} values, needs {needed}",
                    token_embd.len()
                ),
            ));
        }
        let idx = token as usize;
        if idx >= dims.vocab {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("token id {token} is outside vocabulary size {}", dims.vocab),
            ));
        }
        Ok(token_embd[idx * dims.hidden..idx * dims.hidden + dims.hidden].to_vec())
    }

    fn compute_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        if tokens.is_empty() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "token sequence is empty",
            ));
        }
        let dims = self.dims()?;
        if out.len() < dims.vocab {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("logit buffer too small: {} < {}", out.len(), dims.vocab),
            ));
        }
        let mut k_caches = vec![vec![0.0f32; tokens.len() * dims.hidden]; dims.layers];
        let mut v_caches = vec![vec![0.0f32; tokens.len() * dims.hidden]; dims.layers];
        for (pos, &token) in tokens.iter().enumerate() {
            let mut x = self.embedding(token, dims)?;
            self.forward_layers(&mut x, pos, &mut k_caches, &mut v_caches, dims)?;
            self.compute_output_head_from_hc(&x, 1, &mut out[..dims.vocab])?;
        }
        Ok(())
    }

    fn compute_layer_slice(
        &self,
        tokens: &[u32],
        pos0: usize,
        layer_start: usize,
        layer_end: usize,
        input_hc: &[f32],
        output_hc: &mut [f32],
    ) -> Ds4Result<()> {
        let dims = self.dims()?;
        if layer_end < layer_start {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("invalid layer slice {layer_start}..{layer_end}"),
            ));
        }
        if dims.layers < layer_end {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "layer slice {layer_start}..{layer_end} exceeds model layer count {}",
                    dims.layers
                ),
            ));
        }
        if !input_hc.len().is_multiple_of(dims.hidden) {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "input_hc length {} is not a multiple of hidden {}",
                    input_hc.len(),
                    dims.hidden
                ),
            ));
        }
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
        let n_tokens = input_hc.len() / dims.hidden;
        if !tokens.is_empty() && tokens.len() < n_tokens {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("token slice too short: {} < {}", tokens.len(), n_tokens),
            ));
        }
        output_hc[..input_hc.len()].copy_from_slice(input_hc);
        if layer_start == layer_end || n_tokens == 0 {
            return Ok(());
        }

        let local_layers = layer_end - layer_start;
        let mut k_caches = vec![vec![0.0f32; n_tokens * dims.hidden]; local_layers];
        let mut v_caches = vec![vec![0.0f32; n_tokens * dims.hidden]; local_layers];
        for token_idx in 0..n_tokens {
            let start = token_idx * dims.hidden;
            let end = start + dims.hidden;
            let mut x = output_hc[start..end].to_vec();
            for layer in layer_start..layer_end {
                let local_layer = layer - layer_start;
                self.forward_layer_at(
                    &mut x,
                    pos0 + token_idx,
                    token_idx,
                    token_idx + 1,
                    layer,
                    &mut k_caches[local_layer],
                    &mut v_caches[local_layer],
                    dims,
                )?;
            }
            output_hc[start..end].copy_from_slice(&x);
        }
        Ok(())
    }

    fn compute_output_head_from_hc(
        &self,
        hidden_hc: &[f32],
        n_tokens: usize,
        logits: &mut [f32],
    ) -> Ds4Result<()> {
        let dims = self.dims()?;
        if n_tokens == 0 {
            if hidden_hc.is_empty() {
                return Ok(());
            }
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "n_tokens is zero but hidden_hc is not empty",
            ));
        }
        let hidden_needed = n_tokens * dims.hidden;
        if hidden_hc.len() < hidden_needed {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "hidden_hc too small: {} < {}",
                    hidden_hc.len(),
                    hidden_needed
                ),
            ));
        }
        let logits_needed = n_tokens * dims.vocab;
        if logits.len() < logits_needed {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "logit buffer too small: {} < {}",
                    logits.len(),
                    logits_needed
                ),
            ));
        }
        let output_norm = match self.get("output_norm.weight") {
            Some(TensorData::F32(weight)) => Some(weight.as_slice()),
            Some(_) => {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::NotImplemented,
                    "output_norm.weight is not available as f32",
                ));
            }
            None => None,
        };
        if let Some(weight) = output_norm {
            if weight.len() != dims.hidden {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::Model,
                    format!(
                        "output_norm.weight length {} != hidden {}",
                        weight.len(),
                        dims.hidden
                    ),
                ));
            }
        }
        for token_idx in 0..n_tokens {
            let hidden_start = token_idx * dims.hidden;
            let hidden_end = hidden_start + dims.hidden;
            let mut x = hidden_hc[hidden_start..hidden_end].to_vec();
            if let Some(weight) = output_norm {
                rms_norm(&mut x, weight, 1e-6);
            }
            let logits_start = token_idx * dims.vocab;
            let logits_end = logits_start + dims.vocab;
            self.matvec_f32(&x, "output.weight", &mut logits[logits_start..logits_end])?;
        }
        Ok(())
    }

    fn forward_layers(
        &self,
        x: &mut [f32],
        pos: usize,
        k_caches: &mut [Vec<f32>],
        v_caches: &mut [Vec<f32>],
        dims: ModelDims,
    ) -> Ds4Result<()> {
        for layer in 0..dims.layers {
            self.forward_layer(
                x,
                pos,
                layer,
                &mut k_caches[layer],
                &mut v_caches[layer],
                dims,
            )?;
        }
        Ok(())
    }

    fn forward_layer(
        &self,
        x: &mut [f32],
        pos: usize,
        layer: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        dims: ModelDims,
    ) -> Ds4Result<()> {
        self.forward_layer_at(x, pos, pos, pos + 1, layer, k_cache, v_cache, dims)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_at(
        &self,
        x: &mut [f32],
        rope_pos: usize,
        cache_pos: usize,
        prefix_len: usize,
        layer: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        dims: ModelDims,
    ) -> Ds4Result<()> {
        let names = self.layer_tensor_names(layer)?;

        let mut attn_in = x.to_vec();
        let attn_norm = self.f32_tensor(&names.attn_norm)?;
        if attn_norm.len() != dims.hidden {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("{} length mismatch", names.attn_norm),
            ));
        }
        rms_norm(&mut attn_in, attn_norm, 1e-6);

        let mut q = vec![0.0f32; dims.hidden];
        let mut k = vec![0.0f32; dims.hidden];
        let mut v = vec![0.0f32; dims.hidden];
        self.matvec_f32(&attn_in, &names.attn_q, &mut q)?;
        self.matvec_f32(&attn_in, &names.attn_k, &mut k)?;
        self.matvec_f32(&attn_in, &names.attn_v, &mut v)?;
        apply_rope(&mut q, rope_pos, dims.n_heads, dims.head_dim, 10_000.0);
        apply_rope(&mut k, rope_pos, dims.n_heads, dims.head_dim, 10_000.0);

        let cache_start = cache_pos * dims.hidden;
        let cache_end = cache_start + dims.hidden;
        let prefix_values = prefix_len * dims.hidden;
        if k_cache.len() < prefix_values || v_cache.len() < prefix_values {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "attention cache is too small for requested prefix",
            ));
        }
        k_cache[cache_start..cache_end].copy_from_slice(&k);
        v_cache[cache_start..cache_end].copy_from_slice(&v);

        let mut attn = vec![0.0f32; dims.hidden];
        attention_decode(
            &q,
            &k_cache[..prefix_values],
            &v_cache[..prefix_values],
            &mut attn,
            prefix_len,
            dims.n_heads,
            dims.head_dim,
        )
        .map_err(|message| Ds4Error::new(Ds4ErrorKind::Model, message))?;

        let mut attn_proj = vec![0.0f32; dims.hidden];
        self.matvec_f32(&attn, &names.attn_output, &mut attn_proj)?;
        add_inplace(x, &attn_proj);

        let mut ffn_in = x.to_vec();
        let ffn_norm = self.f32_tensor(&names.ffn_norm)?;
        if ffn_norm.len() != dims.hidden {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("{} length mismatch", names.ffn_norm),
            ));
        }
        rms_norm(&mut ffn_in, ffn_norm, 1e-6);

        match names.ffn {
            FfnTensorNames::Dense { gate, up, down } => {
                let mut gate_values = vec![0.0f32; dims.ffn];
                let mut up_values = vec![0.0f32; dims.ffn];
                self.matvec_f32(&ffn_in, &gate, &mut gate_values)?;
                self.matvec_f32(&ffn_in, &up, &mut up_values)?;
                for (g, u) in gate_values.iter_mut().zip(up_values.iter()) {
                    *g = silu(*g) * *u;
                }
                let mut down_values = vec![0.0f32; dims.hidden];
                self.matvec_f32(&gate_values, &down, &mut down_values)?;
                add_inplace(x, &down_values);
            }
            FfnTensorNames::RoutedMoe { .. } => {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::NotImplemented,
                    "CPU routed-MoE FFN execution requires expert dispatch",
                ));
            }
        }
        Ok(())
    }
}
fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// CPU backend handle. Implements [`Backend`] from `ds4-types`.
///
/// `CpuBackend` is a *value type*: construction is cheap
/// (`new()` is `Self`), so the GPU backend equivalents that need
/// to load device libraries use `Box<dyn Backend>` and call
/// `CpuBackend::new()` at the trait-object boundary.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend {
    _private: (),
}

impl CpuBackend {
    /// Construct a fresh CPU backend handle.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Load a model's tensor-data section from a GGUF file at `path`.
    /// Mirrors the C `ds4_engine_open` followed by a CPU-backend
    /// `load_model`. Returns a fresh `CpuModel` keyed by the GGUF
    /// tensor names. Real per-tensor loading happens here; the
    /// v0.3 narrow end-to-end test uses this to drive inference.
    ///
    /// * F32 tensors are decoded eagerly into `TensorData::F32`.
    /// * F16 and supported quantized tensors are decoded eagerly
    ///   into `TensorData::F32` for the CPU correctness path.
    /// * Unknown dtypes land in `TensorData::Unknown` verbatim.
    pub fn load_model(&self, path: &Path) -> Ds4Result<CpuModel> {
        let gguf = GgufFile::open(path)?;
        Self::load_model_from_gguf(&gguf)
    }

    /// Internal: build a `CpuModel` from an already-opened `GgufFile`.
    pub fn load_model_from_gguf(gguf: &GgufFile) -> Ds4Result<CpuModel> {
        let mut cache = CpuModel::new();
        let mut total_elements: usize = 0;
        cache.metadata = gguf.metadata.clone();
        cache.spec = Some(ModelSpec::from_gguf(gguf)?);

        for descriptor in &gguf.tensors {
            let n_elems: usize = descriptor.numel().try_into().map_err(|_| {
                Ds4Error::new(
                    Ds4ErrorKind::Model,
                    format!(
                        "tensor {:?}: element count overflows usize",
                        descriptor.name
                    ),
                )
            })?;

            let tensor = gguf.tensor(&descriptor.name)?;
            let data = match descriptor.dtype {
                GgufDType::F32 => {
                    let v = decode_f32(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                GgufDType::Q8_0 => {
                    let v = decode_q8_0(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                GgufDType::Q4_K => {
                    let v = decode_q4_k(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                GgufDType::Q3_K => {
                    let v = decode_q3_k(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                GgufDType::Q2_K => {
                    let v = decode_q2_k(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                GgufDType::Iq2Xxs => {
                    let v = decode_iq2_xxs(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                GgufDType::F16 => {
                    let v = decode_f16(&tensor)?;
                    total_elements += n_elems;
                    TensorData::F32(v)
                }
                _ => TensorData::Unknown(tensor.bytes.to_vec()),
            };

            cache.insert(&descriptor.name, data);
            cache
                .descriptor_index
                .insert(descriptor.name.clone(), descriptor.clone());
        }

        cache.n_elements = total_elements;
        Ok(cache)
    }
}

/// Decode an F32 tensor from a `QuantizedTensor` whose dtype is `F32`.
///
/// Safe Rust path: round-trips every 4 bytes through `f32::from_le_bytes`.
/// Returns an error if the dtype is wrong or the byte count is short.
fn decode_f32(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    if tensor.descriptor.dtype != GgufDType::F32 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("expected F32 tensor, got {:?}", tensor.descriptor.dtype),
        ));
    }
    let n = tensor.descriptor.numel() as usize;
    let bytes_needed = n * 4;
    if tensor.bytes.len() < bytes_needed {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!(
                "F32 tensor bytes {} < needed {}",
                tensor.bytes.len(),
                bytes_needed
            ),
        ));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in tensor.bytes[..bytes_needed].chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(b));
    }
    Ok(out)
}

fn decode_f16(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    if tensor.descriptor.dtype != GgufDType::F16 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("expected F16 tensor, got {:?}", tensor.descriptor.dtype),
        ));
    }
    let n = tensor.descriptor.numel() as usize;
    let bytes_needed = n * 2;
    if tensor.bytes.len() < bytes_needed {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!(
                "F16 tensor bytes {} < needed {}",
                tensor.bytes.len(),
                bytes_needed
            ),
        ));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in tensor.bytes[..bytes_needed].chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(F16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn decode_q8_0(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    let (n, blocks) = quant_shape(tensor, GgufDType::Q8_0, Q8_0Block::BLOCK_SIZE, 34)?;
    let mut out = Vec::with_capacity(n);
    let mut scratch = [0.0f32; Q8_0Block::BLOCK_SIZE];
    for block_idx in 0..blocks {
        let off = block_idx * 34;
        let mut qs = [0i8; 32];
        for (idx, slot) in qs.iter_mut().enumerate() {
            *slot = tensor.bytes[off + 2 + idx] as i8;
        }
        let block = Q8_0Block {
            d: read_f16(tensor.bytes, off),
            qs,
        };
        q8_0::dequantize(&block, &mut scratch);
        extend_logical(&mut out, &scratch, n);
    }
    Ok(out)
}

fn decode_q4_k(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    let (n, blocks) = quant_shape(tensor, GgufDType::Q4_K, Q4_KBlock::BLOCK_SIZE, 144)?;
    let mut out = Vec::with_capacity(n);
    let mut scratch = [0.0f32; Q4_KBlock::BLOCK_SIZE];
    for block_idx in 0..blocks {
        let off = block_idx * 144;
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        scales.copy_from_slice(&tensor.bytes[off + 4..off + 16]);
        qs.copy_from_slice(&tensor.bytes[off + 16..off + 144]);
        let block = Q4_KBlock {
            d: read_f16(tensor.bytes, off),
            dmin: read_f16(tensor.bytes, off + 2),
            scales,
            qs,
        };
        q4_k::dequantize(&block, &mut scratch);
        extend_logical(&mut out, &scratch, n);
    }
    Ok(out)
}

fn decode_q3_k(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    let (n, blocks) = quant_shape(tensor, GgufDType::Q3_K, Q3_KBlock::BLOCK_SIZE, 110)?;
    let mut out = Vec::with_capacity(n);
    let mut scratch = [0.0f32; Q3_KBlock::BLOCK_SIZE];
    for block_idx in 0..blocks {
        let off = block_idx * 110;
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 12];
        hmask.copy_from_slice(&tensor.bytes[off..off + 32]);
        qs.copy_from_slice(&tensor.bytes[off + 32..off + 96]);
        scales.copy_from_slice(&tensor.bytes[off + 96..off + 108]);
        let block = Q3_KBlock {
            d: read_f16(tensor.bytes, off + 108),
            hmask,
            qs,
            scales,
        };
        q3_k::dequantize(&block, &mut scratch);
        extend_logical(&mut out, &scratch, n);
    }
    Ok(out)
}

fn decode_q2_k(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    let (n, blocks) = quant_shape(tensor, GgufDType::Q2_K, Q2_KBlock::BLOCK_SIZE, 84)?;
    let mut out = Vec::with_capacity(n);
    let mut scratch = [0.0f32; Q2_KBlock::BLOCK_SIZE];
    for block_idx in 0..blocks {
        let off = block_idx * 84;
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        scales.copy_from_slice(&tensor.bytes[off..off + 16]);
        qs.copy_from_slice(&tensor.bytes[off + 16..off + 80]);
        let block = Q2_KBlock {
            d: read_f16(tensor.bytes, off + 80),
            dmin: read_f16(tensor.bytes, off + 82),
            scales,
            qs,
        };
        q2_k::dequantize(&block, &mut scratch);
        extend_logical(&mut out, &scratch, n);
    }
    Ok(out)
}

fn decode_iq2_xxs(tensor: &ds4_gguf::QuantizedTensor<'_>) -> Ds4Result<Vec<f32>> {
    let (n, blocks) = quant_shape(tensor, GgufDType::Iq2Xxs, Iq2XxsBlock::BLOCK_SIZE, 66)?;
    let mut out = Vec::with_capacity(n);
    let mut scratch = [0.0f32; Iq2XxsBlock::BLOCK_SIZE];
    for block_idx in 0..blocks {
        let off = block_idx * 66;
        let mut qs = [0u16; 32];
        for (idx, slot) in qs.iter_mut().enumerate() {
            let qoff = off + 2 + idx * 2;
            *slot = u16::from_le_bytes([tensor.bytes[qoff], tensor.bytes[qoff + 1]]);
        }
        let block = Iq2XxsBlock {
            d: read_f16(tensor.bytes, off),
            qs,
        };
        iq2_xxs::dequantize(&block, &mut scratch);
        extend_logical(&mut out, &scratch, n);
    }
    Ok(out)
}

fn quant_shape(
    tensor: &ds4_gguf::QuantizedTensor<'_>,
    expected: GgufDType,
    block_size: usize,
    block_bytes: usize,
) -> Ds4Result<(usize, usize)> {
    if tensor.descriptor.dtype != expected {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!(
                "expected {expected:?} tensor, got {:?}",
                tensor.descriptor.dtype
            ),
        ));
    }
    let n = tensor.descriptor.numel() as usize;
    let blocks = n.div_ceil(block_size);
    let bytes_needed = blocks * block_bytes;
    if tensor.bytes.len() < bytes_needed {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!(
                "{expected:?} tensor bytes {} < needed {}",
                tensor.bytes.len(),
                bytes_needed
            ),
        ));
    }
    Ok((n, blocks))
}

fn read_f16(bytes: &[u8], off: usize) -> f32 {
    F16::from_bits(u16::from_le_bytes([bytes[off], bytes[off + 1]])).to_f32()
}

fn extend_logical(out: &mut Vec<f32>, scratch: &[f32], n: usize) {
    let remaining = n.saturating_sub(out.len());
    let take = remaining.min(scratch.len());
    out.extend_from_slice(&scratch[..take]);
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn memory_estimate(ctx_size: usize, prefill_chunk: usize) -> u64 {
        // Context-only lower bound. Production callers should use the
        // higher-level `ds4_context_memory_estimate_with_prefill`
        // helper in `ds4-core` for the full accounting.
        let hidden_dim: u64 = 7168;
        let n_layers: u64 = 61;
        let vocab_size: u64 = 129280;
        let kv_bytes = ctx_size as u64 * hidden_dim * 2 * n_layers;
        let vocab_bytes = vocab_size * 4;
        let prefill_bytes = prefill_chunk as u64 * hidden_dim * 4;
        kv_bytes + vocab_bytes + prefill_bytes
    }

    fn load_model(&self, path: &Path) -> Ds4Result<Box<dyn BackendModel>> {
        let model = Self::load_model(self, path)?;
        Ok(Box::new(model))
    }
}

impl BackendModel for CpuModel {
    fn quant_kind(&self) -> Ds4QuantKind {
        Ds4QuantKind::F32
    }

    fn eval_layer_slice(
        &self,
        tokens: &[u32],
        pos0: usize,
        layer_start: usize,
        layer_end: usize,
        input_hc: &[f32],
        output_hc: &mut [f32],
    ) -> Ds4Result<()> {
        self.compute_layer_slice(tokens, pos0, layer_start, layer_end, input_hc, output_hc)
    }

    fn eval_output_head_from_hc(
        &self,
        hidden_hc: &[f32],
        n_tokens: usize,
        logits: &mut [f32],
    ) -> Ds4Result<()> {
        self.compute_output_head_from_hc(hidden_hc, n_tokens, logits)
    }

    fn eval_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        self.compute_sequence_logits(tokens, out)
    }

    fn eval_token_logits(&self, token: u32, out: &mut [f32]) -> Ds4Result<()> {
        self.compute_sequence_logits(&[token], out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_gguf::TensorDescriptor;

    fn empty_descriptor(name: &str, dims: Vec<u32>) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            dims,
            dtype: ds4_gguf::GgufDType::F32,
            offset: 0,
        }
    }

    #[test]
    fn cpu_model_insert_and_get() {
        let mut m = CpuModel::new();
        m.insert("a", TensorData::F32(vec![1.0, 2.0]));
        assert!(m.get("a").is_some());
        assert!(m.get("b").is_none());
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn tensor_data_f32_accessor_works() {
        let t = TensorData::F32(vec![3.0, 4.0]);
        assert_eq!(t.as_f32(), Some(&[3.0f32, 4.0][..]));
        assert_eq!(t.as_bytes(), &[] as &[u8]);
    }

    #[test]
    fn tensor_data_raw_bytes_accessor() {
        let t = TensorData::Q8_0(vec![1, 2, 3]);
        assert_eq!(t.as_bytes(), &[1u8, 2, 3][..]);
        assert_eq!(t.as_f32(), None);
    }

    #[test]
    fn descriptor_lookup_returns_inserted() {
        let mut m = CpuModel::new();
        m.descriptor_index
            .insert("foo".into(), empty_descriptor("foo", vec![2, 2]));
        assert!(m.descriptor("foo").is_some());
    }

    #[test]
    fn backend_name_is_cpu() {
        assert_eq!(CpuBackend::new().name(), "cpu");
    }

    #[test]
    fn memory_estimate_is_positive() {
        let n = CpuBackend::memory_estimate(8192, 512);
        assert!(n > 0);
    }

    #[test]
    fn cpu_model_quant_kind_is_f32() {
        let m = CpuModel::new();
        assert_eq!(m.quant_kind(), Ds4QuantKind::F32);
    }

    #[test]
    fn cpu_model_eval_token_logits_from_synthetic_gguf() {
        let dir = std::env::temp_dir().join("ds4-backend-cpu-logits-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let model = CpuBackend::new().load_model(&path).unwrap();
        let mut logits = vec![0.0f32; 16];
        model.eval_token_logits(1, &mut logits).unwrap();
        assert!(logits.iter().any(|v| *v != 0.0));
        assert_eq!(logits.len(), 16);
    }

    #[test]
    fn cpu_model_rejects_token_outside_vocab() {
        let dir = std::env::temp_dir().join("ds4-backend-cpu-token-bounds-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let model = CpuBackend::new().load_model(&path).unwrap();
        let mut logits = vec![0.0f32; 16];
        let err = model.eval_token_logits(16, &mut logits).err().unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn cpu_model_eval_sequence_logits_uses_transformer_layer() {
        let dir = std::env::temp_dir().join("ds4-backend-cpu-sequence-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let mut model = CpuBackend::new().load_model(&path).unwrap();
        let mut original = vec![0.0f32; 16];
        model
            .eval_sequence_logits(&[8, 9, 10], &mut original)
            .unwrap();

        for name in ["blk.0.attn_out.weight", "blk.0.ffn_down.weight"] {
            let tensor = model.get_mut(name).unwrap();
            let TensorData::F32(values) = tensor else {
                panic!("{name} should be f32");
            };
            values.fill(0.0);
        }

        let mut without_layer_projection = vec![0.0f32; 16];
        model
            .eval_sequence_logits(&[8, 9, 10], &mut without_layer_projection)
            .unwrap();
        let max_diff = original
            .iter()
            .zip(without_layer_projection.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-5,
            "sequence logits did not change after zeroing layer projections"
        );
    }

    #[test]
    fn cpu_model_eval_layer_slice_matches_full_layer_for_single_token() {
        let dir = std::env::temp_dir().join("ds4-backend-cpu-layer-slice-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let model = CpuBackend::new().load_model(&path).unwrap();
        let dims = model.dims().unwrap();
        let input = model.embedding(8, dims).unwrap();
        let mut sliced = vec![0.0f32; dims.hidden];
        model
            .eval_layer_slice(&[8], 0, 0, dims.layers, &input, &mut sliced)
            .unwrap();

        let mut full = input.clone();
        let mut k_caches = vec![vec![0.0f32; dims.hidden]; dims.layers];
        let mut v_caches = vec![vec![0.0f32; dims.hidden]; dims.layers];
        model
            .forward_layers(&mut full, 0, &mut k_caches, &mut v_caches, dims)
            .unwrap();
        let max_diff = full
            .iter()
            .zip(sliced.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "layer slice mismatch: {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cpu_model_output_head_from_hc_matches_token_logits() {
        let dir = std::env::temp_dir().join("ds4-backend-cpu-output-head-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let model = CpuBackend::new().load_model(&path).unwrap();
        let dims = model.dims().unwrap();
        let mut hidden = model.embedding(8, dims).unwrap();
        let mut k_caches = vec![vec![0.0f32; dims.hidden]; dims.layers];
        let mut v_caches = vec![vec![0.0f32; dims.hidden]; dims.layers];
        model
            .forward_layers(&mut hidden, 0, &mut k_caches, &mut v_caches, dims)
            .unwrap();

        let mut from_head = vec![0.0f32; dims.vocab];
        model
            .eval_output_head_from_hc(&hidden, 1, &mut from_head)
            .unwrap();
        let mut direct = vec![0.0f32; dims.vocab];
        model.eval_token_logits(8, &mut direct).unwrap();
        let max_diff = direct
            .iter()
            .zip(from_head.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "output head mismatch: {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_f16_tensor_to_f32_values() {
        let values = [F16::from_f32(1.5), F16::from_f32(-2.0)];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let descriptor = TensorDescriptor {
            name: "half.weight".to_string(),
            dims: vec![2],
            dtype: ds4_gguf::GgufDType::F16,
            offset: 0,
        };
        let tensor = ds4_gguf::QuantizedTensor {
            descriptor,
            bytes: &bytes,
        };
        let decoded = decode_f16(&tensor).unwrap();
        assert_eq!(decoded, vec![1.5, -2.0]);
    }

    #[test]
    fn decode_q8_0_tensor_to_f32_values() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&F16::from_f32(0.5).to_bits().to_le_bytes());
        let mut qs = [0u8; 32];
        qs[0] = 2u8;
        qs[1] = (-4i8) as u8;
        bytes.extend_from_slice(&qs);
        let descriptor = TensorDescriptor {
            name: "q8.weight".to_string(),
            dims: vec![2],
            dtype: ds4_gguf::GgufDType::Q8_0,
            offset: 0,
        };
        let tensor = ds4_gguf::QuantizedTensor {
            descriptor,
            bytes: &bytes,
        };
        let decoded = decode_q8_0(&tensor).unwrap();
        assert_eq!(decoded, vec![1.0, -2.0]);
    }

    #[test]
    fn decode_q4_k_tensor_matches_quant_kernel() {
        let mut input = [0.0f32; Q4_KBlock::BLOCK_SIZE];
        input[0] = 1.0;
        input[1] = -2.0;
        input[2] = 3.0;
        let block = q4_k::quantize(&input);
        let raw_block = Q4_KBlock {
            d: F16::from_f32(block.d).to_f32(),
            dmin: F16::from_f32(block.dmin).to_f32(),
            scales: block.scales,
            qs: block.qs,
        };
        let mut expected = [0.0f32; Q4_KBlock::BLOCK_SIZE];
        q4_k::dequantize(&raw_block, &mut expected);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&F16::from_f32(block.d).to_bits().to_le_bytes());
        bytes.extend_from_slice(&F16::from_f32(block.dmin).to_bits().to_le_bytes());
        bytes.extend_from_slice(&block.scales);
        bytes.extend_from_slice(&block.qs);
        let descriptor = TensorDescriptor {
            name: "q4.weight".to_string(),
            dims: vec![Q4_KBlock::BLOCK_SIZE as u32],
            dtype: ds4_gguf::GgufDType::Q4_K,
            offset: 0,
        };
        let tensor = ds4_gguf::QuantizedTensor {
            descriptor,
            bytes: &bytes,
        };
        let decoded = decode_q4_k(&tensor).unwrap();
        assert_eq!(decoded, expected.to_vec());
    }

    type TestQ3ScaleBytes = [u8; 12];
    type TestQ3SignedScales = [i8; 16];

    fn pack_q3_scales(scales: TestQ3SignedScales) -> TestQ3ScaleBytes {
        let mut out = [0u8; 12];
        for (j, scale) in scales.iter().enumerate() {
            let packed = (*scale + 32) as u8;
            if j < 8 {
                out[j] = packed & 0x0f;
            } else {
                out[j - 8] |= (packed & 0x0f) << 4;
            }
            out[j % 4 + 8] |= (packed >> 4) << (2 * (j / 4));
        }
        out
    }

    #[test]
    fn decode_q3_k_tensor_to_f32_values() {
        let mut signed_scales = [0i8; 16];
        signed_scales[0] = 2;
        let scales = pack_q3_scales(signed_scales);
        let mut hmask = [0u8; 32];
        for mask in hmask.iter_mut().take(16) {
            *mask = 1;
        }
        let mut qs = [0u8; 64];
        qs[0] = 3;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&hmask);
        bytes.extend_from_slice(&qs);
        bytes.extend_from_slice(&scales);
        bytes.extend_from_slice(&F16::from_f32(0.5).to_bits().to_le_bytes());
        let descriptor = TensorDescriptor {
            name: "q3.weight".to_string(),
            dims: vec![Q3_KBlock::BLOCK_SIZE as u32],
            dtype: ds4_gguf::GgufDType::Q3_K,
            offset: 0,
        };
        let tensor = ds4_gguf::QuantizedTensor {
            descriptor,
            bytes: &bytes,
        };
        let decoded = decode_q3_k(&tensor).unwrap();
        assert_eq!(decoded[0], 3.0);
        assert!(decoded[1..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn decode_q2_k_zero_block_is_zero() {
        let bytes = vec![0u8; 84];
        let descriptor = TensorDescriptor {
            name: "q2.weight".to_string(),
            dims: vec![Q2_KBlock::BLOCK_SIZE as u32],
            dtype: ds4_gguf::GgufDType::Q2_K,
            offset: 0,
        };
        let tensor = ds4_gguf::QuantizedTensor {
            descriptor,
            bytes: &bytes,
        };
        let decoded = decode_q2_k(&tensor).unwrap();
        assert!(decoded.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn decode_iq2_xxs_zero_block_is_zero() {
        let bytes = vec![0u8; 66];
        let descriptor = TensorDescriptor {
            name: "iq2.weight".to_string(),
            dims: vec![Iq2XxsBlock::BLOCK_SIZE as u32],
            dtype: ds4_gguf::GgufDType::Iq2Xxs,
            offset: 0,
        };
        let tensor = ds4_gguf::QuantizedTensor {
            descriptor,
            bytes: &bytes,
        };
        let decoded = decode_iq2_xxs(&tensor).unwrap();
        assert!(decoded.iter().all(|v| *v == 0.0));
    }
}
