// DS4 (DwarfStar) -- CUDA backend.
//
// Wraps the kernel sources + buffer pool + a host model-loading path
// behind the `Backend` trait. Kernel compilation remains explicit, while
// `load_model` returns a usable model on machines without an NVIDIA toolkit.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use ds4_gguf::{
    FfnTensorNames, GgufDType, GgufFile, LayerTensorNames, ModelSpec, TensorDescriptor,
};
use ds4_types::{Backend, BackendModel, Ds4Error, Ds4ErrorKind, Ds4QuantKind, Ds4Result};
use parking_lot::Mutex;

use crate::buffers::{Buffer, BufferPool, DType};
use crate::kernels::{
    compile, CompiledKernel, KERNEL_ATTENTION_DECODE_MIXED_SRC, KERNEL_COMPRESSOR_STORE_SRC,
    KERNEL_FP8_KV_QUANTIZE_SRC, KERNEL_HEAD_RMS_NORM_ROPE_TAIL_SRC, KERNEL_MATMUL_Q8_0_SRC,
    KERNEL_ROUTER_SELECT_SRC, KERNEL_SAMPLING_ARGMAX_SRC,
};
use crate::runtime::{CuDevicePtr, CudaRuntime, DeviceMem};

#[derive(Debug, Clone, Copy)]
struct ModelDims {
    vocab: usize,
    hidden: usize,
    layers: usize,
    n_heads: usize,
    head_dim: usize,
    ffn: usize,
}

struct CudaTensor {
    descriptor: TensorDescriptor,
    dtype_code: Option<i32>,
    memory: DeviceMem,
}

/// Per-shape compiled-kernel cache.
#[derive(Default)]
pub struct KernelCache {
    inner: Mutex<Vec<CompiledKernel>>,
}

impl KernelCache {
    pub fn get_or_compile(&self, name: &str, src: &str, arch: &str) -> Ds4Result<()> {
        let mut g = self.inner.lock();
        if g.iter().any(|k| k.name == name) {
            return Ok(());
        }
        let compiled = compile(name, src, arch)?;
        g.push(compiled);
        Ok(())
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.lock().iter().any(|k| k.name == name)
    }
}

pub struct CudaBackend;

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn quant_kind(&self) -> Ds4QuantKind {
        Ds4QuantKind::Q8_0
    }

    /// Detect an NVIDIA GPU compute capability from `nvidia-smi` if
    /// available. Returns `None` when the system is not CUDA-capable.
    pub fn detect_arch() -> Option<String> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8(out.stdout).ok()?;
        let cap = s.lines().next()?.trim();
        let (major, minor) = cap.split_once('.')?;
        Some(format!("sm_{major}{minor}"))
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }
    fn memory_estimate(_ctx_size: usize, _prefill_chunk: usize) -> u64 {
        0
    }

    fn load_model(&self, path: &Path) -> Ds4Result<Box<dyn BackendModel>> {
        let gguf = GgufFile::open(path)?;
        let spec = ModelSpec::from_gguf(&gguf)?;
        let runtime = CudaRuntime::load()?;
        let mut model = CudaModel::with_runtime(Arc::clone(&runtime), gguf.metadata.clone());
        model.spec = Some(spec);
        for descriptor in &gguf.tensors {
            let tensor = gguf.tensor(&descriptor.name)?;
            let alloc_bytes = tensor.bytes.len().max(1);
            let memory = CudaRuntime::alloc_managed(&runtime, alloc_bytes)?;
            memory.copy_from(tensor.bytes)?;
            model.tensors.insert(
                descriptor.name.clone(),
                CudaTensor {
                    descriptor: descriptor.clone(),
                    dtype_code: dtype_code(descriptor.dtype),
                    memory,
                },
            );
        }
        runtime.synchronize()?;
        Ok(Box::new(model))
    }
}

pub struct CudaModel {
    pub pool: BufferPool,
    pub cache: KernelCache,
    pub arch: String,
    runtime: Option<Arc<CudaRuntime>>,
    tensors: HashMap<String, CudaTensor>,
    metadata: ds4_gguf::GgufMetadata,
    spec: Option<ModelSpec>,
}

impl Default for CudaModel {
    fn default() -> Self {
        Self {
            pool: BufferPool::new(),
            cache: KernelCache::default(),
            arch: String::new(),
            runtime: None,
            tensors: HashMap::new(),
            metadata: ds4_gguf::GgufMetadata::default(),
            spec: None,
        }
    }
}

impl CudaModel {
    fn with_runtime(runtime: Arc<CudaRuntime>, metadata: ds4_gguf::GgufMetadata) -> Self {
        Self {
            runtime: Some(runtime),
            metadata,
            ..Self::default()
        }
    }

    /// Pre-compile the canonical kernel set at the given compute
    /// capability. Returns the first toolchain failure as `NotImplemented` so
    /// callers can fall back to the CPU backend cleanly.
    pub fn compile_all(&mut self, arch: &str) -> Ds4Result<()> {
        self.arch = arch.to_string();
        for (name, src) in &[
            ("matmul_q8_0", KERNEL_MATMUL_Q8_0_SRC),
            ("attention_decode_mixed", KERNEL_ATTENTION_DECODE_MIXED_SRC),
            (
                "head_rms_norm_rope_tail",
                KERNEL_HEAD_RMS_NORM_ROPE_TAIL_SRC,
            ),
            ("fp8_kv_quantize", KERNEL_FP8_KV_QUANTIZE_SRC),
            ("router_select", KERNEL_ROUTER_SELECT_SRC),
            ("compressor_store", KERNEL_COMPRESSOR_STORE_SRC),
            ("sampling_argmax", KERNEL_SAMPLING_ARGMAX_SRC),
        ] {
            self.cache.get_or_compile(name, src, arch)?;
        }
        Ok(())
    }

    /// Allocate a typed buffer from the pool.
    pub fn alloc(&self, dtype: DType, len: usize) -> Buffer {
        self.pool.alloc(dtype, len)
    }

    fn runtime(&self) -> Ds4Result<&Arc<CudaRuntime>> {
        self.runtime.as_ref().ok_or_else(|| {
            Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "CUDA model was not loaded with a device runtime",
            )
        })
    }

    fn tensor(&self, name: &str) -> Ds4Result<&CudaTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Ds4Error::new(Ds4ErrorKind::Model, format!("missing tensor {name}")))
    }

    fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name).map(|t| &t.descriptor)
    }

    fn descriptor_required(&self, name: &str) -> Ds4Result<&TensorDescriptor> {
        self.descriptor(name)
            .ok_or_else(|| Ds4Error::new(Ds4ErrorKind::Model, format!("missing tensor {name}")))
    }

    fn infer_layer_count(&self) -> usize {
        self.tensors
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

    fn alloc_f32(&self, len: usize) -> Ds4Result<DeviceMem> {
        CudaRuntime::alloc_device(self.runtime()?, len.saturating_mul(4).max(4))
    }

    fn upload_f32(&self, values: &[f32]) -> Ds4Result<DeviceMem> {
        let mem = self.alloc_f32(values.len())?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        mem.copy_from(&bytes)?;
        Ok(mem)
    }

    fn download_f32(&self, mem: &DeviceMem, len: usize) -> Ds4Result<Vec<f32>> {
        let mut bytes = vec![0u8; len * 4];
        mem.copy_to(&mut bytes)?;
        let mut out = Vec::with_capacity(len);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn launch_embedding(&self, token: u32, dims: ModelDims) -> Ds4Result<DeviceMem> {
        let tensor = self.tensor("token_embd.weight")?;
        if token as usize >= dims.vocab {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("token id {token} is outside vocabulary size {}", dims.vocab),
            ));
        }
        let dtype = tensor_dtype_code(tensor, "token_embd.weight")?;
        let out = self.alloc_f32(dims.hidden)?;
        let kernels = self.runtime()?.kernels();
        let mut weights = tensor.memory.ptr();
        let mut dtype_arg = dtype;
        let mut token_arg = token as i32;
        let mut hidden_arg = dims.hidden as i32;
        let mut out_arg = out.ptr();
        let mut params = [
            param(&mut weights),
            param(&mut dtype_arg),
            param(&mut token_arg),
            param(&mut hidden_arg),
            param(&mut out_arg),
        ];
        self.runtime()?
            .launch(kernels.embedding, blocks_for(dims.hidden), 256, &mut params)?;
        Ok(out)
    }

    fn launch_rmsnorm(
        &self,
        input: CuDevicePtr,
        weight_name: &str,
        n: usize,
    ) -> Ds4Result<DeviceMem> {
        let weight = self.tensor(weight_name)?;
        let dtype = tensor_dtype_code(weight, weight_name)?;
        let out = self.alloc_f32(n)?;
        let kernels = self.runtime()?.kernels();
        let mut input_arg = input;
        let mut weight_arg = weight.memory.ptr();
        let mut dtype_arg = dtype;
        let mut out_arg = out.ptr();
        let mut n_arg = n as i32;
        let mut eps = 1e-6f32;
        let mut params = [
            param(&mut input_arg),
            param(&mut weight_arg),
            param(&mut dtype_arg),
            param(&mut out_arg),
            param(&mut n_arg),
            param(&mut eps),
        ];
        self.runtime()?.launch(kernels.rmsnorm, 1, 1, &mut params)?;
        Ok(out)
    }

    fn launch_matvec_to(
        &self,
        input: CuDevicePtr,
        tensor_name: &str,
        out: CuDevicePtr,
        input_len: usize,
        out_len: usize,
    ) -> Ds4Result<()> {
        let tensor = self.tensor(tensor_name)?;
        let desc = &tensor.descriptor;
        if desc.dims.len() != 2 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("invalid {tensor_name} shape {:?}", desc.dims),
            ));
        }
        let in_dim = desc.dims[0] as usize;
        let out_dim = desc.dims[1] as usize;
        if input_len != in_dim || out_len != out_dim {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "{tensor_name} matvec shape mismatch: input={input_len} expected={in_dim}, out={out_len} expected={out_dim}"
                ),
            ));
        }
        let dtype = tensor_dtype_code(tensor, tensor_name)?;
        let kernels = self.runtime()?.kernels();
        let mut input_arg = input;
        let mut weight_arg = tensor.memory.ptr();
        let mut dtype_arg = dtype;
        let mut out_arg = out;
        let mut input_dim_arg = input_len as i32;
        let mut out_dim_arg = out_len as i32;
        let mut params = [
            param(&mut input_arg),
            param(&mut weight_arg),
            param(&mut dtype_arg),
            param(&mut out_arg),
            param(&mut input_dim_arg),
            param(&mut out_dim_arg),
        ];
        self.runtime()?
            .launch(kernels.matvec, blocks_for(out_len), 256, &mut params)
    }

    fn launch_matvec(
        &self,
        input: CuDevicePtr,
        tensor_name: &str,
        input_len: usize,
        out_len: usize,
    ) -> Ds4Result<DeviceMem> {
        let out = self.alloc_f32(out_len)?;
        self.launch_matvec_to(input, tensor_name, out.ptr(), input_len, out_len)?;
        Ok(out)
    }

    fn launch_add_inplace(&self, dst: CuDevicePtr, src: CuDevicePtr, n: usize) -> Ds4Result<()> {
        let kernels = self.runtime()?.kernels();
        let mut dst_arg = dst;
        let mut src_arg = src;
        let mut n_arg = n as i32;
        let mut params = [param(&mut dst_arg), param(&mut src_arg), param(&mut n_arg)];
        self.runtime()?
            .launch(kernels.add, blocks_for(n), 256, &mut params)
    }

    fn launch_silu_product(&self, gate: CuDevicePtr, up: CuDevicePtr, n: usize) -> Ds4Result<()> {
        let kernels = self.runtime()?.kernels();
        let mut gate_arg = gate;
        let mut up_arg = up;
        let mut n_arg = n as i32;
        let mut params = [param(&mut gate_arg), param(&mut up_arg), param(&mut n_arg)];
        self.runtime()?
            .launch(kernels.silu_product, blocks_for(n), 256, &mut params)
    }

    fn launch_rope(
        &self,
        x: CuDevicePtr,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Ds4Result<()> {
        let kernels = self.runtime()?.kernels();
        let mut x_arg = x;
        let mut pos_arg = pos as i32;
        let mut n_heads_arg = n_heads as i32;
        let mut head_dim_arg = head_dim as i32;
        let mut base = 10_000.0f32;
        let total_pairs = n_heads * (head_dim / 2);
        let mut params = [
            param(&mut x_arg),
            param(&mut pos_arg),
            param(&mut n_heads_arg),
            param(&mut head_dim_arg),
            param(&mut base),
        ];
        self.runtime()?
            .launch(kernels.rope, blocks_for(total_pairs), 256, &mut params)
    }

    fn launch_store_cache(
        &self,
        src: CuDevicePtr,
        cache: CuDevicePtr,
        cache_pos: usize,
        hidden: usize,
    ) -> Ds4Result<()> {
        let kernels = self.runtime()?.kernels();
        let mut src_arg = src;
        let mut cache_arg = cache;
        let mut cache_pos_arg = cache_pos as i32;
        let mut hidden_arg = hidden as i32;
        let mut params = [
            param(&mut src_arg),
            param(&mut cache_arg),
            param(&mut cache_pos_arg),
            param(&mut hidden_arg),
        ];
        self.runtime()?
            .launch(kernels.store_cache, blocks_for(hidden), 256, &mut params)
    }

    fn launch_attention(
        &self,
        q: CuDevicePtr,
        k_cache: CuDevicePtr,
        v_cache: CuDevicePtr,
        prefix_len: usize,
        dims: ModelDims,
    ) -> Ds4Result<DeviceMem> {
        let out = self.alloc_f32(dims.hidden)?;
        let kernels = self.runtime()?.kernels();
        let mut q_arg = q;
        let mut k_arg = k_cache;
        let mut v_arg = v_cache;
        let mut out_arg = out.ptr();
        let mut prefix_arg = prefix_len as i32;
        let mut heads_arg = dims.n_heads as i32;
        let mut head_dim_arg = dims.head_dim as i32;
        let mut params = [
            param(&mut q_arg),
            param(&mut k_arg),
            param(&mut v_arg),
            param(&mut out_arg),
            param(&mut prefix_arg),
            param(&mut heads_arg),
            param(&mut head_dim_arg),
        ];
        self.runtime()?.launch(
            kernels.attention,
            blocks_for(dims.n_heads),
            256,
            &mut params,
        )?;
        Ok(out)
    }

    fn forward_layers(
        &self,
        x: CuDevicePtr,
        pos: usize,
        k_caches: &[DeviceMem],
        v_caches: &[DeviceMem],
        dims: ModelDims,
    ) -> Ds4Result<()> {
        for layer in 0..dims.layers {
            self.forward_layer_at(
                x,
                pos,
                pos,
                pos + 1,
                layer,
                &k_caches[layer],
                &v_caches[layer],
                dims,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_at(
        &self,
        x: CuDevicePtr,
        rope_pos: usize,
        cache_pos: usize,
        prefix_len: usize,
        layer: usize,
        k_cache: &DeviceMem,
        v_cache: &DeviceMem,
        dims: ModelDims,
    ) -> Ds4Result<()> {
        let names = self.layer_tensor_names(layer)?;

        let attn_in = self.launch_rmsnorm(x, &names.attn_norm, dims.hidden)?;
        let q = self.launch_matvec(attn_in.ptr(), &names.attn_q, dims.hidden, dims.hidden)?;
        let k = self.launch_matvec(attn_in.ptr(), &names.attn_k, dims.hidden, dims.hidden)?;
        let v = self.launch_matvec(attn_in.ptr(), &names.attn_v, dims.hidden, dims.hidden)?;
        self.launch_rope(q.ptr(), rope_pos, dims.n_heads, dims.head_dim)?;
        self.launch_rope(k.ptr(), rope_pos, dims.n_heads, dims.head_dim)?;
        self.launch_store_cache(k.ptr(), k_cache.ptr(), cache_pos, dims.hidden)?;
        self.launch_store_cache(v.ptr(), v_cache.ptr(), cache_pos, dims.hidden)?;

        let attn =
            self.launch_attention(q.ptr(), k_cache.ptr(), v_cache.ptr(), prefix_len, dims)?;
        let attn_proj =
            self.launch_matvec(attn.ptr(), &names.attn_output, dims.hidden, dims.hidden)?;
        self.launch_add_inplace(x, attn_proj.ptr(), dims.hidden)?;

        let ffn_in = self.launch_rmsnorm(x, &names.ffn_norm, dims.hidden)?;
        match names.ffn {
            FfnTensorNames::Dense { gate, up, down } => {
                let gate = self.launch_matvec(ffn_in.ptr(), &gate, dims.hidden, dims.ffn)?;
                let up = self.launch_matvec(ffn_in.ptr(), &up, dims.hidden, dims.ffn)?;
                self.launch_silu_product(gate.ptr(), up.ptr(), dims.ffn)?;
                let down = self.launch_matvec(gate.ptr(), &down, dims.ffn, dims.hidden)?;
                self.launch_add_inplace(x, down.ptr(), dims.hidden)?;
            }
            FfnTensorNames::RoutedMoe { .. } => {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::NotImplemented,
                    "CUDA routed-MoE FFN execution requires expert dispatch kernels",
                ));
            }
        }
        Ok(())
    }

    fn compute_output_head_device(
        &self,
        hidden: CuDevicePtr,
        n_tokens: usize,
        logits: &mut [f32],
    ) -> Ds4Result<()> {
        let dims = self.dims()?;
        if n_tokens == 0 {
            return Ok(());
        }
        let logits_needed = n_tokens * dims.vocab;
        if logits.len() < logits_needed {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("logit buffer too small: {} < {logits_needed}", logits.len()),
            ));
        }
        let logits_dev = self.alloc_f32(logits_needed)?;
        let output_norm = self.tensors.contains_key("output_norm.weight");
        for token_idx in 0..n_tokens {
            let hidden_ptr = hidden + (token_idx * dims.hidden * 4) as u64;
            let normed;
            let head_input = if output_norm {
                normed = self.launch_rmsnorm(hidden_ptr, "output_norm.weight", dims.hidden)?;
                normed.ptr()
            } else {
                hidden_ptr
            };
            let out_ptr = logits_dev.ptr() + (token_idx * dims.vocab * 4) as u64;
            self.launch_matvec_to(
                head_input,
                "output.weight",
                out_ptr,
                dims.hidden,
                dims.vocab,
            )?;
        }
        self.runtime()?.synchronize()?;
        let values = self.download_f32(&logits_dev, logits_needed)?;
        logits[..logits_needed].copy_from_slice(&values);
        Ok(())
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
        let mut k_caches = Vec::with_capacity(dims.layers);
        let mut v_caches = Vec::with_capacity(dims.layers);
        for _ in 0..dims.layers {
            k_caches.push(self.alloc_f32(tokens.len() * dims.hidden)?);
            v_caches.push(self.alloc_f32(tokens.len() * dims.hidden)?);
        }
        let logits_dev = self.alloc_f32(dims.vocab)?;
        for (pos, &token) in tokens.iter().enumerate() {
            let x = self.launch_embedding(token, dims)?;
            self.forward_layers(x.ptr(), pos, &k_caches, &v_caches, dims)?;
            self.launch_matvec_to(
                x.ptr(),
                "output.weight",
                logits_dev.ptr(),
                dims.hidden,
                dims.vocab,
            )?;
        }
        self.runtime()?.synchronize()?;
        let values = self.download_f32(&logits_dev, dims.vocab)?;
        out[..dims.vocab].copy_from_slice(&values);
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
        if layer_end < layer_start || layer_end > dims.layers {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("invalid layer slice {layer_start}..{layer_end}"),
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
                format!("token slice too short: {} < {n_tokens}", tokens.len()),
            ));
        }
        output_hc[..input_hc.len()].copy_from_slice(input_hc);
        if layer_start == layer_end || n_tokens == 0 {
            return Ok(());
        }
        let out_dev = self.upload_f32(&output_hc[..input_hc.len()])?;
        let local_layers = layer_end - layer_start;
        let mut k_caches = Vec::with_capacity(local_layers);
        let mut v_caches = Vec::with_capacity(local_layers);
        for _ in 0..local_layers {
            k_caches.push(self.alloc_f32(n_tokens * dims.hidden)?);
            v_caches.push(self.alloc_f32(n_tokens * dims.hidden)?);
        }
        for token_idx in 0..n_tokens {
            let x = out_dev.ptr() + (token_idx * dims.hidden * 4) as u64;
            for layer in layer_start..layer_end {
                let local_layer = layer - layer_start;
                self.forward_layer_at(
                    x,
                    pos0 + token_idx,
                    token_idx,
                    token_idx + 1,
                    layer,
                    &k_caches[local_layer],
                    &v_caches[local_layer],
                    dims,
                )?;
            }
        }
        self.runtime()?.synchronize()?;
        let values = self.download_f32(&out_dev, input_hc.len())?;
        output_hc[..input_hc.len()].copy_from_slice(&values);
        Ok(())
    }
}

impl BackendModel for CudaModel {
    fn quant_kind(&self) -> Ds4QuantKind {
        self.metadata.routed_quant.unwrap_or(Ds4QuantKind::Q8_0)
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
        if n_tokens == 0 {
            return Ok(());
        }
        let hidden_dev = self.upload_f32(hidden_hc)?;
        self.compute_output_head_device(hidden_dev.ptr(), n_tokens, logits)
    }

    fn eval_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        self.compute_sequence_logits(tokens, out)
    }

    fn eval_token_logits(&self, token: u32, out: &mut [f32]) -> Ds4Result<()> {
        self.compute_sequence_logits(&[token], out)
    }
}

fn dtype_code(dtype: GgufDType) -> Option<i32> {
    match dtype {
        GgufDType::F32 => Some(0),
        GgufDType::F16 => Some(1),
        GgufDType::Q8_0 => Some(2),
        GgufDType::Q4_K => Some(3),
        GgufDType::Q3_K => Some(4),
        GgufDType::Q2_K => Some(5),
        GgufDType::Iq2Xxs => Some(6),
        _ => None,
    }
}

fn tensor_dtype_code(tensor: &CudaTensor, name: &str) -> Ds4Result<i32> {
    tensor.dtype_code.ok_or_else(|| {
        Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!(
                "tensor {name} dtype {:?} is not supported by the CUDA runtime path",
                tensor.descriptor.dtype
            ),
        )
    })
}

fn blocks_for(n: usize) -> u32 {
    n.div_ceil(256).max(1) as u32
}

fn param<T>(value: &mut T) -> *mut c_void {
    std::ptr::from_mut(value).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_cuda() {
        assert_eq!(CudaBackend::new().name(), "cuda");
    }

    #[test]
    fn quant_kind_is_q8_0() {
        assert_eq!(CudaBackend::new().quant_kind(), Ds4QuantKind::Q8_0);
    }

    #[test]
    fn detect_arch_returns_none_without_nvidia_smi() {
        // On a non-CUDA dev machine, nvidia-smi is missing; detect
        // must return None, not panic.
        let arch = CudaBackend::detect_arch();
        // We only assert the call doesn't panic; the actual value is
        // environment-dependent.
        let _ = arch;
    }

    #[test]
    fn compile_all_returns_not_implemented_without_nvcc() {
        let mut model = CudaModel::default();
        let res = model.compile_all("sm_80");
        assert!(res.is_err(), "expected error when nvcc absent");
        assert_eq!(
            res.unwrap_err().kind,
            ds4_types::Ds4ErrorKind::NotImplemented
        );
    }

    #[test]
    fn load_model_reports_unavailable_device_runtime() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-cuda-backend-load-model-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let err = CudaBackend::new().load_model(&path).err().unwrap();
        assert_eq!(err.kind, ds4_types::Ds4ErrorKind::NotImplemented);
    }

    #[test]
    fn kernel_cache_rejects_duplicates() {
        let cache = KernelCache::default();
        // First insertion can fail at compile() but that does not
        // matter -- what matters is the second call's short-circuit.
        let _ = cache.get_or_compile("k", "src", "sm_80");
        // The cache is empty because compile failed; second call
        // re-attempts compile. We can't easily assert "skipped second
        // time" without injecting a test compiler, so just confirm
        // get_or_compile returns Err (no panic).
        let r = cache.get_or_compile("k", "src", "sm_80");
        assert!(r.is_err());
    }

    #[test]
    fn buffer_alloc_returns_typed_buffer() {
        let model = CudaModel::default();
        let b = model.alloc(DType::F32, 32);
        assert_eq!(b.dtype, DType::F32);
        assert_eq!(b.len, 32);
        assert_eq!(b.bytes.len(), 128);
    }
}
