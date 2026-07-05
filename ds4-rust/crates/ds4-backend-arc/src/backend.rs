use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ds4_gguf::{
    ExpertTensorLayout, FfnTensorNames, GgufDType, GgufFile, ModelDims, ModelSpec,
    TensorDescriptor, TransformerBlockKind,
};
use ds4_types::{Backend, BackendModel, Ds4Error, Ds4ErrorKind, Ds4QuantKind, Ds4Result};

use crate::runtime::{ArcMem, ArcRuntime, KernelId};

pub struct ArcBackend;

impl Default for ArcBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ArcBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for ArcBackend {
    fn name(&self) -> &'static str {
        "arc"
    }

    fn memory_estimate(ctx_size: usize, prefill_chunk: usize) -> u64 {
        let ctx = ctx_size.max(1) as u64;
        let prefill = prefill_chunk.max(1) as u64;
        ctx.saturating_mul(512)
            .saturating_add(prefill.saturating_mul(256))
            .saturating_add(4096)
    }

    fn load_model(&self, path: &Path) -> Ds4Result<Box<dyn BackendModel>> {
        let gguf = GgufFile::open(path)?;
        let spec = ModelSpec::from_gguf(&gguf)?;
        let runtime = ArcRuntime::load()?;
        let metadata = gguf.metadata.clone();
        let mut model = ArcModel {
            runtime,
            spec,
            metadata,
            descriptors: HashMap::new(),
            tensors: HashMap::new(),
            gguf,
            launch_lock: Mutex::new(()),
        };
        model.index_descriptors();
        if model.spec.block_kind == TransformerBlockKind::Dense {
            model.upload_dense_f32_tensors()?;
        } else {
            model.upload_resident_moe_f32_tensors()?;
        }
        Ok(Box::new(model))
    }
}

struct ArcTensor {
    descriptor: TensorDescriptor,
    memory: ArcMem,
}

struct ExpertTensorSet {
    gate_name: String,
    gate: ArcTensor,
    up_name: String,
    up: ArcTensor,
    down_name: String,
    down: ArcTensor,
}

pub struct ArcModel {
    runtime: Arc<ArcRuntime>,
    pub spec: ModelSpec,
    metadata: ds4_gguf::GgufMetadata,
    descriptors: HashMap<String, TensorDescriptor>,
    tensors: HashMap<String, ArcTensor>,
    gguf: GgufFile,
    launch_lock: Mutex<()>,
}

impl ArcModel {
    pub fn runtime_device_name(&self) -> &str {
        self.runtime.device_name()
    }

    pub fn runtime_platform_name(&self) -> &str {
        self.runtime.platform_name()
    }

    pub fn resident_tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }

    fn index_descriptors(&mut self) {
        self.descriptors.clear();
        for descriptor in &self.gguf.tensors {
            self.descriptors
                .insert(descriptor.name.clone(), descriptor.clone());
        }
    }

    fn upload_dense_f32_tensors(&mut self) -> Ds4Result<()> {
        for descriptor in self.gguf.tensors.clone() {
            if dtype_code(descriptor.dtype).is_none() {
                continue;
            }
            self.upload_resident_tensor(&descriptor)?;
        }
        self.runtime.finish()
    }

    fn upload_resident_moe_f32_tensors(&mut self) -> Ds4Result<()> {
        for descriptor in self.gguf.tensors.clone() {
            if dtype_code(descriptor.dtype).is_none() || is_routed_expert_tensor(&descriptor.name) {
                continue;
            }
            self.upload_resident_tensor(&descriptor)?;
        }
        self.runtime.finish()
    }

    fn upload_resident_tensor(&mut self, descriptor: &TensorDescriptor) -> Ds4Result<()> {
        let tensor = self.gguf.tensor(&descriptor.name)?;
        let memory = ArcRuntime::from_bytes(&self.runtime, tensor.bytes)?;
        self.tensors.insert(
            descriptor.name.clone(),
            ArcTensor {
                descriptor: descriptor.clone(),
                memory,
            },
        );
        Ok(())
    }

    fn dims(&self) -> ModelDims {
        self.spec.dims
    }

    fn tensor(&self, name: &str) -> Ds4Result<&ArcTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Ds4Error::new(Ds4ErrorKind::Model, format!("missing Arc tensor {name}")))
    }

    fn tensor_supported(&self, name: &str) -> Ds4Result<&ArcTensor> {
        let tensor = self.tensor(name)?;
        if dtype_code(tensor.descriptor.dtype).is_some() {
            Ok(tensor)
        } else {
            Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!(
                    "Arc execution cannot consume {name} dtype {:?}",
                    tensor.descriptor.dtype
                ),
            ))
        }
    }

    fn descriptor(&self, name: &str) -> Ds4Result<&TensorDescriptor> {
        self.descriptors.get(name).ok_or_else(|| {
            Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("missing tensor descriptor {name}"),
            )
        })
    }

    fn load_transient_supported_tensor(&self, name: &str) -> Ds4Result<ArcTensor> {
        let descriptor = self.descriptor(name)?;
        if dtype_code(descriptor.dtype).is_none() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!(
                    "Arc routed-MoE selected tensor {name} has unsupported dtype {:?}",
                    descriptor.dtype
                ),
            ));
        }
        let tensor = self.gguf.tensor(name)?;
        let memory = ArcRuntime::from_bytes(&self.runtime, tensor.bytes)?;
        Ok(ArcTensor {
            descriptor: descriptor.clone(),
            memory,
        })
    }

    fn alloc_f32(&self, len: usize) -> Ds4Result<ArcMem> {
        ArcRuntime::alloc(&self.runtime, len.saturating_mul(4).max(4))
    }

    fn upload_f32(&self, values: &[f32]) -> Ds4Result<ArcMem> {
        ArcRuntime::from_bytes(&self.runtime, &f32_slice_to_bytes(values))
    }

    fn download_f32(&self, mem: &ArcMem, len: usize) -> Ds4Result<Vec<f32>> {
        let mut bytes = vec![0u8; len.saturating_mul(4)];
        mem.read(&mut bytes)?;
        Ok(bytes_to_f32_vec(&bytes))
    }

    fn launch_embedding(&self, token: u32, dims: ModelDims) -> Ds4Result<ArcMem> {
        let token_embd = self.tensor_supported(&self.spec.token_embedding)?;
        if token as usize >= dims.vocab {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("token id {token} is outside vocabulary size {}", dims.vocab),
            ));
        }
        let out = self.alloc_f32(dims.hidden)?;
        let dtype = tensor_dtype_code(token_embd, &self.spec.token_embedding)?;
        self.with_kernel(KernelId::EmbeddingWeight, dims.hidden, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, &token_embd.memory)?;
            runtime.set_arg_i32(kernel, 1, dtype)?;
            runtime.set_arg_u32(kernel, 2, token)?;
            runtime.set_arg_i32(kernel, 3, dims.hidden as i32)?;
            runtime.set_arg_mem(kernel, 4, &out)
        })?;
        Ok(out)
    }

    fn launch_rmsnorm(&self, input: &ArcMem, weight_name: &str, n: usize) -> Ds4Result<ArcMem> {
        let weight = self.tensor_supported(weight_name)?;
        let out = self.alloc_f32(n)?;
        let dtype = tensor_dtype_code(weight, weight_name)?;
        self.with_kernel(KernelId::RmsNormWeight, n, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, input)?;
            runtime.set_arg_mem(kernel, 1, &weight.memory)?;
            runtime.set_arg_i32(kernel, 2, dtype)?;
            runtime.set_arg_i32(kernel, 3, n as i32)?;
            runtime.set_arg_f32(kernel, 4, self.spec.dims.rms_norm_epsilon)?;
            runtime.set_arg_mem(kernel, 5, &out)
        })?;
        Ok(out)
    }

    fn launch_matvec(&self, input: &ArcMem, tensor_name: &str) -> Ds4Result<ArcMem> {
        let tensor = self.tensor_supported(tensor_name)?;
        self.launch_matvec_tensor(input, tensor_name, tensor)
    }

    fn launch_matvec_tensor(
        &self,
        input: &ArcMem,
        tensor_name: &str,
        tensor: &ArcTensor,
    ) -> Ds4Result<ArcMem> {
        let dims = matvec_dims(&tensor.descriptor, tensor_name)?;
        let out = self.alloc_f32(dims.1)?;
        self.launch_matvec_tensor_to(input, tensor_name, tensor, &out, dims.0, dims.1)?;
        Ok(out)
    }

    fn launch_matvec_to(
        &self,
        input: &ArcMem,
        tensor_name: &str,
        out: &ArcMem,
        in_dim: usize,
        out_dim: usize,
    ) -> Ds4Result<()> {
        let tensor = self.tensor_supported(tensor_name)?;
        self.launch_matvec_tensor_to(input, tensor_name, tensor, out, in_dim, out_dim)
    }

    fn launch_matvec_tensor_to(
        &self,
        input: &ArcMem,
        tensor_name: &str,
        tensor: &ArcTensor,
        out: &ArcMem,
        in_dim: usize,
        out_dim: usize,
    ) -> Ds4Result<()> {
        let (actual_in, actual_out) = matvec_dims(&tensor.descriptor, tensor_name)?;
        if actual_in != in_dim || actual_out != out_dim {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "{tensor_name} matvec shape mismatch: got {actual_in}x{actual_out}, expected {in_dim}x{out_dim}"
                ),
            ));
        }
        let dtype = tensor_dtype_code(tensor, tensor_name)?;
        self.with_kernel(KernelId::MatvecWeight, out_dim, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, input)?;
            runtime.set_arg_mem(kernel, 1, &tensor.memory)?;
            runtime.set_arg_i32(kernel, 2, dtype)?;
            runtime.set_arg_i32(kernel, 3, in_dim as i32)?;
            runtime.set_arg_i32(kernel, 4, out_dim as i32)?;
            runtime.set_arg_mem(kernel, 5, out)
        })
    }

    fn launch_add_inplace(&self, dst: &ArcMem, src: &ArcMem, n: usize) -> Ds4Result<()> {
        self.with_kernel(KernelId::AddInplaceF32, n, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, dst)?;
            runtime.set_arg_mem(kernel, 1, src)?;
            runtime.set_arg_i32(kernel, 2, n as i32)
        })
    }

    fn launch_add_scaled_inplace(
        &self,
        dst: &ArcMem,
        src: &ArcMem,
        scale: f32,
        n: usize,
    ) -> Ds4Result<()> {
        self.with_kernel(KernelId::AddScaledInplaceF32, n, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, dst)?;
            runtime.set_arg_mem(kernel, 1, src)?;
            runtime.set_arg_f32(kernel, 2, scale)?;
            runtime.set_arg_i32(kernel, 3, n as i32)
        })
    }

    fn launch_silu_product(&self, gate: &ArcMem, up: &ArcMem, n: usize) -> Ds4Result<()> {
        self.with_kernel(KernelId::SiluProductF32, n, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, gate)?;
            runtime.set_arg_mem(kernel, 1, up)?;
            runtime.set_arg_i32(kernel, 2, n as i32)
        })
    }

    fn launch_rope(&self, x: &ArcMem, pos: usize, dims: ModelDims) -> Ds4Result<()> {
        let total_pairs = dims.n_heads * (dims.head_dim / 2);
        self.with_kernel(KernelId::RopeF32, total_pairs, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, x)?;
            runtime.set_arg_i32(kernel, 1, pos as i32)?;
            runtime.set_arg_i32(kernel, 2, dims.n_heads as i32)?;
            runtime.set_arg_i32(kernel, 3, dims.head_dim as i32)?;
            runtime.set_arg_f32(kernel, 4, dims.rope_freq_base)
        })
    }

    fn launch_store_cache(
        &self,
        cache: &ArcMem,
        src: &ArcMem,
        token_idx: usize,
        dims: ModelDims,
    ) -> Ds4Result<()> {
        let offset = token_idx * dims.hidden;
        self.with_kernel(KernelId::StoreCacheF32, dims.hidden, |runtime, kernel| {
            runtime.set_arg_mem(kernel, 0, cache)?;
            runtime.set_arg_mem(kernel, 1, src)?;
            runtime.set_arg_i32(kernel, 2, offset as i32)?;
            runtime.set_arg_i32(kernel, 3, dims.hidden as i32)
        })
    }

    fn launch_attention(
        &self,
        q: &ArcMem,
        k_cache: &ArcMem,
        v_cache: &ArcMem,
        prefix_len: usize,
        dims: ModelDims,
    ) -> Ds4Result<ArcMem> {
        let out = self.alloc_f32(dims.hidden)?;
        self.with_kernel(
            KernelId::AttentionDecodeF32,
            dims.hidden,
            |runtime, kernel| {
                runtime.set_arg_mem(kernel, 0, q)?;
                runtime.set_arg_mem(kernel, 1, k_cache)?;
                runtime.set_arg_mem(kernel, 2, v_cache)?;
                runtime.set_arg_i32(kernel, 3, prefix_len as i32)?;
                runtime.set_arg_i32(kernel, 4, dims.n_heads as i32)?;
                runtime.set_arg_i32(kernel, 5, dims.head_dim as i32)?;
                runtime.set_arg_mem(kernel, 6, &out)
            },
        )?;
        Ok(out)
    }

    fn forward_layers(
        &self,
        x: &ArcMem,
        pos: usize,
        k_caches: &[ArcMem],
        v_caches: &[ArcMem],
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
        x: &ArcMem,
        rope_pos: usize,
        cache_pos: usize,
        prefix_len: usize,
        layer: usize,
        k_cache: &ArcMem,
        v_cache: &ArcMem,
        dims: ModelDims,
    ) -> Ds4Result<()> {
        let names = self.spec.layer_tensors(layer)?;
        let attn_in = self.launch_rmsnorm(x, &names.attn_norm, dims.hidden)?;
        let q = self.launch_matvec(&attn_in, &names.attn_q)?;
        let k = self.launch_matvec(&attn_in, &names.attn_k)?;
        let v = self.launch_matvec(&attn_in, &names.attn_v)?;
        self.launch_rope(&q, rope_pos, dims)?;
        self.launch_rope(&k, rope_pos, dims)?;
        self.launch_store_cache(k_cache, &k, cache_pos, dims)?;
        self.launch_store_cache(v_cache, &v, cache_pos, dims)?;
        let attn = self.launch_attention(&q, k_cache, v_cache, prefix_len, dims)?;
        let attn_proj = self.launch_matvec(&attn, &names.attn_output)?;
        self.launch_add_inplace(x, &attn_proj, dims.hidden)?;

        let ffn_in = self.launch_rmsnorm(x, &names.ffn_norm, dims.hidden)?;
        match names.ffn {
            FfnTensorNames::Dense { gate, up, down } => {
                let down = self.launch_gated_ffn_by_name(&ffn_in, &gate, &up, &down)?;
                self.launch_add_inplace(x, &down, dims.hidden)?;
            }
            FfnTensorNames::RoutedMoe {
                router,
                expert_layout,
                shared_gate,
                shared_up,
                shared_down,
            } => {
                self.launch_routed_moe_ffn(
                    x,
                    &ffn_in,
                    layer,
                    &router,
                    &expert_layout,
                    shared_gate.as_deref(),
                    shared_up.as_deref(),
                    shared_down.as_deref(),
                    dims,
                )?;
            }
        }
        self.runtime.finish()
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_routed_moe_ffn(
        &self,
        x: &ArcMem,
        ffn_in: &ArcMem,
        layer: usize,
        router: &str,
        expert_layout: &ExpertTensorLayout,
        shared_gate: Option<&str>,
        shared_up: Option<&str>,
        shared_down: Option<&str>,
        dims: ModelDims,
    ) -> Ds4Result<()> {
        let router_tensor = self.tensor_supported(router)?;
        let (_, router_out) = matvec_dims(&router_tensor.descriptor, router)?;
        let logits = self.launch_matvec_tensor(ffn_in, router, router_tensor)?;
        let logits = self.download_f32(&logits, router_out)?;
        let active = self
            .spec
            .moe
            .as_ref()
            .map_or(1, |moe| moe.active_experts)
            .min(router_out)
            .max(1);
        let selected = topk_softmax(&logits, active);
        let acc = self.upload_f32(&vec![0.0f32; dims.hidden])?;
        for (expert, weight) in selected {
            let ExpertTensorSet {
                gate_name,
                gate,
                up_name,
                up,
                down_name,
                down,
            } = self.load_expert_tensor_set(expert_layout, layer, expert)?;
            let expert_out = self.launch_gated_ffn_tensors(
                ffn_in, &gate_name, &gate, &up_name, &up, &down_name, &down,
            )?;
            self.launch_add_scaled_inplace(&acc, &expert_out, weight, dims.hidden)?;
        }
        if let (Some(gate), Some(up), Some(down)) = (shared_gate, shared_up, shared_down) {
            if self.tensors.contains_key(gate)
                && self.tensors.contains_key(up)
                && self.tensors.contains_key(down)
            {
                let shared = self.launch_gated_ffn_by_name(ffn_in, gate, up, down)?;
                self.launch_add_inplace(&acc, &shared, dims.hidden)?;
            }
        }
        self.launch_add_inplace(x, &acc, dims.hidden)
    }

    fn load_expert_tensor_set(
        &self,
        layout: &ExpertTensorLayout,
        layer: usize,
        expert: usize,
    ) -> Ds4Result<ExpertTensorSet> {
        match layout {
            ExpertTensorLayout::PerExpert {
                gate_pattern,
                up_pattern,
                down_pattern,
            } => {
                let gate_name = render_expert_pattern(gate_pattern, layer, expert);
                let up_name = render_expert_pattern(up_pattern, layer, expert);
                let down_name = render_expert_pattern(down_pattern, layer, expert);
                Ok(ExpertTensorSet {
                    gate: self.load_transient_supported_tensor(&gate_name)?,
                    up: self.load_transient_supported_tensor(&up_name)?,
                    down: self.load_transient_supported_tensor(&down_name)?,
                    gate_name,
                    up_name,
                    down_name,
                })
            }
            ExpertTensorLayout::Packed { gate, up, down } => {
                let gate_name = render_layer_pattern(gate, layer);
                let up_name = render_layer_pattern(up, layer);
                let down_name = render_layer_pattern(down, layer);
                Ok(ExpertTensorSet {
                    gate: self.load_packed_expert_tensor(&gate_name, expert)?,
                    up: self.load_packed_expert_tensor(&up_name, expert)?,
                    down: self.load_packed_expert_tensor(&down_name, expert)?,
                    gate_name: format!("{gate_name}#{expert}"),
                    up_name: format!("{up_name}#{expert}"),
                    down_name: format!("{down_name}#{expert}"),
                })
            }
        }
    }

    fn load_packed_expert_tensor(&self, name: &str, expert: usize) -> Ds4Result<ArcTensor> {
        let descriptor = self.descriptor(name)?;
        if dtype_code(descriptor.dtype).is_none() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!(
                    "Arc packed expert tensor {name} has unsupported dtype {:?}",
                    descriptor.dtype
                ),
            ));
        }
        if descriptor.dims.len() != 3 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "packed expert tensor {name} must be 3D, got {:?}",
                    descriptor.dims
                ),
            ));
        }
        let expert_count = self
            .spec
            .moe
            .as_ref()
            .map_or(0, |moe| moe.experts)
            .max(expert + 1);
        let expert_axis = descriptor
            .dims
            .iter()
            .position(|dim| *dim as usize == expert_count)
            .ok_or_else(|| {
                Ds4Error::new(
                    Ds4ErrorKind::Model,
                    format!(
                        "packed expert tensor {name} shape {:?} does not contain expert count {expert_count}",
                        descriptor.dims
                    ),
                )
            })?;
        let matrix_dims = descriptor
            .dims
            .iter()
            .enumerate()
            .filter_map(|(idx, dim)| (idx != expert_axis).then_some(*dim))
            .collect::<Vec<_>>();
        if matrix_dims.len() != 2 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("packed expert tensor {name} does not reduce to a 2D matrix"),
            ));
        }
        let tensor = self.gguf.tensor(name)?;
        let bytes = if descriptor.dtype == GgufDType::F32 {
            packed_f32_expert_slice(tensor.bytes, &descriptor.dims, expert_axis, expert)?
        } else {
            packed_quant_expert_slice(tensor.bytes, &matrix_dims, descriptor.dtype, expert)?
        };
        let memory = ArcRuntime::from_bytes(&self.runtime, &bytes)?;
        Ok(ArcTensor {
            descriptor: TensorDescriptor {
                name: format!("{name}#{expert}"),
                dims: matrix_dims,
                dtype: descriptor.dtype,
                offset: 0,
            },
            memory,
        })
    }

    fn launch_gated_ffn_by_name(
        &self,
        input: &ArcMem,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
    ) -> Ds4Result<ArcMem> {
        let gate = self.tensor_supported(gate_name)?;
        let up = self.tensor_supported(up_name)?;
        let down = self.tensor_supported(down_name)?;
        self.launch_gated_ffn_tensors(input, gate_name, gate, up_name, up, down_name, down)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gated_ffn_tensors(
        &self,
        input: &ArcMem,
        gate_name: &str,
        gate_tensor: &ArcTensor,
        up_name: &str,
        up_tensor: &ArcTensor,
        down_name: &str,
        down_tensor: &ArcTensor,
    ) -> Ds4Result<ArcMem> {
        let (_, ffn) = matvec_dims(&gate_tensor.descriptor, gate_name)?;
        let (_, up_out) = matvec_dims(&up_tensor.descriptor, up_name)?;
        if up_out != ffn {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("FFN gate/up width mismatch: {gate_name}={ffn}, {up_name}={up_out}"),
            ));
        }
        let (down_in, _) = matvec_dims(&down_tensor.descriptor, down_name)?;
        if down_in != ffn {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("FFN down input mismatch: {down_name} expects {down_in}, gate has {ffn}"),
            ));
        }
        let gate = self.launch_matvec_tensor(input, gate_name, gate_tensor)?;
        let up = self.launch_matvec_tensor(input, up_name, up_tensor)?;
        self.launch_silu_product(&gate, &up, ffn)?;
        self.launch_matvec_tensor(&gate, down_name, down_tensor)
    }

    fn compute_output_head_device(
        &self,
        hidden: &ArcMem,
        n_tokens: usize,
        logits: &mut [f32],
    ) -> Ds4Result<()> {
        let dims = self.dims();
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
        for token_idx in 0..n_tokens {
            let hidden_slice = self.slice_hidden_to_owned(hidden, token_idx, dims.hidden)?;
            let head_input = if let Some(norm) = self.spec.output_norm.as_deref() {
                self.launch_rmsnorm(&hidden_slice, norm, dims.hidden)?
            } else {
                hidden_slice
            };
            let out_slice = self.alloc_f32(dims.vocab)?;
            self.launch_matvec_to(
                &head_input,
                &self.spec.output,
                &out_slice,
                dims.hidden,
                dims.vocab,
            )?;
            self.copy_logits_slice(&out_slice, &logits_dev, token_idx, dims.vocab)?;
        }
        self.runtime.finish()?;
        let values = self.download_f32(&logits_dev, logits_needed)?;
        logits[..logits_needed].copy_from_slice(&values);
        Ok(())
    }

    fn slice_hidden_to_owned(
        &self,
        hidden: &ArcMem,
        token_idx: usize,
        hidden_dim: usize,
    ) -> Ds4Result<ArcMem> {
        if token_idx == 0 {
            let bytes = read_prefix_bytes(hidden, hidden_dim * 4)?;
            return ArcRuntime::from_bytes(&self.runtime, &bytes);
        }
        let mut all = vec![0u8; (token_idx + 1) * hidden_dim * 4];
        hidden.read(&mut all)?;
        let start = token_idx * hidden_dim * 4;
        ArcRuntime::from_bytes(&self.runtime, &all[start..start + hidden_dim * 4])
    }

    fn copy_logits_slice(
        &self,
        src: &ArcMem,
        dst: &ArcMem,
        token_idx: usize,
        vocab: usize,
    ) -> Ds4Result<()> {
        let mut bytes = vec![0u8; vocab * 4];
        src.read(&mut bytes)?;
        let mut all = vec![0u8; (token_idx + 1) * vocab * 4];
        if token_idx > 0 {
            dst.read(&mut all[..token_idx * vocab * 4])?;
        }
        let start = token_idx * vocab * 4;
        all[start..start + vocab * 4].copy_from_slice(&bytes);
        dst.write(&all)
    }

    fn compute_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        if tokens.is_empty() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "token sequence is empty",
            ));
        }
        let dims = self.dims();
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
        let mut last_hidden = None;
        for (pos, &token) in tokens.iter().enumerate() {
            let x = self.launch_embedding(token, dims)?;
            self.forward_layers(&x, pos, &k_caches, &v_caches, dims)?;
            last_hidden = Some(x);
        }
        let hidden = last_hidden.ok_or_else(|| {
            Ds4Error::new(Ds4ErrorKind::InvalidArgument, "token sequence is empty")
        })?;
        self.compute_output_head_device(&hidden, 1, &mut out[..dims.vocab])
    }

    fn with_kernel(
        &self,
        id: KernelId,
        global: usize,
        set_args: impl FnOnce(&ArcRuntime, crate::runtime::ClKernel) -> Ds4Result<()>,
    ) -> Ds4Result<()> {
        let _guard = self.launch_lock.lock().map_err(|_| {
            Ds4Error::new(Ds4ErrorKind::Backend, "Arc kernel launch lock was poisoned")
        })?;
        let kernel = self.runtime.kernel(id);
        set_args(&self.runtime, kernel)?;
        self.runtime.launch_1d(kernel, global)
    }
}

impl BackendModel for ArcModel {
    fn quant_kind(&self) -> Ds4QuantKind {
        self.metadata.routed_quant.unwrap_or_else(|| {
            if self.spec.block_kind == TransformerBlockKind::Dense {
                Ds4QuantKind::F32
            } else {
                Ds4QuantKind::Q3_K
            }
        })
    }

    fn eval_output_head_from_hc(
        &self,
        hidden_hc: &[f32],
        n_tokens: usize,
        logits: &mut [f32],
    ) -> Ds4Result<()> {
        let dims = self.dims();
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
                format!("hidden_hc too small: {} < {hidden_needed}", hidden_hc.len()),
            ));
        }
        let hidden = self.upload_f32(&hidden_hc[..hidden_needed])?;
        self.compute_output_head_device(&hidden, n_tokens, logits)
    }

    fn eval_sequence_logits(&self, tokens: &[u32], out: &mut [f32]) -> Ds4Result<()> {
        self.compute_sequence_logits(tokens, out)
    }

    fn eval_token_logits(&self, token: u32, out: &mut [f32]) -> Ds4Result<()> {
        self.compute_sequence_logits(&[token], out)
    }
}

fn matvec_dims(descriptor: &TensorDescriptor, tensor_name: &str) -> Ds4Result<(usize, usize)> {
    if descriptor.dims.len() != 2 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("invalid {tensor_name} shape {:?}", descriptor.dims),
        ));
    }
    Ok((descriptor.dims[0] as usize, descriptor.dims[1] as usize))
}

fn dtype_code(dtype: GgufDType) -> Option<i32> {
    match dtype {
        GgufDType::F32 => Some(0),
        GgufDType::F16 => Some(1),
        GgufDType::Q8_0 => Some(2),
        GgufDType::Q4_K => Some(3),
        GgufDType::Q3_K => Some(4),
        GgufDType::Q2_K => Some(5),
        _ => None,
    }
}

fn tensor_dtype_code(tensor: &ArcTensor, name: &str) -> Ds4Result<i32> {
    dtype_code(tensor.descriptor.dtype).ok_or_else(|| {
        Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!(
                "tensor {name} dtype {:?} is not supported by the Arc runtime path",
                tensor.descriptor.dtype
            ),
        )
    })
}

fn is_routed_expert_tensor(name: &str) -> bool {
    if name.contains("_exps.weight") {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let parts = rest.split('.').collect::<Vec<_>>();
    parts.len() == 5
        && matches!(parts[1], "ffn_gate" | "ffn_up" | "ffn_down")
        && parts[2].parse::<usize>().is_ok()
        && parts[3] == "weight"
}

fn render_expert_pattern(pattern: &str, layer: usize, expert: usize) -> String {
    pattern
        .replace("{layer}", &layer.to_string())
        .replace("{expert}", &expert.to_string())
}

fn render_layer_pattern(pattern: &str, layer: usize) -> String {
    pattern.replace("{layer}", &layer.to_string())
}

fn tensor_data_bytes(dims: &[u32], dtype: GgufDType) -> usize {
    let numel = dims.iter().map(|dim| *dim as u64).product::<u64>();
    let blocks = numel.div_ceil(dtype.block_size());
    blocks.saturating_mul(dtype.byte_size()) as usize
}

fn packed_quant_expert_slice(
    bytes: &[u8],
    matrix_dims: &[u32],
    dtype: GgufDType,
    expert: usize,
) -> Ds4Result<Vec<u8>> {
    let stride = tensor_data_bytes(matrix_dims, dtype);
    let start = expert.checked_mul(stride).ok_or_else(|| {
        Ds4Error::new(
            Ds4ErrorKind::Model,
            "packed expert byte offset overflowed usize",
        )
    })?;
    let end = start.checked_add(stride).ok_or_else(|| {
        Ds4Error::new(
            Ds4ErrorKind::Model,
            "packed expert byte range overflowed usize",
        )
    })?;
    if end > bytes.len() {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!(
                "packed expert byte range {start}..{end} exceeds tensor bytes {}",
                bytes.len()
            ),
        ));
    }
    Ok(bytes[start..end].to_vec())
}

fn packed_f32_expert_slice(
    bytes: &[u8],
    dims: &[u32],
    expert_axis: usize,
    expert: usize,
) -> Ds4Result<Vec<u8>> {
    let matrix_dims = dims
        .iter()
        .enumerate()
        .filter_map(|(idx, dim)| (idx != expert_axis).then_some(*dim as usize))
        .collect::<Vec<_>>();
    if matrix_dims.len() != 2 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            "packed F32 expert tensor did not reduce to two matrix dims",
        ));
    }
    let mut out = Vec::with_capacity(matrix_dims[0] * matrix_dims[1] * 4);
    for row in 0..matrix_dims[0] {
        for col in 0..matrix_dims[1] {
            let mut coords = [0usize; 3];
            let mut matrix_idx = 0usize;
            for (axis, coord) in coords.iter_mut().enumerate() {
                if axis == expert_axis {
                    *coord = expert;
                } else {
                    *coord = if matrix_idx == 0 { row } else { col };
                    matrix_idx += 1;
                }
            }
            let elem = row_major_index(&coords, dims)?;
            let start = elem.checked_mul(4).ok_or_else(|| {
                Ds4Error::new(
                    Ds4ErrorKind::Model,
                    "packed F32 byte offset overflowed usize",
                )
            })?;
            let end = start + 4;
            if end > bytes.len() {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::Model,
                    format!(
                        "packed F32 byte range {start}..{end} exceeds tensor bytes {}",
                        bytes.len()
                    ),
                ));
            }
            out.extend_from_slice(&bytes[start..end]);
        }
    }
    Ok(out)
}

fn row_major_index(coords: &[usize; 3], dims: &[u32]) -> Ds4Result<usize> {
    let mut index = 0usize;
    for (coord, dim) in coords.iter().zip(dims.iter()) {
        let dim = *dim as usize;
        if *coord >= dim {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("packed expert coordinate {coord} is outside dim {dim}"),
            ));
        }
        index = index
            .checked_mul(dim)
            .and_then(|value| value.checked_add(*coord))
            .ok_or_else(|| {
                Ds4Error::new(
                    Ds4ErrorKind::Model,
                    "packed expert element offset overflowed usize",
                )
            })?;
    }
    Ok(index)
}

fn topk_softmax(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k.min(ranked.len()));
    if ranked.is_empty() {
        return ranked;
    }
    let max = ranked
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for (_, value) in &mut ranked {
        *value = (*value - max).exp();
        denom += *value;
    }
    if denom > 0.0 {
        for (_, value) in &mut ranked {
            *value /= denom;
        }
    }
    ranked
}

fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_prefix_bytes(mem: &ArcMem, len: usize) -> Ds4Result<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    mem.read(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_arc() {
        assert_eq!(ArcBackend::new().name(), "arc");
    }

    #[test]
    fn packed_f32_expert_slice_extracts_selected_expert_axis() {
        let dims = vec![2, 3, 2];
        let mut bytes = Vec::new();
        for row in 0..2 {
            for col in 0..3 {
                for expert in 0..2 {
                    let value = (row * 100 + col * 10 + expert) as f32;
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let slice = packed_f32_expert_slice(&bytes, &dims, 2, 1).unwrap();
        let values = bytes_to_f32_vec(&slice);
        assert_eq!(values, vec![1.0, 11.0, 21.0, 101.0, 111.0, 121.0]);
    }

    #[test]
    fn packed_q3_expert_slice_uses_matrix_byte_stride() {
        let mut bytes = vec![1u8; 110];
        bytes.extend(std::iter::repeat_n(2u8, 110));
        let slice = packed_quant_expert_slice(&bytes, &[8, 16], GgufDType::Q3_K, 1).unwrap();
        assert_eq!(slice.len(), 110);
        assert!(slice.iter().all(|byte| *byte == 2));
    }

    #[test]
    fn arc_backend_loads_synthetic_or_reports_absent_arc() {
        let dir = std::env::temp_dir().join(format!("ds4-backend-arc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        match ArcBackend::new().load_model(&path) {
            Ok(model) => {
                let mut logits = vec![0.0f32; 16];
                model.eval_token_logits(1, &mut logits).unwrap();
                assert!(logits
                    .iter()
                    .any(|value| value.is_finite() && *value != 0.0));
            }
            Err(err) => assert_eq!(err.kind, Ds4ErrorKind::NotImplemented, "{err}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn arc_backend_runs_synthetic_qwen_moe_or_reports_absent_arc() {
        let dir = std::env::temp_dir().join(format!("ds4-backend-arc-moe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qwen-moe-synth.gguf");
        ds4_gguf::write_synthetic_qwen_moe_gguf(&path).unwrap();
        match ArcBackend::new().load_model(&path) {
            Ok(model) => {
                let mut logits = vec![0.0f32; 16];
                model.eval_token_logits(1, &mut logits).unwrap();
                assert!(logits
                    .iter()
                    .any(|value| value.is_finite() && *value != 0.0));
            }
            Err(err) => assert_eq!(err.kind, Ds4ErrorKind::NotImplemented, "{err}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
