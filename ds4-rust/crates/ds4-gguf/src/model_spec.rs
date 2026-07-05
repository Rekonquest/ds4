use std::collections::{HashMap, HashSet};

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

use crate::{GgufFile, GgufMetadata, KvRaw, TensorDescriptor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchitectureKind {
    Ds4,
    Qwen3,
    Qwen3Moe,
    Qwen35,
    Qwen35Moe,
    Unknown(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerBlockKind {
    Dense,
    RoutedMoe,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelDims {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub context: usize,
    pub rope_freq_base: f32,
    pub rms_norm_epsilon: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeSpec {
    pub experts: usize,
    pub active_experts: usize,
    pub shared_experts: usize,
    pub router: String,
    pub expert_layout: ExpertTensorLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertTensorLayout {
    PerExpert {
        gate_pattern: String,
        up_pattern: String,
        down_pattern: String,
    },
    Packed {
        gate: String,
        up: String,
        down: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTensorNames {
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub ffn_norm: String,
    pub ffn: FfnTensorNames,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfnTensorNames {
    Dense {
        gate: String,
        up: String,
        down: String,
    },
    RoutedMoe {
        router: String,
        expert_layout: ExpertTensorLayout,
        shared_gate: Option<String>,
        shared_up: Option<String>,
        shared_down: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub architecture: ArchitectureKind,
    pub block_kind: TransformerBlockKind,
    pub dims: ModelDims,
    pub output_norm: Option<String>,
    pub output: String,
    pub token_embedding: String,
    pub moe: Option<MoeSpec>,
}

impl ModelSpec {
    pub fn from_gguf(file: &GgufFile) -> Ds4Result<Self> {
        Self::from_metadata(&file.metadata, &file.tensors, |key| file.kv_raw(key))
    }

    pub fn from_metadata<'a>(
        metadata: &GgufMetadata,
        tensors: &[TensorDescriptor],
        kv_raw: impl Fn(&str) -> Option<&'a KvRaw>,
    ) -> Ds4Result<Self> {
        let tensor_names = tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<HashSet<_>>();
        let tensor_index = tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<HashMap<_, _>>();
        let token_desc = tensor_index
            .get("token_embd.weight")
            .copied()
            .ok_or_else(|| Ds4Error::new(Ds4ErrorKind::Model, "missing token_embd.weight"))?;
        if token_desc.dims.len() != 2 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("invalid token_embd.weight shape {:?}", token_desc.dims),
            ));
        }

        let arch_key = metadata.architecture.to_ascii_lowercase();
        let has_router = has_any_layer_tensor(&tensor_names, "ffn_gate_inp.weight");
        let has_qwen35_hybrid = has_qwen35_hybrid_layout(&tensor_names);
        let qwen35_family = is_qwen35_arch(&arch_key) || has_qwen35_hybrid;
        let has_experts = metadata.expert_count.unwrap_or(0) > 0 || has_router;
        let architecture = if arch_key.starts_with("ds4") {
            ArchitectureKind::Ds4
        } else if qwen35_family && has_experts {
            ArchitectureKind::Qwen35Moe
        } else if qwen35_family {
            ArchitectureKind::Qwen35
        } else if arch_key.contains("qwen") && has_experts {
            ArchitectureKind::Qwen3Moe
        } else if arch_key.contains("qwen") {
            ArchitectureKind::Qwen3
        } else if has_experts {
            ArchitectureKind::Qwen3Moe
        } else {
            ArchitectureKind::Unknown(metadata.architecture.clone())
        };

        let layers = metadata
            .layer_count
            .or_else(|| kv_u32(&kv_raw, &arch_key, "block_count"))
            .map_or_else(|| infer_layer_count(&tensor_names), |v| v as usize);
        let n_heads = metadata
            .head_count
            .or_else(|| kv_u32(&kv_raw, &arch_key, "attention.head_count"))
            .unwrap_or(1) as usize;
        let n_kv_heads = metadata
            .kv_head_count
            .or_else(|| kv_u32(&kv_raw, &arch_key, "attention.head_count_kv"))
            .map_or(n_heads, |v| v as usize);
        if n_heads == 0 || n_kv_heads == 0 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                "attention head counts must be nonzero",
            ));
        }
        let (vocab, hidden) = infer_token_embedding_dims(metadata, token_desc)?;
        let head_dim = metadata
            .head_dim
            .or_else(|| kv_u32(&kv_raw, &arch_key, "attention.key_length"))
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

        let block_kind = if has_experts {
            TransformerBlockKind::RoutedMoe
        } else {
            TransformerBlockKind::Dense
        };
        let ffn = infer_ffn(metadata, &tensor_index, block_kind, hidden);
        let context = metadata
            .context_length
            .or_else(|| kv_u32(&kv_raw, &arch_key, "context_length"))
            .unwrap_or(0) as usize;
        let rope_freq_base = metadata
            .rope_freq_base
            .or_else(|| kv_f32(&kv_raw, &arch_key, "rope.freq_base"))
            .unwrap_or(10_000.0);
        let rms_norm_epsilon = metadata
            .rms_norm_epsilon
            .or_else(|| kv_f32(&kv_raw, &arch_key, "attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let dims = ModelDims {
            vocab,
            hidden,
            layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn,
            context,
            rope_freq_base,
            rms_norm_epsilon,
        };
        if has_qwen35_hybrid {
            return Err(qwen35_hybrid_layout_error(&tensor_names));
        }
        let moe = if block_kind == TransformerBlockKind::RoutedMoe {
            Some(infer_moe_spec(metadata, &tensor_names)?)
        } else {
            None
        };

        Ok(Self {
            architecture,
            block_kind,
            dims,
            output_norm: tensor_exists(&tensor_names, "output_norm.weight")
                .then(|| "output_norm.weight".to_string()),
            output: pick_existing(
                &tensor_names,
                &["output.weight", "token_embd.weight"],
                "output.weight",
            ),
            token_embedding: "token_embd.weight".to_string(),
            moe,
        })
    }

    pub fn layer_tensors(&self, layer: usize) -> Ds4Result<LayerTensorNames> {
        if layer >= self.dims.layers {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "layer {layer} is outside model layer count {}",
                    self.dims.layers
                ),
            ));
        }
        let prefix = format!("blk.{layer}");
        let attn_output = if self.is_qwen_like() {
            format!("{prefix}.attn_output.weight")
        } else {
            format!("{prefix}.attn_out.weight")
        };
        let ffn = match &self.moe {
            Some(moe) => FfnTensorNames::RoutedMoe {
                router: format!("{prefix}.ffn_gate_inp.weight"),
                expert_layout: moe.expert_layout.clone(),
                shared_gate: Some(format!("{prefix}.ffn_gate_shexp.weight")),
                shared_up: Some(format!("{prefix}.ffn_up_shexp.weight")),
                shared_down: Some(format!("{prefix}.ffn_down_shexp.weight")),
            },
            None => FfnTensorNames::Dense {
                gate: format!("{prefix}.ffn_gate.weight"),
                up: format!("{prefix}.ffn_up.weight"),
                down: format!("{prefix}.ffn_down.weight"),
            },
        };
        Ok(LayerTensorNames {
            attn_norm: format!("{prefix}.attn_norm.weight"),
            attn_q: format!("{prefix}.attn_q.weight"),
            attn_k: format!("{prefix}.attn_k.weight"),
            attn_v: format!("{prefix}.attn_v.weight"),
            attn_output,
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            ffn,
        })
    }

    fn is_qwen_like(&self) -> bool {
        matches!(
            self.architecture,
            ArchitectureKind::Qwen3
                | ArchitectureKind::Qwen3Moe
                | ArchitectureKind::Qwen35
                | ArchitectureKind::Qwen35Moe
        )
    }
}

fn is_qwen35_arch(arch_key: &str) -> bool {
    arch_key.contains("qwen35")
        || arch_key.contains("qwen3.5")
        || arch_key.contains("qwen3_5")
        || arch_key.contains("qwen3-5")
        || arch_key.contains("qwen3next")
        || arch_key.contains("qwen3_next")
}

fn has_qwen35_hybrid_layout(tensor_names: &HashSet<&str>) -> bool {
    tensor_exists(tensor_names, "blk.0.attn_qkv.weight")
        || has_any_layer_tensor(tensor_names, "attn_qkv.weight")
        || has_any_layer_tensor(tensor_names, "ssm_a")
        || has_any_layer_tensor(tensor_names, "ssm_conv1d.weight")
        || has_any_layer_tensor(tensor_names, "ssm_out.weight")
}

fn qwen35_hybrid_layout_error(tensor_names: &HashSet<&str>) -> Ds4Error {
    let mut found = Vec::new();
    if has_any_layer_tensor(tensor_names, "attn_qkv.weight") {
        found.push("attn_qkv.weight");
    }
    if has_any_layer_tensor(tensor_names, "ssm_a")
        || has_any_layer_tensor(tensor_names, "ssm_conv1d.weight")
        || has_any_layer_tensor(tensor_names, "ssm_out.weight")
    {
        found.push("SSM tensors");
    }
    let found = if found.is_empty() {
        "Qwen3.5 metadata".to_string()
    } else {
        found.join(" and ")
    };
    Ds4Error::new(
        Ds4ErrorKind::Model,
        format!(
            "Qwen3.5 hybrid QKV/SSM layout requires a Qwen3.5 execution path; found {found}, while this DS4 loader expects split attention tensors (attn_q/attn_k/attn_v) and Transformer FFN blocks"
        ),
    )
}

fn infer_moe_spec(metadata: &GgufMetadata, tensor_names: &HashSet<&str>) -> Ds4Result<MoeSpec> {
    let experts = metadata.expert_count.unwrap_or(0) as usize;
    let active_experts = metadata.expert_used_count.unwrap_or(1) as usize;
    if active_experts == 0 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            "MoE active expert count must be nonzero",
        ));
    }
    if experts != 0 && active_experts > experts {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("active expert count {active_experts} exceeds expert count {experts}"),
        ));
    }

    let expert_layout = if tensor_exists(tensor_names, "blk.0.ffn_gate.0.weight") {
        ExpertTensorLayout::PerExpert {
            gate_pattern: "blk.{layer}.ffn_gate.{expert}.weight".to_string(),
            up_pattern: "blk.{layer}.ffn_up.{expert}.weight".to_string(),
            down_pattern: "blk.{layer}.ffn_down.{expert}.weight".to_string(),
        }
    } else {
        ExpertTensorLayout::Packed {
            gate: "blk.{layer}.ffn_gate_exps.weight".to_string(),
            up: "blk.{layer}.ffn_up_exps.weight".to_string(),
            down: "blk.{layer}.ffn_down_exps.weight".to_string(),
        }
    };

    Ok(MoeSpec {
        experts,
        active_experts,
        shared_experts: metadata.shared_expert_count.unwrap_or(0) as usize,
        router: "blk.{layer}.ffn_gate_inp.weight".to_string(),
        expert_layout,
    })
}

fn infer_layer_count(tensor_names: &HashSet<&str>) -> usize {
    tensor_names
        .iter()
        .filter_map(|name| {
            let rest = name.strip_prefix("blk.")?;
            let (idx, _) = rest.split_once('.')?;
            idx.parse::<usize>().ok()
        })
        .max()
        .map_or(0, |idx| idx + 1)
}

fn infer_ffn(
    metadata: &GgufMetadata,
    tensor_index: &HashMap<&str, &TensorDescriptor>,
    block_kind: TransformerBlockKind,
    hidden: usize,
) -> usize {
    if let Some(ffn) = metadata.feed_forward_length {
        return ffn as usize;
    }
    match block_kind {
        TransformerBlockKind::Dense => {
            tensor_dim(tensor_index, "blk.0.ffn_gate.weight", 1).unwrap_or(hidden * 4)
        }
        TransformerBlockKind::RoutedMoe => tensor_dim(tensor_index, "blk.0.ffn_gate.0.weight", 1)
            .or_else(|| tensor_dim(tensor_index, "blk.0.ffn_gate_exps.weight", 1))
            .unwrap_or(hidden * 4),
    }
}

fn infer_token_embedding_dims(
    metadata: &GgufMetadata,
    token_desc: &TensorDescriptor,
) -> Ds4Result<(usize, usize)> {
    if token_desc.dims.len() != 2 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("invalid token_embd.weight shape {:?}", token_desc.dims),
        ));
    }

    let a = token_desc.dims[0] as usize;
    let b = token_desc.dims[1] as usize;
    if a == 0 || b == 0 {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            "token embedding dimensions must be nonzero",
        ));
    }

    if let Some(hidden) = metadata.embedding_dim.map(|v| v as usize) {
        return match (a == hidden, b == hidden) {
            (true, false) => Ok((b, a)),
            (false, true) => Ok((a, b)),
            (true, true) => Ok((a, b)),
            (false, false) => Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "token_embd.weight shape {:?} does not contain embedding_length {hidden}",
                    token_desc.dims
                ),
            )),
        };
    }

    if let Some(vocab) = metadata.vocab_size.map(|v| v as usize) {
        return match (a == vocab, b == vocab) {
            (true, false) => Ok((a, b)),
            (false, true) => Ok((b, a)),
            (true, true) => Ok((a, b)),
            (false, false) => Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!(
                    "token_embd.weight shape {:?} does not contain vocab_size {vocab}",
                    token_desc.dims
                ),
            )),
        };
    }

    if let (Some(heads), Some(head_dim)) = (metadata.head_count, metadata.head_dim) {
        let hidden = heads as usize * head_dim as usize;
        return match (a == hidden, b == hidden) {
            (true, false) => Ok((b, a)),
            (false, true) => Ok((a, b)),
            _ => Ok((a, b)),
        };
    }

    Ok((a, b))
}

fn tensor_dim(
    tensor_index: &HashMap<&str, &TensorDescriptor>,
    name: &str,
    idx: usize,
) -> Option<usize> {
    tensor_index
        .get(name)
        .and_then(|tensor| tensor.dims.get(idx))
        .map(|dim| *dim as usize)
}

fn has_any_layer_tensor(tensor_names: &HashSet<&str>, suffix: &str) -> bool {
    tensor_names.iter().any(|name| name.ends_with(suffix))
}

fn tensor_exists(tensor_names: &HashSet<&str>, name: &str) -> bool {
    tensor_names.contains(name)
}

fn pick_existing(tensor_names: &HashSet<&str>, choices: &[&str], fallback: &str) -> String {
    choices
        .iter()
        .copied()
        .find(|name| tensor_exists(tensor_names, name))
        .unwrap_or(fallback)
        .to_string()
}

fn kv_u32<'a>(
    kv_raw: &impl Fn(&str) -> Option<&'a KvRaw>,
    arch_key: &str,
    suffix: &str,
) -> Option<u32> {
    let key = format!("{arch_key}.{suffix}");
    kv_raw(&key).and_then(KvRaw::as_u32)
}

fn kv_f32<'a>(
    kv_raw: &impl Fn(&str) -> Option<&'a KvRaw>,
    arch_key: &str,
    suffix: &str,
) -> Option<f32> {
    let key = format!("{arch_key}.{suffix}");
    kv_raw(&key).and_then(|value| value.as_f64().map(|v| v as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufDType;

    fn tensor(name: &str, dims: &[u32]) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            dims: dims.to_vec(),
            dtype: GgufDType::F32,
            offset: 0,
        }
    }

    #[test]
    fn synthetic_ds4_spec_uses_dense_names() {
        let metadata = GgufMetadata {
            architecture: "ds4-synth".to_string(),
            layer_count: Some(1),
            head_count: Some(1),
            feed_forward_length: Some(16),
            ..GgufMetadata::default()
        };
        let tensors = vec![
            tensor("token_embd.weight", &[16, 8]),
            tensor("output.weight", &[8, 16]),
            tensor("output_norm.weight", &[8]),
            tensor("blk.0.attn_out.weight", &[8, 8]),
            tensor("blk.0.ffn_gate.weight", &[8, 16]),
        ];
        let spec = ModelSpec::from_metadata(&metadata, &tensors, |_| None).unwrap();
        assert_eq!(spec.architecture, ArchitectureKind::Ds4);
        assert_eq!(spec.block_kind, TransformerBlockKind::Dense);
        assert_eq!(spec.dims.ffn, 16);
        let layer = spec.layer_tensors(0).unwrap();
        assert_eq!(layer.attn_output, "blk.0.attn_out.weight");
        assert!(matches!(layer.ffn, FfnTensorNames::Dense { .. }));
    }

    #[test]
    fn qwen_moe_spec_uses_router_and_expert_names() {
        let metadata = GgufMetadata {
            architecture: "qwen3moe".to_string(),
            layer_count: Some(2),
            head_count: Some(64),
            kv_head_count: Some(8),
            head_dim: Some(128),
            feed_forward_length: Some(4096),
            expert_count: Some(256),
            expert_used_count: Some(8),
            shared_expert_count: Some(1),
            rope_freq_base: Some(1_000_000.0),
            rms_norm_epsilon: Some(1e-6),
            ..GgufMetadata::default()
        };
        let tensors = vec![
            tensor("token_embd.weight", &[152_064, 8192]),
            tensor("output.weight", &[8192, 152_064]),
            tensor("output_norm.weight", &[8192]),
            tensor("blk.0.attn_output.weight", &[8192, 8192]),
            tensor("blk.0.ffn_gate_inp.weight", &[8192, 256]),
            tensor("blk.0.ffn_gate.0.weight", &[8192, 4096]),
            tensor("blk.0.ffn_up.0.weight", &[8192, 4096]),
            tensor("blk.0.ffn_down.0.weight", &[4096, 8192]),
        ];
        let spec = ModelSpec::from_metadata(&metadata, &tensors, |_| None).unwrap();
        assert_eq!(spec.architecture, ArchitectureKind::Qwen3Moe);
        assert_eq!(spec.block_kind, TransformerBlockKind::RoutedMoe);
        assert_eq!(spec.dims.n_kv_heads, 8);
        assert_eq!(spec.moe.as_ref().unwrap().active_experts, 8);
        let layer = spec.layer_tensors(0).unwrap();
        assert_eq!(layer.attn_output, "blk.0.attn_output.weight");
        match layer.ffn {
            FfnTensorNames::RoutedMoe { router, .. } => {
                assert_eq!(router, "blk.0.ffn_gate_inp.weight");
            }
            FfnTensorNames::Dense { .. } => panic!("expected routed MoE FFN"),
        }
    }

    #[test]
    fn qwen_spec_accepts_hidden_first_token_embedding_shape() {
        let metadata = GgufMetadata {
            architecture: "qwen3".to_string(),
            layer_count: Some(1),
            embedding_dim: Some(2048),
            head_count: Some(8),
            kv_head_count: Some(2),
            head_dim: Some(256),
            feed_forward_length: Some(6144),
            ..GgufMetadata::default()
        };
        let tensors = vec![
            tensor("token_embd.weight", &[2048, 248_320]),
            tensor("output.weight", &[2048, 248_320]),
            tensor("output_norm.weight", &[2048]),
            tensor("blk.0.attn_output.weight", &[2048, 2048]),
            tensor("blk.0.ffn_gate.weight", &[2048, 6144]),
        ];
        let spec = ModelSpec::from_metadata(&metadata, &tensors, |_| None).unwrap();
        assert_eq!(spec.architecture, ArchitectureKind::Qwen3);
        assert_eq!(spec.dims.hidden, 2048);
        assert_eq!(spec.dims.vocab, 248_320);
        assert_eq!(spec.dims.head_dim, 256);
    }

    #[test]
    fn qwen35_hybrid_layout_reports_architecture_requirement() {
        let metadata = GgufMetadata {
            architecture: "qwen35".to_string(),
            layer_count: Some(1),
            embedding_dim: Some(2048),
            head_count: Some(8),
            kv_head_count: Some(2),
            head_dim: Some(256),
            feed_forward_length: Some(6144),
            ..GgufMetadata::default()
        };
        let tensors = vec![
            tensor("token_embd.weight", &[2048, 248_320]),
            tensor("output_norm.weight", &[2048]),
            tensor("blk.0.attn_gate.weight", &[2048, 2048]),
            tensor("blk.0.attn_norm.weight", &[2048]),
            tensor("blk.0.attn_qkv.weight", &[2048, 6144]),
            tensor("blk.0.ffn_down.weight", &[6144, 2048]),
            tensor("blk.0.ffn_gate.weight", &[2048, 6144]),
            tensor("blk.0.ffn_up.weight", &[2048, 6144]),
            tensor("blk.0.post_attention_norm.weight", &[2048]),
            tensor("blk.0.ssm_a", &[16]),
            tensor("blk.0.ssm_conv1d.weight", &[4, 6144]),
            tensor("blk.0.ssm_out.weight", &[2048, 2048]),
        ];
        let err = ModelSpec::from_metadata(&metadata, &tensors, |_| None).unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::Model);
        assert!(err.message.contains("Qwen3.5 hybrid QKV/SSM"));
        assert!(err.message.contains("attn_qkv.weight"));
        assert!(err.message.contains("SSM tensors"));
    }
}
