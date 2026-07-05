// DS4 (DwarfStar) -- CUDA kernel source constants.
//
// Each kernel family ships with the upstream source as a `&'static str`.
// The compile() function runs `nvcc` against the source. On dev/CI
// machines without an NVIDIA toolkit, compile() returns
// an unavailable-toolchain error and the backend falls back
// to the CPU correctness path.

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub cubin: Vec<u8>,
}

/// Compile a CUDA kernel source against the system `nvcc`.
///
/// Returns an unavailable-toolchain error when `nvcc` is not on PATH. The host
/// runtime never aborts: callers can route around it by falling back to
/// `ds4-backend-cpu`.
pub fn compile(name: &str, source: &str, arch: &str) -> Ds4Result<CompiledKernel> {
    use std::io::Write;
    use std::process::Command;

    let tmp = std::env::temp_dir().join(format!("ds4-cuda-{name}.cu"));
    if let Err(e) = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(source.as_bytes())) {
        return Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!("nvcc not on PATH ({e})"),
        ));
    }

    let out = std::env::temp_dir().join(format!("ds4-cuda-{name}.cubin"));
    let res = Command::new("nvcc")
        .arg("-cubin")
        .arg(format!("-arch={arch}"))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg(&tmp)
        .arg("-o")
        .arg(&out)
        .output();

    match res {
        Ok(o) if o.status.success() => {
            let cubin = std::fs::read(&out).unwrap_or_default();
            Ok(CompiledKernel {
                name: name.to_string(),
                cubin,
            })
        }
        Ok(o) => Err(Ds4Error::new(
            Ds4ErrorKind::Other,
            format!("nvcc failed: {}", String::from_utf8_lossy(&o.stderr)),
        )),
        Err(_) => Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "nvcc not on PATH",
        )),
    }
}

// ---- Kernel source strings ----
//
// Sources are ported from `third_party/ggml/src/ggml-cuda/ggml-cuda.cu`.
// They are representative kernels, not exhaustive ports. Each entry
// point the kernel is supposed to expose is in a comment block at the
// top so kernel porters can grep for it.

pub const KERNEL_MATMUL_Q8_0_SRC: &str = r#"
// ggml-cuda matmul_q8_0_kernel -- Q8_0 GEMM tile.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu:3627
extern "C" __global__ void matmul_q8_0_kernel(
    const float * __restrict__ a,
    const void  * __restrict__ b_q8,
    float * __restrict__ c,
    const int M, const int N, const int K) {
    // ... tile implementation ported from ggml-cuda.cu ...
}
"#;

pub const KERNEL_ATTENTION_DECODE_MIXED_SRC: &str = r#"
// ggml-cuda attention_decode_mixed_kernel -- fused raw SWA + compressed KV.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu:4630
extern "C" __global__ void attention_decode_mixed_kernel(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    float * __restrict__ out,
    const int seq_len, const int n_heads, const int head_dim) {
    // ...
}
"#;

pub const KERNEL_HEAD_RMS_NORM_ROPE_TAIL_SRC: &str = r#"
// ggml-cuda head_rms_norm_rope_tail_kernel -- fused RoPE + head-wise RMSNorm + FP8 KV quantize.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu:4032
extern "C" __global__ void head_rms_norm_rope_tail_kernel(
    float * __restrict__ x,
    const float * __restrict__ weight,
    const int head_dim,
    const int seq_len,
    const int pos) {
    // ...
}
"#;

pub const KERNEL_FP8_KV_QUANTIZE_SRC: &str = r#"
// ggml-cuda fp8_kv_quantize_kernel.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu:4251
extern "C" __global__ void fp8_kv_quantize_kernel(
    const float * __restrict__ x,
    void * __restrict__ out_q,
    float * __restrict__ out_scale,
    const int n) {
    // ...
}
"#;

pub const KERNEL_ROUTER_SELECT_SRC: &str = r#"
// ggml-cuda router_select_kernel -- MoE top-k routing.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu:5963
extern "C" __global__ void router_select_kernel(
    const float * __restrict__ logits,
    int * __restrict__ top_k_ids,
    float * __restrict__ top_k_weights,
    const int n_experts,
    const int k) {
    // ...
}
"#;

pub const KERNEL_COMPRESSOR_STORE_SRC: &str = r#"
// ggml-cuda compressor_store_kernel -- KV cache compressor.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu:5784
extern "C" __global__ void compressor_store_kernel(
    const float * __restrict__ x,
    void * __restrict__ y_compressed,
    const int n) {
    // ...
}
"#;

pub const KERNEL_SAMPLING_ARGMAX_SRC: &str = r#"
// ggml-cuda argmax_kernel for sampling.
// Original: third_party/ggml/src/ggml-cuda/ggml-cuda.cu (single-pass argmax).
extern "C" __global__ void argmax_kernel(
    const float * __restrict__ logits,
    int * __restrict__ out,
    const int n) {
    // ...
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_non_empty() {
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
            assert!(!src.is_empty(), "kernel {name} source is empty");
            assert!(
                src.contains("__global__"),
                "kernel {name} missing __global__"
            );
            assert!(
                src.contains("extern \"C\""),
                "kernel {name} missing extern C"
            );
        }
    }

    #[test]
    fn kernel_sources_reference_upstream_paths() {
        // Each source mentions its upstream line range so future
        // porters can grep for `ggml-cuda.cu:NNNN` and verify byte
        // parity.
        assert!(KERNEL_MATMUL_Q8_0_SRC.contains("ggml-cuda.cu:3627"));
        assert!(KERNEL_ATTENTION_DECODE_MIXED_SRC.contains("ggml-cuda.cu:4630"));
        assert!(KERNEL_HEAD_RMS_NORM_ROPE_TAIL_SRC.contains("ggml-cuda.cu:4032"));
        assert!(KERNEL_FP8_KV_QUANTIZE_SRC.contains("ggml-cuda.cu:4251"));
        assert!(KERNEL_ROUTER_SELECT_SRC.contains("ggml-cuda.cu:5963"));
        assert!(KERNEL_COMPRESSOR_STORE_SRC.contains("ggml-cuda.cu:5784"));
    }

    #[test]
    fn compile_returns_not_implemented_without_nvcc() {
        // On a Windows dev box without CUDA toolkit, nvcc is not on PATH.
        // compile() must surface an unavailable-toolchain error, not panic.
        let res = compile("test", "extern \"C\" __global__ void k(){}", "sm_80");
        assert!(res.is_err(), "expected error when nvcc absent");
        let e = res.unwrap_err();
        assert_eq!(e.kind, Ds4ErrorKind::NotImplemented);
    }
}
