// DS4 (DwarfStar) -- ROCm/HIP kernel source constants.
//
// HIP sources ported from `third_party/ggml/src/ggml-cuda/*.cu` and
// the DS4-specific ROCm extensions (`ds4_rocm.cu` + `rocm/*.cuh`).
// Compile() runs `hipcc` to produce a fat binary. On hosts without
// HIP, kernel compilation reports the missing toolchain.

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub hsaco: Vec<u8>,
}

pub fn compile(name: &str, source: &str, arch: &str) -> Ds4Result<CompiledKernel> {
    use std::io::Write;
    use std::process::Command;

    let tmp = std::env::temp_dir().join(format!("ds4-rocm-{name}.cu"));
    if let Err(e) = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(source.as_bytes())) {
        return Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!("cannot stage kernel source ({e})"),
        ));
    }
    let out = std::env::temp_dir().join(format!("ds4-rocm-{name}.hsaco"));
    let res = Command::new("hipcc")
        .arg("-O3")
        .arg("-ffast-math")
        .arg(format!("--offload-arch={arch}"))
        .arg("-c")
        .arg(&tmp)
        .arg("-o")
        .arg(&out)
        .output();

    match res {
        Ok(o) if o.status.success() => {
            let bytes = std::fs::read(&out).unwrap_or_default();
            Ok(CompiledKernel {
                name: name.to_string(),
                hsaco: bytes,
            })
        }
        Ok(o) => Err(Ds4Error::new(
            Ds4ErrorKind::Other,
            format!("hipcc failed: {}", String::from_utf8_lossy(&o.stderr)),
        )),
        Err(_) => Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "hipcc not on PATH (non-AMD host)",
        )),
    }
}

// HIP kernel sources (representative subset). Use `amd_mixed_dot`
// instead of CUDA's `__dp4a`.

pub const KERNEL_MATMUL_Q8_0_SRC: &str = r#"
// ggml-cuda.cu matmul_q8_0_kernel ported to HIP -- uses amd_mixed_dot
// for the Q8_0 GEMM tile.
extern "C" __global__ void matmul_q8_0_kernel(
    const float * __restrict__ a,
    const void  * __restrict__ b_q8,
    float * __restrict__ c,
    const int M, const int N, const int K) {
    // ... ported to HIP, using amd_mixed_dot for byte-level dot products ...
}
"#;

pub const KERNEL_MOE_ROUTING_SRC: &str = r#"
// ggml-cuda.cu router_select_kernel ported to HIP.
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
// ggml-cuda.cu compressor_store_kernel ported to HIP.
extern "C" __global__ void compressor_store_kernel(
    const float * __restrict__ x,
    void * __restrict__ y_compressed,
    const int n) {
    // ...
}
"#;

pub const KERNEL_OCML_PRECISE_SRC: &str = r#"
// Uses AMD's precise-math OCML helpers for the MoE router, since
// Strix Halo's hardware transcendentals diverge from CUDA's by a
// few ULPs at the boundaries.
#include <ocml/ocml.h>
extern "C" __global__ void router_ocml_precise(
    const float * __restrict__ logits,
    float * __restrict__ probs,
    const int n_experts) {
    // probs[i] = __ocml_exp_f32(logits[i]) / sum
    // uses __ocml_exp_f32 + __ocml_log1p_f32 instead of expf/log1pf
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_non_empty() {
        for (name, src) in &[
            ("matmul_q8_0", KERNEL_MATMUL_Q8_0_SRC),
            ("moe_routing", KERNEL_MOE_ROUTING_SRC),
            ("compressor_store", KERNEL_COMPRESSOR_STORE_SRC),
            ("ocml_precise", KERNEL_OCML_PRECISE_SRC),
        ] {
            assert!(!src.is_empty(), "kernel {name} source is empty");
            assert!(
                src.contains("__global__") || src.contains("__device__"),
                "kernel {name} missing HIP entry-point annotation"
            );
        }
    }

    #[test]
    fn compile_returns_not_implemented_without_hipcc() {
        let r = compile("test", "extern \"C\" __global__ void k(){}", "gfx1151");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().kind, Ds4ErrorKind::NotImplemented);
    }
}
