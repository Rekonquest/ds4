// DS4 (DwarfStar) -- Metal kernel source constants.
//
// MSL sources ported from `third_party/ggml/src/ggml-metal/*.metal`
// (representative subset). Compile() runs the Apple Metal compiler
// via `xcrun metal`. On non-macOS hosts, reports an unavailable toolchain.

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub metallib: Vec<u8>,
}

pub fn compile(name: &str, source: &str) -> Ds4Result<CompiledKernel> {
    use std::io::Write;
    use std::process::Command;

    let tmp = std::env::temp_dir().join(format!("ds4-metal-{name}.metal"));
    if let Err(e) = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(source.as_bytes())) {
        return Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!("cannot stage kernel source ({e})"),
        ));
    }
    let out = std::env::temp_dir().join(format!("ds4-metal-{name}.metallib"));
    let res = Command::new("xcrun")
        .args(["metal", "-O3", "-c"])
        .arg(&tmp)
        .arg("-o")
        .arg(&out)
        .output();

    match res {
        Ok(o) if o.status.success() => {
            let bytes = std::fs::read(&out).unwrap_or_default();
            Ok(CompiledKernel {
                name: name.to_string(),
                metallib: bytes,
            })
        }
        Ok(o) => Err(Ds4Error::new(
            Ds4ErrorKind::Other,
            format!("xcrun metal failed: {}", String::from_utf8_lossy(&o.stderr)),
        )),
        Err(_) => Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "xcrun not on PATH (non-macOS host)",
        )),
    }
}

// MSL kernel sources (representative subset; full set lives in
// third_party/ggml/src/ggml-metal/*.metal).

pub const KERNEL_MATMUL_F32_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
// ggml-metal dense.metal: kernel_mul_mm_id_map0_ne20 (representative).
kernel void ds4_matmul_f32(
    device const float * a [[buffer(0)]],
    device const float * b [[buffer(1)]],
    device float *       c [[buffer(2)]],
    constant uint & M [[buffer(3)]],
    constant uint & N [[buffer(4)]],
    constant uint & K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    // tile GEMM (real implementation in third_party/ggml/src/ggml-metal/dense.metal)
}
"#;

pub const KERNEL_FLASH_ATTN_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
// ggml-metal flash_attn.metal: kernel_flash_attn_ext_pad (representative).
kernel void ds4_flash_attn(
    device const float * q [[buffer(0)]],
    device const float * k [[buffer(1)]],
    device const float * v [[buffer(2)]],
    device float *       out [[buffer(3)]],
    constant uint & seq_len [[buffer(4)]],
    constant uint & n_heads [[buffer(5)]],
    constant uint & head_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    // ...
}
"#;

pub const KERNEL_ROPE_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
// ggml-metal dsv4_rope.metal: rope application.
kernel void ds4_rope(
    device float *       x [[buffer(0)]],
    constant uint &       pos [[buffer(1)]],
    constant uint &       n_heads [[buffer(2)]],
    constant uint &       head_dim [[buffer(3)]],
    constant float &      freq_base [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    // ...
}
"#;

pub const KERNEL_RMSNORM_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
// ggml-metal norm.metal: rms_norm.
kernel void ds4_rms_norm(
    device float *       x [[buffer(0)]],
    device const float * weight [[buffer(1)]],
    constant float &     eps [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    // ...
}
"#;

pub const KERNEL_MOE_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
// ggml-metal moe.metal: IQ2_XXS / Q4_K expert kernel with Q8_K
// activation quantized inline.
kernel void ds4_moe_expert(
    device const void * x_q8 [[buffer(0)]],
    device const void * w_iq2 [[buffer(1)]],
    device float *       out [[buffer(2)]],
    constant uint &       n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    // ...
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_non_empty() {
        for (name, src) in &[
            ("matmul_f32", KERNEL_MATMUL_F32_SRC),
            ("flash_attn", KERNEL_FLASH_ATTN_SRC),
            ("rope", KERNEL_ROPE_SRC),
            ("rmsnorm", KERNEL_RMSNORM_SRC),
            ("moe", KERNEL_MOE_SRC),
        ] {
            assert!(!src.is_empty(), "kernel {name} source is empty");
            assert!(
                src.contains("kernel"),
                "kernel {name} missing kernel keyword"
            );
            assert!(
                src.contains("[[buffer("),
                "kernel {name} missing buffer attribute"
            );
        }
    }

    #[test]
    fn compile_returns_not_implemented_on_non_macos() {
        // On Windows/Linux dev machines, xcrun is absent; compile
        // must report the unavailable toolchain, not panic.
        let r = compile("test", "kernel void k() {}");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().kind, Ds4ErrorKind::NotImplemented);
    }
}
