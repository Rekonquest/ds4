// DS4 (DwarfStar) -- ROCm/HIP backend.
//
// Hand-rolled port of `third_party/ggml/src/ggml-cuda/*.cu` (HIP
// translation) and the DS4-specific ROCm extensions
// (`ds4_rocm.cu` + `rocm/*.cuh`). Targets Strix Halo (gfx1151).

#![allow(non_camel_case_types)]

pub const CRATE_NAME: &str = "ds4-backend-rocm";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod backend;
pub mod buffers;
pub mod kernels;

pub use backend::{KernelCache, RocmBackend, RocmModel};
pub use buffers::{Buffer, BufferPool, DType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-backend-rocm");
        assert!(!VERSION.is_empty());
    }
}
