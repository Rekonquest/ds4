// DS4 (DwarfStar) -- CUDA backend.
//
// Hand-rolled port of `third_party/ggml/src/ggml-cuda/*.cu` and the
// DS4-specific kernels (`ds4_cuda.cu` + `ds4_iq2_tables_cuda.inc`).
//
// Strategy:
// - Kernel sources are stored as `&'static str` constants.
// - `kernels::compile()` runs `nvcc` to produce a cubin.
// - On machines without the CUDA toolkit, kernel compilation reports
//   the missing toolchain while `load_model` uses a CPU-compatible path.

#![allow(non_camel_case_types)]

pub const CRATE_NAME: &str = "ds4-backend-cuda";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod backend;
pub mod buffers;
pub mod kernels;

pub use backend::{CudaBackend, CudaModel, KernelCache};
pub use buffers::{Buffer, BufferPool, DType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-backend-cuda");
        assert!(!VERSION.is_empty());
    }
}
