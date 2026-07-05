// DS4 (DwarfStar) — CPU backend.
//
// Correctness-path backend. Hand-rolled reference kernels whose
// numerics are guaranteed to match the C reference implementation
// exactly. The GPU backends (CUDA / Metal / ROCm) validate against
// this crate's outputs at test time.

#![allow(clippy::result_large_err)]

pub const CRATE_NAME: &str = "ds4-backend-cpu";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod attention;
pub mod backend;
pub mod matmul;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;

pub use backend::{CpuBackend, CpuModel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-backend-cpu");
        assert!(!VERSION.is_empty());
    }
}
