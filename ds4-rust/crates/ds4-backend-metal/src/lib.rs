// DS4 (DwarfStar) -- Metal backend.
//
// Hand-rolled port of `third_party/ggml/src/ggml-metal/*.metal` (19 MSL
// files, ~12k LoC) and the Objective-C `ds4_metal.m` host wrapper.
//
// Strategy mirrors `ds4-backend-cuda`: kernel sources are stored as
// `&'static str` constants and compiled on first use via
// `xcrun metal`. On non-macOS hosts, kernel compilation reports
// the missing toolchain while `load_model` uses a CPU-compatible path.

#![allow(non_camel_case_types)]

pub const CRATE_NAME: &str = "ds4-backend-metal";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod backend;
pub mod buffers;
pub mod kernels;

pub use backend::{KernelCache, MetalBackend, MetalModel};
pub use buffers::{Buffer, BufferPool, DType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-backend-metal");
        assert!(!VERSION.is_empty());
    }
}
