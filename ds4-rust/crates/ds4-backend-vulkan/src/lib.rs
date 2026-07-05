// DS4 (DwarfStar) -- Vulkan backend.
//
// Provides cross-vendor Vulkan device discovery and memory allocation for the
// DS4 runtime. Compute graph execution is wired to fail closed until SPIR-V
// kernels are present for this backend.

#![allow(unsafe_code)]

pub const CRATE_NAME: &str = "ds4-backend-vulkan";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod backend;
pub mod runtime;

pub use backend::VulkanBackend;
pub use runtime::{
    VulkanBuffer, VulkanDeviceInfo, VulkanMemoryHeap, VulkanMemoryKind, VulkanRuntime,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-backend-vulkan");
        assert!(!VERSION.is_empty());
    }
}
