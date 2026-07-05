use std::path::Path;

use ds4_gguf::{GgufFile, ModelSpec};
use ds4_types::{Backend, BackendModel, Ds4Error, Ds4ErrorKind, Ds4Result};

use crate::runtime::{VulkanDeviceInfo, VulkanRuntime};

pub struct VulkanBackend;

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn probe(&self) -> Ds4Result<VulkanDeviceInfo> {
        Ok(VulkanRuntime::load()?.device_info().clone())
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
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
        let runtime = VulkanRuntime::load().map_err(|err| {
            Ds4Error::new(
                Ds4ErrorKind::Backend,
                format!("Vulkan backend unavailable: {err}"),
            )
        })?;
        Err(Ds4Error::new(
            Ds4ErrorKind::Backend,
            format!(
                "Vulkan runtime is available on {}, but DS4 Vulkan model execution requires SPIR-V kernels for {} layers",
                runtime.device_name(),
                spec.dims.layers
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_vulkan() {
        assert_eq!(VulkanBackend::new().name(), "vulkan");
    }

    #[test]
    fn vulkan_backend_loads_runtime_then_fails_closed_for_synthetic_model() {
        let dir = std::env::temp_dir().join(format!("ds4-backend-vulkan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let err = match VulkanBackend::new().load_model(&path) {
            Ok(_) => panic!("Vulkan model loading should fail closed without SPIR-V kernels"),
            Err(err) => err,
        };
        assert_eq!(err.kind, Ds4ErrorKind::Backend, "{err}");
        assert!(
            err.message.contains("Vulkan backend unavailable")
                || err.message.contains("SPIR-V kernels")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
