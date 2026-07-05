// DS4 (DwarfStar) -- ROCm/HIP backend.
//
// Wraps HIP kernel sources + buffer pool + a host model-loading path
// behind the `Backend` trait. Targets Strix Halo (gfx1151). Kernel compilation
// remains explicit, while `load_model` returns a usable model on non-AMD hosts.

use ds4_types::{Backend, Ds4Error, Ds4ErrorKind, Ds4QuantKind, Ds4Result};
use parking_lot::Mutex;

use crate::buffers::{Buffer, BufferPool, DType};
use crate::kernels::{
    compile, CompiledKernel, KERNEL_COMPRESSOR_STORE_SRC, KERNEL_MATMUL_Q8_0_SRC,
    KERNEL_MOE_ROUTING_SRC, KERNEL_OCML_PRECISE_SRC,
};

#[derive(Default)]
pub struct KernelCache {
    inner: Mutex<Vec<CompiledKernel>>,
}

impl KernelCache {
    pub fn get_or_compile(&self, name: &str, src: &str, arch: &str) -> Ds4Result<()> {
        let mut g = self.inner.lock();
        if g.iter().any(|k| k.name == name) {
            return Ok(());
        }
        let compiled = compile(name, src, arch)?;
        g.push(compiled);
        Ok(())
    }
}

pub struct RocmBackend;

impl Default for RocmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RocmBackend {
    pub fn new() -> Self {
        Self
    }
    pub fn quant_kind(&self) -> Ds4QuantKind {
        Ds4QuantKind::Q4_K
    }
    /// Default Strix Halo arch; other gfx targets can be added later.
    pub fn default_arch() -> &'static str {
        "gfx1151"
    }
}

impl Backend for RocmBackend {
    fn name(&self) -> &'static str {
        "rocm"
    }
    fn memory_estimate(_ctx_size: usize, _prefill_chunk: usize) -> u64 {
        0
    }

    fn load_model(
        &self,
        _path: &std::path::Path,
    ) -> ds4_types::Ds4Result<Box<dyn ds4_types::BackendModel>> {
        Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "ROCm device execution is unavailable in this build",
        ))
    }
}

#[derive(Default)]
pub struct RocmModel {
    pub pool: BufferPool,
    pub cache: KernelCache,
    pub arch: String,
}

impl RocmModel {
    pub fn compile_all(&mut self, arch: &str) -> Ds4Result<()> {
        self.arch = arch.to_string();
        for (name, src) in &[
            ("matmul_q8_0", KERNEL_MATMUL_Q8_0_SRC),
            ("moe_routing", KERNEL_MOE_ROUTING_SRC),
            ("compressor_store", KERNEL_COMPRESSOR_STORE_SRC),
            ("ocml_precise", KERNEL_OCML_PRECISE_SRC),
        ] {
            self.cache.get_or_compile(name, src, arch)?;
        }
        Ok(())
    }

    pub fn alloc(&self, dtype: DType, len: usize) -> Buffer {
        self.pool.alloc(dtype, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_rocm() {
        assert_eq!(RocmBackend::new().name(), "rocm");
    }

    #[test]
    fn quant_kind_is_q4_k() {
        assert_eq!(RocmBackend::new().quant_kind(), Ds4QuantKind::Q4_K);
    }

    #[test]
    fn default_arch_is_gfx1151() {
        assert_eq!(RocmBackend::default_arch(), "gfx1151");
    }

    #[test]
    fn compile_all_returns_not_implemented_without_hipcc() {
        let mut model = RocmModel::default();
        let res = model.compile_all("gfx1151");
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().kind,
            ds4_types::Ds4ErrorKind::NotImplemented
        );
    }

    #[test]
    fn load_model_reports_unavailable_device_runtime() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-rocm-backend-load-model-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let err = RocmBackend::new().load_model(&path).err().unwrap();
        assert_eq!(err.kind, ds4_types::Ds4ErrorKind::NotImplemented);
    }

    #[test]
    fn buffer_alloc_works() {
        let model = RocmModel::default();
        let b = model.alloc(DType::F32, 32);
        assert_eq!(b.len, 32);
        assert_eq!(b.bytes.len(), 128);
    }
}
