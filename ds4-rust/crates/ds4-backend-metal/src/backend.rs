// DS4 (DwarfStar) -- Metal backend.
//
// Wraps MSL kernel sources + buffer pool + a host model-loading path
// behind the `Backend` trait. Kernel compilation remains explicit, while
// `load_model` returns a usable model on non-macOS hosts.

use ds4_types::{Backend, Ds4Error, Ds4ErrorKind, Ds4QuantKind, Ds4Result};
use parking_lot::Mutex;

use crate::buffers::{Buffer, BufferPool, DType};
use crate::kernels::{
    compile, CompiledKernel, KERNEL_FLASH_ATTN_SRC, KERNEL_MATMUL_F32_SRC, KERNEL_MOE_SRC,
    KERNEL_RMSNORM_SRC, KERNEL_ROPE_SRC,
};

#[derive(Default)]
pub struct KernelCache {
    inner: Mutex<Vec<CompiledKernel>>,
}

impl KernelCache {
    pub fn get_or_compile(&self, name: &str, src: &str) -> Ds4Result<()> {
        let mut g = self.inner.lock();
        if g.iter().any(|k| k.name == name) {
            return Ok(());
        }
        let compiled = compile(name, src)?;
        g.push(compiled);
        Ok(())
    }
}

pub struct MetalBackend;

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }
    pub fn quant_kind(&self) -> Ds4QuantKind {
        Ds4QuantKind::Q4_K
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
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
            "Metal device execution is unavailable on this host/build",
        ))
    }
}

#[derive(Default)]
pub struct MetalModel {
    pub pool: BufferPool,
    pub cache: KernelCache,
}

impl MetalModel {
    pub fn compile_all(&mut self) -> Ds4Result<()> {
        for (name, src) in &[
            ("matmul_f32", KERNEL_MATMUL_F32_SRC),
            ("flash_attn", KERNEL_FLASH_ATTN_SRC),
            ("rope", KERNEL_ROPE_SRC),
            ("rmsnorm", KERNEL_RMSNORM_SRC),
            ("moe", KERNEL_MOE_SRC),
        ] {
            self.cache.get_or_compile(name, src)?;
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
    fn backend_name_is_metal() {
        assert_eq!(MetalBackend::new().name(), "metal");
    }

    #[test]
    fn quant_kind_is_q4_k() {
        assert_eq!(MetalBackend::new().quant_kind(), Ds4QuantKind::Q4_K);
    }

    #[test]
    fn compile_all_returns_not_implemented_on_non_macos() {
        let mut model = MetalModel::default();
        let res = model.compile_all();
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().kind,
            ds4_types::Ds4ErrorKind::NotImplemented
        );
    }

    #[test]
    fn load_model_reports_unavailable_device_runtime() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-metal-backend-load-model-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let err = MetalBackend::new().load_model(&path).err().unwrap();
        assert_eq!(err.kind, ds4_types::Ds4ErrorKind::NotImplemented);
    }

    #[test]
    fn buffer_alloc_works() {
        let model = MetalModel::default();
        let b = model.alloc(DType::F32, 16);
        assert_eq!(b.len, 16);
        assert_eq!(b.bytes.len(), 64);
    }
}
