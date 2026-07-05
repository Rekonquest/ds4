// DS4 (DwarfStar) — tensor crate.
//
// Goal of this crate
// ------------------
// `ds4-tensor` is the typed tensor surface for the entire workspace.
// Downstream crates (ds4-quant, ds4-core, the backends, the CLI, …)
// all depend on it for `DType`, `Shape`, `Tensor` and a couple of
// round-trip helpers (`from_f32`, `as_f32`). The C source defines
// `ds4_tensor` as a thin wrapper, and the Rust rewrite was approved
// to vendor `candle-core` and re-export its `DType` / `Tensor` /
// `Device` / `Shape` types through the same `ds4_tensor` API.
//
// Vendored candle-core status
// ---------------------------
// `candle-core` is vendored byte-identical at
// `third_party/candle-core/candle-core/` (see
// `target/vendor-candle.log` for the verification trail). The path
// dep is *not* currently declared in `Cargo.toml`; see the comment
// in that file for the precise reason.
//
// Why the path dep is missing right now
// -------------------------------------
// The vendored `candle-core/Cargo.toml` was copied verbatim from
// upstream and still inherits a number of fields from a *parent*
// Cargo workspace (`version.workspace = true`, `edition.workspace
// = true`, `license.workspace = true`, every dependency that uses
// `workspace = true`). Upstream candle is consumed from inside the
// huggingface/candle workspace; we vendored only the inner
// `candle-core/` subcrate without its surrounding workspace, so
// cargo fails to parse its manifest with:
//
//   error inheriting `description` from workspace root manifest's
//   `workspace.package.description`
//   `workspace.package.description` was not defined
//
// Confirmed by:
//   cargo check -p candle-core
//     -> package ID specification `candle-core` did not match any
//        packages
//   cargo check --manifest-path \
//       third_party/candle-core/candle-core/Cargo.toml
//     -> failed to parse manifest … `workspace.package.description`
//        was not defined
//   cargo check -p ds4-tensor (with the path dep declared)
//     -> failed to load manifest for dependency `candle-core`
//
// Cargo parses the manifest of every dependency listed in a member
// crate's `[dependencies]` table — even optional ones, even unused
// ones — so the path dep *cannot* be present in `Cargo.toml` until
// the upstream manifest is patched. Until then we also can't gate
// the candle re-exports behind a `vendored-candle` feature, because
// features are crate-local and cannot remove a path declaration.
//
// To re-enable the integration:
//   1. Patch `third_party/candle-core/candle-core/Cargo.toml` so it
//      hard-codes the fields it currently inherits from the
//      upstream workspace (`version`, `edition`, `license`,
//      `description`, `keywords`, `categories`, `repository`).
//   2. Replace the inline `workspace = true` deps with version
//      specs (or wrap candle-core inside a synthetic
//      `third_party/candle-core/Cargo.toml` workspace that supplies
//      `workspace.package` and the `candle-kernels`, `cudarc`, etc.
//      workspace dependencies it references).
//   3. Re-add the path dep to `Cargo.toml`:
//        candle-core = { path = "../../third_party/candle-core/candle-core", optional = true }
//   4. Add a `vendored-candle = ["dep:candle-core"]` feature and
//      switch the re-exports in `dtype.rs`, `shape.rs` and
//      `tensor.rs` to gate on that feature (the diff is already
//      sketched in the prior version of those files).
//
// What we ship today
// ------------------
// With no `vendored-candle` feature, this crate exposes a real,
// working pure-Rust tensor implementation under `crate::pure` and
// re-exports it as the public `DType` / `Shape` / `Tensor` /
// `Device` types. `from_f32` and `as_f32` are real,
// `from_le_bytes`-based round-trips, and every public tensor helper
// works in the pure-Rust path. Once the
// upstream manifest is patched, the same public API can swap to Candle —
// the public API and round-trip semantics are preserved by the
// `dtype.rs` / `shape.rs` / `tensor.rs` re-export modules.

pub const CRATE_NAME: &str = "ds4-tensor";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pure-Rust tensor implementation used while the vendored
/// `candle-core` manifest is being integrated. It is a complete typed
/// tensor surface (`DType`, `Shape`, `Tensor`) with correct
/// byte-level f32 round-trip semantics.
pub mod pure {
    /// Local DType enum. Mirrors the candle DType variants the
    /// rest of the workspace actually consumes. Kept intentionally
    /// small — we only model what ds4-quant / ds4-core need today.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum DType {
        F32,
        F16,
        BF16,
        F64,
        I8,
        U8,
        I64,
        U32,
        U64,
    }

    impl DType {
        /// Size in bytes of a single element of this dtype.
        pub fn byte_size(self) -> usize {
            match self {
                DType::F32 | DType::U32 => 4,
                DType::F16 | DType::BF16 => 2,
                DType::I8 | DType::U8 => 1,
                DType::F64 | DType::I64 | DType::U64 => 8,
            }
        }

        /// Whether the dtype is a floating-point type.
        pub fn is_float(self) -> bool {
            matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::F64)
        }
    }

    /// Local Shape container. Models a row-major contiguous tensor
    /// shape — exactly what candle-core's `Shape` models too, so
    /// swapping the implementation is a no-op at call sites.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct Shape {
        dims: Vec<usize>,
    }

    impl Shape {
        pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
            Self {
                dims: dims.into_iter().collect(),
            }
        }

        pub fn from_dims(dims: &[usize]) -> Self {
            Self {
                dims: dims.to_vec(),
            }
        }

        pub fn dims(&self) -> &[usize] {
            &self.dims
        }

        pub fn rank(&self) -> usize {
            self.dims.len()
        }

        pub fn numel(&self) -> usize {
            self.dims.iter().product()
        }

        pub fn is_contiguous(&self) -> bool {
            true
        }
    }

    impl From<Vec<usize>> for Shape {
        fn from(dims: Vec<usize>) -> Self {
            Self { dims }
        }
    }

    impl From<&[usize]> for Shape {
        fn from(dims: &[usize]) -> Self {
            Self {
                dims: dims.to_vec(),
            }
        }
    }

    /// Local tensor container. Stores a row-major contiguous byte
    /// buffer tagged with a `DType` and a `Shape`. The f32
    /// `from_f32` / `as_f32` round-trip uses `f32::to_le_bytes` /
    /// `f32::from_le_bytes` — no `unsafe`, no alignment UB.
    #[derive(Debug, Clone)]
    pub struct Tensor {
        pub dtype: DType,
        pub shape: Shape,
        pub data: Vec<u8>,
        pub device: Device,
    }

    impl Tensor {
        /// Allocate an uninitialised tensor. The backing buffer is
        /// zero-filled (the workspace only ever treats uninitialised
        /// memory as input data, never as a security-sensitive
        /// surface, so zeroing is safe and matches what
        /// `candle-core::Tensor::zeros` does).
        pub fn new(dtype: DType, shape: Shape) -> Self {
            let numel = shape.numel();
            let bytes = numel * dtype.byte_size();
            Self {
                dtype,
                shape,
                data: vec![0u8; bytes],
                device: Device::Cpu,
            }
        }

        pub fn zeros(dtype: DType, shape: Shape) -> Self {
            Self::new(dtype, shape)
        }

        /// Allocate a tensor and copy a host f32 slice into it.
        /// The slice length must match `shape.numel()`; we assert
        /// to surface bugs at the call site rather than silently
        /// truncate.
        pub fn from_f32(data: &[f32], shape: Shape) -> Self {
            assert_eq!(
                data.len(),
                shape.numel(),
                "ds4_tensor::Tensor::from_f32: data.len() ({}) != shape.numel() ({})",
                data.len(),
                shape.numel()
            );
            let mut t = Self::new(DType::F32, shape);
            for (i, v) in data.iter().enumerate() {
                let bytes = v.to_le_bytes();
                let off = i * 4;
                t.data[off..off + 4].copy_from_slice(&bytes);
            }
            t
        }

        /// Read the whole tensor back as a host `Vec<f32>`. Panics
        /// if the dtype is not F32; callers that need cross-dtype
        /// conversion go through the `convert` module.
        pub fn as_f32(&self) -> Vec<f32> {
            assert_eq!(self.dtype, DType::F32, "Tensor::as_f32 requires F32 dtype");
            let mut out = Vec::with_capacity(self.data.len() / 4);
            for chunk in self.data.chunks_exact(4) {
                let mut b = [0u8; 4];
                b.copy_from_slice(chunk);
                out.push(f32::from_le_bytes(b));
            }
            out
        }

        /// Read the whole tensor back into a caller-supplied
        /// `&mut [f32]`. Same constraints as [`Self::as_f32`].
        pub fn as_f32_into(&self, out: &mut [f32]) {
            assert_eq!(
                self.dtype,
                DType::F32,
                "Tensor::as_f32_into requires F32 dtype"
            );
            assert_eq!(
                out.len() * 4,
                self.data.len(),
                "Tensor::as_f32_into: out.len() ({}) does not match tensor element count ({})",
                out.len(),
                self.data.len() / 4
            );
            for (chunk, slot) in self.data.chunks_exact(4).zip(out.iter_mut()) {
                let mut b = [0u8; 4];
                b.copy_from_slice(chunk);
                *slot = f32::from_le_bytes(b);
            }
        }

        /// Total number of bytes occupied by the backing buffer.
        pub fn byte_len(&self) -> usize {
            self.data.len()
        }

        /// Borrow the underlying byte buffer.
        pub fn as_bytes(&self) -> &[u8] {
            &self.data
        }
    }

    /// Local device enum. We only need the CPU device today; the
    /// `vendored-candle` feature surfaces candle's full
    /// `Device` enum (`Cpu`, `Cuda(n)`, `Metal(n)`, …) once the
    /// upstream manifest is wired up.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub enum Device {
        #[default]
        Cpu,
    }

    impl Device {
        pub fn cpu() -> Self {
            Self::Cpu
        }

        pub fn is_cpu(self) -> bool {
            matches!(self, Device::Cpu)
        }
    }
}

pub mod dtype;
pub mod shape;
pub mod tensor;

// ---------------------------------------------------------------------------
// Public re-exports.
//
// Today this surfaces the local pure-Rust implementations (see the
// module-level note above for the path back to candle). Once the
// `vendored-candle` feature is wired up these `pub use` statements
// will swap over to `candle_core::{DType, Device, Shape, Tensor}`.
// ---------------------------------------------------------------------------

pub use self::pure::{DType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Errors.
//
// `ds4_tensor` errors are surfaced as `thiserror` enums. With the
// vendored-candle feature on, this becomes a thin newtype over
// `candle_core::Error`; today we expose a self-contained enum that
// covers everything the workspace actually produces.
// ---------------------------------------------------------------------------

/// Errors produced by `ds4-tensor` operations.
#[derive(Debug, thiserror::Error)]
pub enum D4TensorError {
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("dtype mismatch: operation requires {required:?}, tensor is {actual:?}")]
    DtypeMismatch { required: DType, actual: DType },

    #[error("tensor is empty (no elements)")]
    EmptyTensor,

    #[error("tensor backend error: {0}")]
    Backend(String),
}

/// Crate-wide `Result` alias.
pub type D4TensorResult<T> = Result<T, D4TensorError>;

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-tensor");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn shape_numel_and_rank() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::U8.byte_size(), 1);
        assert_eq!(DType::U32.byte_size(), 4);
        assert_eq!(DType::F64.byte_size(), 8);
        assert_eq!(DType::I64.byte_size(), 8);
        assert_eq!(DType::U64.byte_size(), 8);
    }

    #[test]
    fn dtype_is_float() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I8.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::U32.is_float());
    }

    #[test]
    fn f32_tensor_roundtrip() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_f32(&data, Shape::new([4]));
        let out = t.as_f32();
        assert_eq!(out, data);
    }

    #[test]
    fn f32_tensor_roundtrip_into_buffer() {
        let data = [-1.5f32, 0.0, 0.5, 42.0];
        let t = Tensor::from_f32(&data, Shape::new([4]));
        let mut out = [0.0f32; 4];
        t.as_f32_into(&mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn tensor_helpers_roundtrip() {
        let data = [10.0f32, 20.0, 30.0];
        let t = tensor::from_f32(&data, Shape::new([3]));
        let out = tensor::as_f32(&t);
        assert_eq!(out, data);
    }

    #[test]
    fn tensor_byte_len_matches_shape() {
        let t = Tensor::new(DType::F32, Shape::new([3, 5]));
        assert_eq!(t.byte_len(), 3 * 5 * 4);
        assert_eq!(t.as_bytes().len(), 3 * 5 * 4);
    }

    #[test]
    fn device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
        assert!(Device::cpu().is_cpu());
    }

    #[test]
    fn shape_from_vec_and_slice() {
        let a: Shape = vec![2, 3].into();
        let b: Shape = Shape::from_dims(&[2_usize, 3]);
        assert_eq!(a, b);
        assert_eq!(a.numel(), 6);
    }

    #[test]
    #[should_panic(expected = "data.len()")]
    fn from_f32_panics_on_size_mismatch() {
        let data = [1.0f32, 2.0];
        let _ = Tensor::from_f32(&data, Shape::new([3]));
    }

    #[test]
    fn dtype_compat_byte_size() {
        assert_eq!(dtype::compat::byte_size(DType::F32), DType::F32.byte_size());
        assert_eq!(dtype::compat::byte_size(DType::F16), DType::F16.byte_size());
        assert_eq!(dtype::compat::byte_size(DType::U8), DType::U8.byte_size());
    }

    #[test]
    fn error_display_messages_are_meaningful() {
        let e = D4TensorError::ShapeMismatch {
            expected: 6,
            actual: 4,
        };
        let s = format!("{e}");
        assert!(s.contains("6"));
        assert!(s.contains("4"));
    }
}
