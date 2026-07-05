//! Tensor re-exports and round-trip helpers.
//!
//! Today this surfaces the local pure-Rust `Tensor` (see
//! `crate::pure`). When the `vendored-candle` feature is wired up
//! (see `Cargo.toml` for the current blocker) this module will
//! switch to re-exporting `candle_core::Tensor` instead and the
//! helpers will go through `candle_core::Tensor::new` /
//! `to_vec1::<f32>`.
//!
//! Either way the round-trip helpers below are real: they do the
//! work with deterministic pure-Rust storage.

use crate::shape::Shape;

/// The tensor type surfaced through this crate.
pub use crate::pure::Tensor as Ds4Tensor;

/// Convenience alias so `use ds4_tensor::tensor::Tensor` keeps
/// working even after switching to the candle-backed implementation.
pub use self::Ds4Tensor as Tensor;

/// Build a tensor from a host `f32` slice using the local pure-Rust
/// backend.
pub fn from_f32(data: &[f32], shape: Shape) -> crate::pure::Tensor {
    crate::pure::Tensor::from_f32(data, shape)
}

/// Read a 1-D f32 tensor back to a host `Vec<f32>` using the local
/// pure-Rust backend.
pub fn as_f32(t: &crate::pure::Tensor) -> Vec<f32> {
    t.as_f32()
}
