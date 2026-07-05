//! DType re-exports.
//!
//! Today this surfaces the local pure-Rust `DType` enum (see
//! `crate::pure`). When the `vendored-candle` feature is wired up
//! (see `Cargo.toml` for the current blocker) this module will
//! switch to re-exporting `candle_core::DType` instead.

pub use crate::pure::DType;

pub mod compat {
    //! Conversion helpers for the local DType. Mirrors the
    //! `dtype::byte_size` API the workspace relied on before the
    //! vendored-candle feature existed.
    use super::DType;

    /// Size in bytes of a single element of `d`.
    pub fn byte_size(d: DType) -> usize {
        d.byte_size()
    }
}
