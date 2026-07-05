//! Shape re-exports.
//!
//! Today this surfaces the local pure-Rust `Shape` (see
//! `crate::pure`). When the `vendored-candle` feature is wired up
//! (see `Cargo.toml` for the current blocker) this module will
//! switch to re-exporting `candle_core::Shape` instead.

pub use crate::pure::Shape;
