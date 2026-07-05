// DS4 (DwarfStar) — quantization crate.
//
// Hosts the per-format quant / dequant / dot kernels ported from the
// upstream CUDA sources and the IQ2_XXS LUT tables.
//
// The variant and type names below intentionally mirror the upstream
// GGML/quant format identifiers (Q4_K, Q2_K, IQ2_XXS). Suppress the
// camel-case lint for these symbols; do not rename them.

#![allow(non_camel_case_types)]

pub const CRATE_NAME: &str = "ds4-quant";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The set of quantization formats we support, mirroring the upstream
/// GGML `ggml_type` enum subset. Naming uses Rust's PascalCase
/// convention but the wire-level identifiers stay verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantKind {
    Q8_0,
    Q4_K,
    Q2_K,
    Iq2Xxs,
    F16,
    F32,
}

pub mod f16;
pub mod f32;
pub mod format;
pub mod iq2_xxs;
pub mod luts;
pub mod q2_k;
pub mod q4_k;
pub mod q8_0;

#[cfg(test)]
mod tests;
