//! Cross-format smoke tests for the crate-level metadata.

use super::*;

#[test]
fn metadata_is_sane() {
    assert_eq!(CRATE_NAME, "ds4-quant");
    assert!(!VERSION.is_empty());
}

#[test]
fn block_sizes_match_c_layout() {
    use crate::format::{
        block_bytes, block_size, IQ2_XXS_BLOCK, Q2_K_BLOCK, Q4_K_BLOCK, Q8_0_BLOCK,
    };
    assert_eq!(block_size(QuantKind::Q8_0), Q8_0_BLOCK);
    assert_eq!(block_size(QuantKind::Q4_K), Q4_K_BLOCK);
    assert_eq!(block_size(QuantKind::Q2_K), Q2_K_BLOCK);
    assert_eq!(block_size(QuantKind::Iq2Xxs), IQ2_XXS_BLOCK);
    assert_eq!(block_bytes(QuantKind::Q8_0), 2 + 32);
    assert_eq!(block_bytes(QuantKind::Q4_K), 2 + 2 + 12 + 128);
    assert_eq!(block_bytes(QuantKind::Q2_K), 16 + 64 + 2 + 2);
    assert_eq!(block_bytes(QuantKind::Iq2Xxs), 2 + 2 * 32);
}
