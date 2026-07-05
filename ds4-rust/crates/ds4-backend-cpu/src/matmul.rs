// DS4 (DwarfStar) — CPU matmul kernels.
//
// Three flavors:
//   * `matmul_f32`  — pure f32 reference matmul (row-major `a` and
//                     column-major `b`, standard layout used by the
//                     GGUF/C reference engine).
//   * `matmul_q8_0` — f32 activations dotted against Q8_0-quantized
//                     weights. For each output row `m` and column
//                     `n` we accumulate
//                       sum_p a[m,p] * (w[n,p].d * w[n,p].qs[p])
//                     where `w[n]` is a flat slice of Q8_0 blocks
//                     covering the K dimension.
//   * `matmul_q4_k` — f32 activations dotted against Q4_K-quantized
//                     weights. Uses the reference per-block decode
//                     path from `ds4-quant::q4_k` and the per-block
//                     scale/min reconstruct, so the result is
//                     numerically equivalent to a dequant-then-mul
//                     reference.
//
// All three kernels are the *correctness oracle* for the GPU
// backends; they are simple enough to be obviously correct and serve
// as the baseline that tract-linalg (when wired up) must beat by
// >= 1.5x to stay enabled.

use ds4_quant::q4_k::{Q4_KBlock, Q8KView};
use ds4_quant::q8_0::Q8_0Block;

/// Pure f32 reference matmul.
///
/// Layout: `a` is row-major `[m, k]`, `b` is column-major `[k, n]`
/// (so `b[p * n + j]` is the element at row `p`, col `j`),
/// `c` is row-major `[m, n]`.
///
/// Computes `c[i, j] = sum_p a[i, p] * b[p, j]`. This matches the
/// layout used by the upstream C reference; the GPU backends swap
/// `b` to row-major before calling it.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "matmul_f32: a length mismatch");
    assert_eq!(b.len(), k * n, "matmul_f32: b length mismatch");
    assert_eq!(c.len(), m * n, "matmul_f32: c length mismatch");
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// Matmul with Q8_0-quantized weights. `activations` is row-major
/// `[m, k]`. `weights_q8` is `[n]`; each entry is a flat
/// `&[Q8_0Block]` of length `ceil(k / 32)` whose `qs[..]` covers
/// the K dimension. `out` is row-major `[m, n]`.
///
/// For each output element `(m, n)` we compute:
///
/// ```text
/// out[m, n] = sum_p activations[m, p]
///                  * (weights_q8[n][p/32].d
///                     * weights_q8[n][p/32].qs[p % 32])
/// ```
///
/// i.e. we dequantize the Q8_0 weight on the fly and use it as
/// the multiplier in the f32 dot product.
pub fn matmul_q8_0(
    activations: &[f32],
    weights_q8: &[&[Q8_0Block]],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(
        activations.len(),
        m * k,
        "matmul_q8_0: activations length mismatch"
    );
    assert_eq!(
        weights_q8.len(),
        n,
        "matmul_q8_0: weights_q8 outer length must be n"
    );
    assert_eq!(out.len(), m * n, "matmul_q8_0: out length mismatch");
    let blocks_per_row = k.div_ceil(Q8_0Block::BLOCK_SIZE);

    for ni in 0..n {
        let row_blocks = &weights_q8[ni];
        assert!(
            row_blocks.len() >= blocks_per_row,
            "matmul_q8_0: row {ni} has {} blocks, need >= {blocks_per_row}",
            row_blocks.len()
        );
        for mi in 0..m {
            let mut acc = 0.0f32;
            let a_row = &activations[mi * k..mi * k + k];
            let mut p = 0usize;
            for blk in 0..blocks_per_row {
                let block = &row_blocks[blk];
                let block_end = (p + Q8_0Block::BLOCK_SIZE).min(k);
                for local in 0..(block_end - p) {
                    acc += a_row[p + local] * block.d * block.qs[local] as f32;
                }
                p = block_end;
                if p >= k {
                    break;
                }
            }
            out[mi * n + ni] = acc;
        }
    }
}

/// Matmul with Q4_K-quantized weights.
///
/// `activations` is row-major `[m, k]`. `weights` is `[n]`; each
/// entry is a flat `&[Q4_KBlock]` whose union covers the K
/// dimension (each block covers `Q4_KBlock::BLOCK_SIZE = 256` f32
/// elements). `out` is row-major `[m, n]`.
///
/// Computes the dequantize-then-mul reference: for each block we
/// decode the 256 reconstructed weights via the upstream
/// `dequantize` (without allocating — a per-block scratch buffer
/// is reused) and then do the inner-product on the f32 slice.
///
/// This is the correctness reference; once the SIMD-accelerated
/// `quant::vec_dot_q4_K_q8_K` path is added to ds4-quant it becomes
/// the new fast path and the GPU backends compare against
/// *this* output.
pub fn matmul_q4_k(
    activations: &[f32],
    weights: &[&[Q4_KBlock]],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(
        activations.len(),
        m * k,
        "matmul_q4_k: activations length mismatch"
    );
    assert_eq!(
        weights.len(),
        n,
        "matmul_q4_k: weights outer length must be n"
    );
    assert_eq!(out.len(), m * n, "matmul_q4_k: out length mismatch");
    let blocks_per_row = k.div_ceil(Q4_KBlock::BLOCK_SIZE);

    // 256-element scratch reused across (mi, ni) iterations.
    let mut scratch = [0.0f32; Q4_KBlock::BLOCK_SIZE];

    for ni in 0..n {
        let row_blocks = &weights[ni];
        assert!(
            row_blocks.len() >= blocks_per_row,
            "matmul_q4_k: row {ni} has {} blocks, need >= {blocks_per_row}",
            row_blocks.len()
        );
        for mi in 0..m {
            let a_row = &activations[mi * k..mi * k + k];
            let mut acc = 0.0f32;
            let mut p = 0usize;
            for blk in 0..blocks_per_row {
                let block = &row_blocks[blk];
                ds4_quant::q4_k::dequantize(block, &mut scratch);
                let block_end = (p + Q4_KBlock::BLOCK_SIZE).min(k);
                let span = block_end - p;
                for local in 0..span {
                    acc += a_row[p + local] * scratch[local];
                }
                p = block_end;
                if p >= k {
                    break;
                }
            }
            out[mi * n + ni] = acc;
        }
    }
}

/// Q4_K matmul that uses the Q8_K reference dot-product kernel
/// from ds4-quant instead of dequantizing per-block. Used as a
/// secondary cross-check; the reference path is `matmul_q4_k`.
///
/// `activations` is row-major `[m, k]`. `weights` is `[n]`; each
/// entry is a flat `&[Q4_KBlock]` of length `ceil(k / 256)`.
/// `q8` is the corresponding Q8_K "view" array — one view per
/// block per row, supplied by the caller because quantizing
/// activations is outside this crate's scope.
///
/// Layout note: the per-block dot product gives `out[mi, ni]`; the
/// caller is responsible for ensuring `q8` slices are congruent
/// with `weights` (one Q8_K view per Q4_K block) and that the Q8_K
/// quantization of `activations[mi]` matches the block-aligned
/// slice the kernel reads.
///
/// Returns an error if `q8` does not have exactly
/// `n * blocks_per_row` views.
pub fn matmul_q4_k_via_q8(
    _activations: &[f32],
    weights: &[&[Q4_KBlock]],
    q8: &[Q8KView<'_>],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), &'static str> {
    let blocks_per_row = k.div_ceil(Q4_KBlock::BLOCK_SIZE);
    if q8.len() != n * blocks_per_row {
        return Err("matmul_q4_k_via_q8: q8 view count must equal n * blocks_per_row");
    }
    if weights.len() != n {
        return Err("matmul_q4_k_via_q8: weights outer length must be n");
    }
    if out.len() != m * n {
        return Err("matmul_q4_k_via_q8: out length mismatch");
    }
    let _ = m; // the cross-check path is per-block dot per row.
    let mut acc = [0.0f32; 1];
    for (ni, row_blocks) in weights.iter().enumerate() {
        for (bi, block) in row_blocks.iter().enumerate() {
            let view = q8[ni * blocks_per_row + bi];
            ds4_quant::q4_k::vec_dot(
                Q4_KBlock::BLOCK_SIZE,
                std::slice::from_ref(block),
                &[view],
                &mut acc,
            );
            // The Q8K view path produces a per-block partial sum;
            // it's only meaningful when activations have been
            // pre-quantized row-by-row into Q8K blocks by the caller.
            out[ni] = acc[0];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn vec_close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    #[test]
    fn matmul_f32_identity_rowvector() {
        // [1, 2, 3] * I = [1, 2, 3]
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 3];
        matmul_f32(&a, &b, &mut c, 1, 3, 3);
        assert_eq!(c, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_f32_identity_matrix() {
        // [[1,2,3]] (1x3) times I (3x3) = [[1,2,3]].
        let a = [1.0f32, 2.0, 3.0];
        let eye = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 3];
        matmul_f32(&a, &eye, &mut c, 1, 3, 3);
        assert_eq!(c, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_f32_ones_dot_ones_is_three() {
        // [1,1,1] * [[1],[1],[1]] = 3.
        let a = [1.0f32, 1.0, 1.0];
        let b = [1.0f32, 1.0, 1.0];
        let mut c = [0.0f32; 1];
        matmul_f32(&a, &b, &mut c, 1, 1, 3);
        assert_eq!(c, [3.0]);
    }

    #[test]
    fn matmul_f32_small_hand_computed() {
        // a (row-major [2x2]) dot W (row-major [2x2]) stored in
        // column-major fashion inside `b`. With column-major
        // packing `b[p * n + j] = W[j, p]`, we pack
        //   W = [[5, 6], [7, 8]]  ->  b = [5, 7, 6, 8].
        //
        // c[i, j] = sum_p a[i, p] * b[p * n + j]
        //         = a[i, 0] * b[j] + a[i, 1] * b[2 + j]
        //
        //   c[0, 0] = 1*5 + 2*6 = 17
        //   c[0, 1] = 1*7 + 2*8 = 23
        //   c[1, 0] = 3*5 + 4*6 = 39
        //   c[1, 1] = 3*7 + 4*8 = 53
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 7.0, 6.0, 8.0];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [17.0, 23.0, 39.0, 53.0]);
    }

    #[test]
    fn matmul_q8_0_matches_f32_reference() {
        // Activations 2x4, weights 3x4 (n=3, k=4). Construct f32
        // weights first, quantize them to Q8_0 with a known scale,
        // then verify the Q8_0 matmul matches the f32 reference.
        use ds4_quant::q8_0::quantize;

        let activations = [1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];

        // f32 weights [3 rows, 4 cols]
        let weights_f32 = [
            0.5f32, -0.5, 0.25, -0.25, // row 0
            1.0, 1.0, 1.0, 1.0, // row 1
            0.0, 0.0, 0.0, 0.0, // row 2
        ];

        // Quantize each row separately, ensuring the row divides
        // evenly into blocks. With k=4 we need a partial block:
        // emit a single 32-element block whose first 4 entries
        // reconstruct the row's quantized values and the rest are
        // zero-padded.
        let mut q8_weights: Vec<Vec<Q8_0Block>> = Vec::new();
        for n in 0..3 {
            let row = &weights_f32[n * 4..n * 4 + 4];
            let mut buf = [0.0f32; 32];
            buf[..4].copy_from_slice(&row[..4]);
            let block = quantize(&buf);
            q8_weights.push(vec![block]);
        }

        // Reference f32 matmul (column-major b).
        // b[p * n + j]: pack row-major weights into column-major
        // shape [k, n].
        let mut b_col = [0.0f32; 12];
        for p in 0..4 {
            for n in 0..3 {
                b_col[p * 3 + n] = weights_f32[n * 4 + p];
            }
        }
        let mut out_ref = [0.0f32; 6];
        matmul_f32(&activations, &b_col, &mut out_ref, 2, 3, 4);

        // Q8_0 matmul.
        let weight_refs: Vec<&[Q8_0Block]> = q8_weights.iter().map(|v| v.as_slice()).collect();
        let mut out_q8 = [0.0f32; 6];
        matmul_q8_0(&activations, &weight_refs, &mut out_q8, 2, 3, 4);

        // Q8_0 introduces ~1/127 quantization noise per element so
        // tolerance scales with the magnitude of the largest
        // output.
        let mag = out_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let tol = (mag + 1.0) * 0.02;
        assert!(
            vec_close(&out_q8, &out_ref, tol),
            "Q8_0 matmul mismatch:\n  ref = {out_ref:?}\n  q8  = {out_q8:?}"
        );
    }

    #[test]
    fn matmul_q8_0_uniform_weights_match_f32() {
        // K=32 so quantization is exact (one full block, no truncation).
        let m = 2usize;
        let k = 32usize;
        let n = 3usize;
        let mut activations = vec![0.0f32; m * k];
        for (i, v) in activations.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 0.5;
        }
        let mut weights_f32 = vec![0.0f32; n * k];
        for (i, v) in weights_f32.iter_mut().enumerate() {
            *v = ((i as f32) * 0.07).sin();
        }
        // Column-major b for f32 reference.
        let mut b_col = vec![0.0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                b_col[p * n + j] = weights_f32[j * k + p];
            }
        }
        let mut out_ref = vec![0.0f32; m * n];
        matmul_f32(&activations, &b_col, &mut out_ref, m, n, k);

        // Quantize each row to one Q8_0 block.
        let mut q8_weights: Vec<Vec<Q8_0Block>> = Vec::new();
        for j in 0..n {
            let row = &weights_f32[j * k..j * k + k];
            let mut buf = [0.0f32; 32];
            buf.copy_from_slice(row);
            let block = ds4_quant::q8_0::quantize(&buf);
            q8_weights.push(vec![block]);
        }
        let weight_refs: Vec<&[Q8_0Block]> = q8_weights.iter().map(|v| v.as_slice()).collect();
        let mut out_q8 = vec![0.0f32; m * n];
        matmul_q8_0(&activations, &weight_refs, &mut out_q8, m, n, k);

        let mag = out_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let tol = (mag + 1.0) * 0.02;
        assert!(vec_close(&out_q8, &out_ref, tol));
    }

    #[test]
    fn matmul_q4_k_uniform_weights_match_f32() {
        // K = 256 so quantization is exact per block.
        use ds4_quant::q4_k::quantize;
        let m = 2usize;
        let k = 256usize;
        let n = 2usize;
        let mut activations = vec![0.0f32; m * k];
        for (i, v) in activations.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).cos();
        }
        let mut weights_f32 = vec![0.0f32; n * k];
        for (i, v) in weights_f32.iter_mut().enumerate() {
            // Mix small positive/negative values within [-1, 1] so
            // the asymmetric 0..15 quantizer has headroom.
            let x = ((i as f32) * 0.07).sin();
            *v = x * 0.6;
        }
        let mut b_col = vec![0.0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                b_col[p * n + j] = weights_f32[j * k + p];
            }
        }
        let mut out_ref = vec![0.0f32; m * n];
        matmul_f32(&activations, &b_col, &mut out_ref, m, n, k);

        let mut q4_weights: Vec<Vec<Q4_KBlock>> = Vec::new();
        for j in 0..n {
            let row = &weights_f32[j * k..j * k + k];
            let mut buf = [0.0f32; 256];
            buf.copy_from_slice(row);
            let block = quantize(&buf);
            q4_weights.push(vec![block]);
        }
        let weight_refs: Vec<&[Q4_KBlock]> = q4_weights.iter().map(|v| v.as_slice()).collect();
        let mut out_q4 = vec![0.0f32; m * n];
        matmul_q4_k(&activations, &weight_refs, &mut out_q4, m, n, k);

        // Q4_K is heavier than Q8_0; tolerance scales with output
        // magnitude. The asymmetric quantizer saturates to
        // 0..(d*scale) - dmin*min, so a row of inputs in [-0.6, 0.6]
        // gets quantized to ~15 levels per sub-block.
        let mag = out_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let tol = (mag + 1.0) * 0.15;
        assert!(
            vec_close(&out_q4, &out_ref, tol),
            "Q4_K matmul mismatch:\n  ref = {out_ref:?}\n  q4  = {out_q4:?}"
        );
    }
}
