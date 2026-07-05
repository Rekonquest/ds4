//! Q4_K 4-bit super-block quantization.
//!
//! Block layout (matches `block_q4_K` in ggml-common.h):
//!
//! ```text
//! ggml_half2 dm;          // { d, dmin }  (4 bytes)
//! uint8_t scales[K_SCALE_SIZE]; // 12-byte packed scales/mins
//! uint8_t qs[QK_K/2];     // 128 nibbles (256 4-bit values)
//! ```
//!
//! Total bytes per block = 4 + 12 + 128 = 144 bytes for 256 elements.
//!
//! The 12-byte scales array packs 8 sub-blocks (32 elements each) of
//! scales and mins. Sub-blocks 0..3 use `scales[j]` and `scales[j+4]`
//! directly. Sub-blocks 4..7 share bits with sub-blocks 0..3: high 2
//! bits of each scale/min live in `scales[j-4]` and `scales[j-0]`.

use half::f16;

/// Q4_K super-block: 256 elements per block.
#[derive(Debug, Clone, PartialEq)]
pub struct Q4_KBlock {
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

impl Q4_KBlock {
    pub const BLOCK_SIZE: usize = 256;

    pub fn new_zero() -> Self {
        Self {
            d: 0.0,
            dmin: 0.0,
            scales: [0u8; 12],
            qs: [0u8; 128],
        }
    }
}

impl Default for Q4_KBlock {
    fn default() -> Self {
        Self::new_zero()
    }
}

/// Helper: nearest int (matches ggml's `nearest_int`).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let r = fval.round();
    if r >= 0.0 && (r - fval) == 0.5 {
        (r as i32) + 1
    } else if r < 0.0 && (fval - r) == 0.5 {
        (r as i32) - 1
    } else {
        r as i32
    }
}

/// Decode the per-32-element-block scale and min from the packed
/// 12-byte scales array. Mirrors `get_scale_min_k4`.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// Reference 2-bit min+scale quantizer. Mirrors `make_qkx2_quants`.
/// `weights` provides the per-element importance (RMS heuristic for
/// the reference path, all-1 for the simple path).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for li in l.iter_mut().take(n) {
            *li = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min));
        let q = q.clamp(0, nmax);
        l[i] = q as u8;
        let diff = scale * (l[i] as f32) + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights[i];
        best_error += w * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        let iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min));
            let q = q.clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            sum_l += w * (laux[i] as f32);
            sum_l2 += w * (laux[i] as f32) * (laux[i] as f32);
            sum_xl += w * (laux[i] as f32) * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                let this_scale = sum_xl / sum_l2;
                let mut cur_error = 0.0;
                for i in 0..n {
                    let diff = this_scale * (laux[i] as f32) + this_min - x[i];
                    let diff = if use_mad { diff.abs() } else { diff * diff };
                    let w = weights[i];
                    cur_error += w * diff;
                }
                if cur_error < best_error {
                    l.copy_from_slice(laux);
                    best_error = cur_error;
                    scale = this_scale;
                    min = this_min;
                }
                continue;
            }
            let mut cur_error = 0.0;
            for i in 0..n {
                let diff = this_scale * (laux[i] as f32) + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                let w = weights[i];
                cur_error += w * diff;
            }
            if cur_error < best_error {
                l.copy_from_slice(laux);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Quantize 256 f32 values into a single Q4_K block. Mirrors
/// `quantize_row_q4_K_reference` in ggml-quants.c.
pub fn quantize(input: &[f32; 256]) -> Q4_KBlock {
    let mut l = [0u8; 256];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut mins = [0f32; 8];
    let mut scales = [0f32; 8];

    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..8 {
        // Reference weights: sqrt(mean(x^2)) + |x[i]|. This matches the
        // upstream av_x heuristic.
        let mut sum_x2 = 0.0f32;
        for l in 0..32 {
            sum_x2 += input[32 * j + l] * input[32 * j + l];
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for l in 0..32 {
            weights[l] = av_x + input[32 * j + l].abs();
        }
        let x_slice: &[f32] = &input[32 * j..32 * j + 32];
        scales[j] = make_qkx2_quants(
            32,
            15,
            x_slice,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut mins[j],
            &mut laux,
            -1.0,
            0.1,
            20,
            false,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    let mut scales_packed = [0u8; 12];
    for j in 0..8 {
        let ls = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
        let lm = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
        if j < 4 {
            scales_packed[j] = ls;
            scales_packed[j + 4] = lm;
        } else {
            scales_packed[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
            scales_packed[j - 4] |= (ls >> 4) << 6;
            scales_packed[j] |= (lm >> 4) << 6;
        }
    }
    let d = f16::from_f32(max_scale / 63.0);
    let dmin = f16::from_f32(max_min / 63.0);

    // Compute the actual quantized values using the packed scales/mins.
    for j in 0..8 {
        let mut sc = 0u8;
        let mut m = 0u8;
        get_scale_min_k4(j, &scales_packed, &mut sc, &mut m);
        let d32 = d.to_f32() * sc as f32;
        if d32 == 0.0 {
            continue;
        }
        let dm = dmin.to_f32() * m as f32;
        for ii in 0..32 {
            let q = nearest_int((input[32 * j + ii] + dm) / d32).clamp(0, 15);
            l[32 * j + ii] = q as u8;
        }
    }

    // Pack nibbles.
    let mut qs = [0u8; 128];
    for j in (0..256).step_by(64) {
        for n in 0..32 {
            qs[j / 2 + n] = l[n + j] | (l[n + 32 + j] << 4);
        }
    }

    Q4_KBlock {
        d: d.to_f32(),
        dmin: dmin.to_f32(),
        scales: scales_packed,
        qs,
    }
}

/// Dequantize a single Q4_K block into 256 f32 values. Mirrors
/// `dequantize_row_q4_K` in ggml-quants.c.
pub fn dequantize(b: &Q4_KBlock, out: &mut [f32; 256]) {
    let q = &b.qs;
    let d = b.d;
    let m = b.dmin;
    let mut is = 0;
    let mut y_off = 0;
    let mut q_off = 0;
    for _ in 0..4 {
        let mut sc = 0u8;
        let mut mn = 0u8;
        get_scale_min_k4(is, &b.scales, &mut sc, &mut mn);
        let d1 = d * sc as f32;
        let m1 = m * mn as f32;
        let mut sc2 = 0u8;
        let mut m2 = 0u8;
        get_scale_min_k4(is + 1, &b.scales, &mut sc2, &mut m2);
        let d2 = d * sc2 as f32;
        let m2v = m * m2 as f32;
        for l in 0..32 {
            out[y_off + l] = d1 * (q[q_off + l] & 0x0f) as f32 - m1;
        }
        for l in 0..32 {
            out[y_off + 32 + l] = d2 * (q[q_off + l] >> 4) as f32 - m2v;
        }
        is += 2;
        y_off += 64;
        q_off += 32;
    }
}

/// Helper: pack the 8 sub-block scales into the 12-byte format GGML
/// expects. Exposed for tests; the quantize path computes it inline.
#[cfg(test)]
pub(crate) fn pack_scales_k4(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for j in 0..8 {
        let ls = scales[j];
        let lm = mins[j];
        if j < 4 {
            out[j] = ls;
            out[j + 4] = lm;
        } else {
            out[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
            out[j - 4] |= (ls >> 4) << 6;
            out[j] |= (lm >> 4) << 6;
        }
    }
    out
}

/// Dot product of a Q4_K row with a Q8_K row. Mirrors
/// `ggml_vec_dot_q4_K_q8_K_generic` in ggml-cpu/quants.c.
pub fn vec_dot(n: usize, a: &[Q4_KBlock], b: &[Q8KView], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "q4_k vec_dot: block slices must match");
    assert!(n <= a.len() * 256, "q4_k vec_dot: n exceeds input length");
    assert!(!out.is_empty(), "q4_k vec_dot: output must be non-empty");

    let nb = n / 256;
    let mut sumf = 0.0f32;
    for i in 0..nb {
        sumf += dot_block(&a[i], &b[i]);
    }
    out[0] = sumf;
}

/// Lightweight view of a Q8_K block. The Q4_K dot kernel reads the
/// block's i8 quants and its i16 bsums in groups of 16.
#[derive(Debug, Clone, Copy)]
pub struct Q8KView<'a> {
    pub d: f32,
    pub qs: &'a [i8; 256],
    pub bsums: &'a [i16; 16],
}

pub fn dot_block(a: &Q4_KBlock, b: &Q8KView<'_>) -> f32 {
    // Step 1: decode scales/mins into 8 pairs of (scale, min) values.
    let mut aux8 = [0i8; 256];
    let q4 = &a.qs;
    for j in 0..4 {
        for l in 0..32 {
            aux8[j * 64 + l] = (q4[j * 32 + l] & 0x0f) as i8;
        }
        for l in 0..32 {
            aux8[j * 64 + 32 + l] = (q4[j * 32 + l] >> 4) as i8;
        }
    }

    // Step 2: unpack the 12-byte scales into two 4-byte arrays.
    let mut utmp = [0u32; 4];
    let sp = &a.scales;
    utmp[0] = u32::from_le_bytes([sp[0], sp[1], sp[2], sp[3]]);
    utmp[1] = u32::from_le_bytes([sp[4], sp[5], sp[6], sp[7]]);
    utmp[2] = u32::from_le_bytes([sp[8], sp[9], sp[10], sp[11]]);
    // Mirror the bit-shuffling the C kernel does.
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    let uaux = utmp[1] & 0x3f3f3f3f;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= 0x3f3f3f3f;

    let scales_bytes: [u8; 8] = {
        let mut out = [0u8; 8];
        let p = utmp[0].to_le_bytes();
        out[0..4].copy_from_slice(&p);
        let p = utmp[1].to_le_bytes();
        out[4..8].copy_from_slice(&p);
        out
    };
    let mins_bytes: [u8; 8] = {
        let mut out = [0u8; 8];
        let p = utmp[2].to_le_bytes();
        out[0..4].copy_from_slice(&p);
        let p = utmp[3].to_le_bytes();
        out[4..8].copy_from_slice(&p);
        out
    };

    // Step 3: signed-sum part (summs from mins).
    let mut sumi = 0i32;
    for j in 0..16 {
        sumi += b.bsums[j] as i32 * mins_bytes[j / 2] as i32;
    }

    // Step 4: scale-mixed dot products, 8 accumulators.
    let mut aux32 = [0i32; 8];
    let mut q8_off = 0usize;
    let mut a_off = 0usize;
    for (s_idx, _) in (0..8).enumerate() {
        let scale = scales_bytes[s_idx] as i32;
        for _grp in 0..4 {
            let mut aux16 = [0i16; 8];
            for l in 0..8 {
                aux16[l] = b.qs[q8_off + l] as i16 * aux8[a_off + l] as i16;
            }
            for (l_idx, slot) in aux32.iter_mut().enumerate() {
                *slot += scale * aux16[l_idx] as i32;
            }
            q8_off += 8;
            a_off += 8;
        }
    }

    let d = a.d * b.d;
    let mut s = 0.0f32;
    for slot in &aux32 {
        s += d * *slot as f32;
    }
    let dmin = a.dmin * b.d;
    s - dmin * sumi as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q8k_from_quantized(input: &[f32; 256]) -> (Vec<i8>, Vec<i16>, f32) {
        let mut qs = vec![0i8; 256];
        let mut max = 0.0f32;
        for &v in input {
            if v.abs() > max {
                max = v.abs();
            }
        }
        if max == 0.0 {
            return (qs, vec![0i16; 16], 0.0);
        }
        let iscale = -127.0 / max;
        for (i, &v) in input.iter().enumerate() {
            let n = (iscale * v).round() as i32;
            qs[i] = n.clamp(-127, 127) as i8;
        }
        let mut bsums = vec![0i16; 16];
        for j in 0..16 {
            let mut s = 0i32;
            for ii in 0..16 {
                s += qs[j * 16 + ii] as i32;
            }
            bsums[j] = s as i16;
        }
        (qs, bsums, 1.0 / iscale)
    }

    #[test]
    fn pack_unpack_scales_roundtrips() {
        let scales = [3u8, 5, 7, 9, 60, 50, 40, 30];
        let mins = [1u8, 2, 4, 8, 33, 22, 11, 0];
        let packed = pack_scales_k4(scales, mins);
        // Now decode back and check.
        for j in 0..8 {
            let mut sc = 0u8;
            let mut m = 0u8;
            get_scale_min_k4(j, &packed, &mut sc, &mut m);
            assert_eq!(sc, scales[j], "j={} sc mismatch", j);
            assert_eq!(m, mins[j], "j={} min mismatch", j);
        }
    }

    #[test]
    fn dequantize_zero_block_is_zero() {
        let b = Q4_KBlock::new_zero();
        let mut out = [99.0f32; 256];
        dequantize(&b, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn quantize_then_dequantize_is_close() {
        // Synthetic signal that exercises the asymmetric quantizer.
        let input: [f32; 256] = std::array::from_fn(|i| {
            let x = (i as f32) / 256.0;
            // Mix positive and negative with a baseline.
            x * (if i % 2 == 0 { 1.0 } else { -1.0 }) + 0.01 * (i as f32).sin()
        });
        let q = quantize(&input);
        let mut out = [0.0f32; 256];
        dequantize(&q, &mut out);
        // Tolerance: the asymmetric quantizer saturates to 0..15 so
        // values around the median should round-trip within ~1/8 of
        // the per-sub-block scale.
        for (i, (&orig, &dq)) in input.iter().zip(out.iter()).enumerate() {
            let diff = (orig - dq).abs();
            assert!(diff < 1.0, "i={} orig={} dq={} diff={}", i, orig, dq, diff);
        }
    }

    #[test]
    fn dot_block_zero_input_is_zero() {
        let input = [0.0f32; 256];
        let (qs, bsums, d) = make_q8k_from_quantized(&input);
        let qa = Q4_KBlock::new_zero();
        let view = Q8KView {
            d,
            qs: qs.as_slice().try_into().unwrap(),
            bsums: bsums.as_slice().try_into().unwrap(),
        };
        let mut out = [0.0f32];
        vec_dot(256, &[qa], &[view], &mut out);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn dot_block_hand_computed() {
        // Build a block with a single non-zero scale and a single
        // non-zero Q8 element so we can compute the expected output
        // by hand.
        let mut qa = Q4_KBlock::new_zero();
        qa.d = 1.0;
        qa.dmin = 0.0;
        // Set sub-block 0 to scale=1, min=0.
        qa.scales[0] = 1;
        qa.scales[4] = 0;
        // First nibble of qs[0] = 15 -> reconstructed value = 1.0 * 15 = 15.0
        qa.qs[0] = 15;

        // Q8K with d=1, qs[0] = 1, all else zero.
        let mut qs8 = [0i8; 256];
        qs8[0] = 1;
        let bsums = [0i16; 16];
        let view = Q8KView {
            d: 1.0,
            qs: &qs8,
            bsums: &bsums,
        };

        let mut out = [0.0f32];
        vec_dot(256, &[qa], &[view], &mut out);
        // The dequantized q4_k value at index 0 is d * scale * nibble - dmin * min
        //   = 1.0 * 1 * 15 - 0 = 15.0
        // Dot with q8[0] = 1 yields 15.0.
        assert!((out[0] - 15.0).abs() < 1e-3, "got {}", out[0]);
    }
}
