//! Q2_K 2-bit super-block quantization.
//!
//! Block layout (matches `block_q2_K` in ggml-common.h):
//!
//! ```text
//! uint8_t scales[QK_K/16];  // 16 bytes, each packs one 4-bit scale
//!                           // (low nibble) and one 4-bit min
//!                           // (high nibble) for a 16-element sub-block
//! uint8_t qs[QK_K/4];       // 64 bytes, each holds four 2-bit values
//! ggml_half2 dm;            // { d, dmin }
//! ```
//!
//! Total bytes per block = 16 + 64 + 4 = 84 bytes for 256 elements.
//!
//! Weight reconstruction: `x = d * (sc & 0xF) * q - dmin * (sc >> 4)`.

use half::f16;

use crate::q4_k::Q8KView;

/// Q2_K super-block: 256 elements per block.
#[derive(Debug, Clone, PartialEq)]
pub struct Q2_KBlock {
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 16],
    pub qs: [u8; 64],
}

impl Q2_KBlock {
    pub const BLOCK_SIZE: usize = 256;

    pub fn new_zero() -> Self {
        Self {
            d: 0.0,
            dmin: 0.0,
            scales: [0u8; 16],
            qs: [0u8; 64],
        }
    }
}

impl Default for Q2_KBlock {
    fn default() -> Self {
        Self::new_zero()
    }
}

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

/// Reference 2-bit min+scale quantizer (see ggml-quants.c
/// `make_qkx2_quants`).
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

/// Quantize 256 f32 values into a Q2_K block. Mirrors
/// `quantize_row_q2_K_ref` in ggml-quants.c.
pub fn quantize(input: &[f32; 256]) -> Q2_KBlock {
    let mut l = [0u8; 256];
    let mut laux = [0u8; 16];
    let mut weights = [0f32; 16];
    let mut mins = [0f32; 16];
    let mut scales = [0f32; 16];

    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..16 {
        for l in 0..16 {
            weights[l] = input[16 * j + l].abs();
        }
        let x_slice: &[f32] = &input[16 * j..16 * j + 16];
        scales[j] = make_qkx2_quants(
            16,
            3,
            x_slice,
            &weights,
            &mut l[16 * j..16 * j + 16],
            &mut mins[j],
            &mut laux,
            -0.5,
            0.1,
            15,
            true,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }

    let q4scale = 15.0f32;
    let mut scales_packed = [0u8; 16];
    let d = if max_scale > 0.0 {
        let iscale = q4scale / max_scale;
        for j in 0..16 {
            let lq = nearest_int(iscale * scales[j]);
            scales_packed[j] = lq.clamp(0, 15) as u8;
        }
        f16::from_f32(max_scale / q4scale)
    } else {
        f16::from_f32(0.0)
    };
    let dmin = if max_min > 0.0 {
        let iscale = q4scale / max_min;
        for j in 0..16 {
            let lq = nearest_int(iscale * mins[j]);
            scales_packed[j] |= (lq.clamp(0, 15) as u8) << 4;
        }
        f16::from_f32(max_min / q4scale)
    } else {
        f16::from_f32(0.0)
    };

    // Compute the actual quantized values using the packed scales/mins.
    for j in 0..16 {
        let sd = d.to_f32() * (scales_packed[j] & 0x0f) as f32;
        if sd == 0.0 {
            continue;
        }
        let sdm = dmin.to_f32() * (scales_packed[j] >> 4) as f32;
        for ii in 0..16 {
            let q = nearest_int((input[16 * j + ii] + sdm) / sd).clamp(0, 3);
            l[16 * j + ii] = q as u8;
        }
    }

    // Pack 2-bit quants into bytes (4 values per byte).
    let mut qs = [0u8; 64];
    for j in (0..256).step_by(128) {
        for n in 0..32 {
            qs[j / 4 + n] =
                l[n + j] | (l[n + 32 + j] << 2) | (l[n + 64 + j] << 4) | (l[n + 96 + j] << 6);
        }
    }

    Q2_KBlock {
        d: d.to_f32(),
        dmin: dmin.to_f32(),
        scales: scales_packed,
        qs,
    }
}

/// Dequantize a single Q2_K block into 256 f32 values. Mirrors
/// `dequantize_row_q2_K` in ggml-quants.c.
pub fn dequantize(b: &Q2_KBlock, out: &mut [f32; 256]) {
    let d = b.d;
    let min = b.dmin;
    let q = &b.qs;
    let mut is = 0;
    let mut y_off = 0;
    let mut q_off = 0;
    for _ in 0..2 {
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = b.scales[is];
            let dl = d * (sc & 0x0f) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                out[y_off + l] = dl * (((q[q_off + l] >> shift) & 3) as i8 as f32) - ml;
            }
            is += 1;
            let sc2 = b.scales[is];
            let dl2 = d * (sc2 & 0x0f) as f32;
            let ml2 = min * (sc2 >> 4) as f32;
            for l in 16..32 {
                out[y_off + l] = dl2 * (((q[q_off + l] >> shift) & 3) as i8 as f32) - ml2;
            }
            is += 1;
            shift += 2;
            y_off += 32;
        }
        q_off += 32;
    }
}

/// Dot product of a Q2_K row with a Q8_K row. Mirrors
/// `ggml_vec_dot_q2_K_q8_K_generic` in ggml-cpu/quants.c.
pub fn vec_dot(n: usize, a: &[Q2_KBlock], b: &[Q8KView<'_>], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "q2_k vec_dot: block slices must match");
    assert!(n <= a.len() * 256, "q2_k vec_dot: n exceeds input length");
    assert!(!out.is_empty(), "q2_k vec_dot: output must be non-empty");

    let nb = n / 256;
    let mut sumf = 0.0f32;
    for i in 0..nb {
        sumf += dot_block(&a[i], &b[i]);
    }
    out[0] = sumf;
}

pub fn dot_block(a: &Q2_KBlock, b: &Q8KView<'_>) -> f32 {
    let q2 = &a.qs;
    let sc = &a.scales;

    let mut summs = 0i32;
    for (j, &bs) in b.bsums.iter().enumerate() {
        summs += bs as i32 * (sc[j] >> 4) as i32;
    }

    let dall = b.d * a.d;
    let dmin = b.d * a.dmin;

    // The C kernel walks `q` through 64 bytes (QK_K/4) in two chunks
    // of 32 bytes each. We use a single byte index.
    let mut isum = 0i32;
    let mut is = 0;
    let mut q8_off = 0;
    let mut q_off = 0;
    for _ in 0..2 {
        let mut shift = 0u32;
        for _ in 0..4 {
            let d_sc = (sc[is] & 0x0f) as i32;
            let mut isuml = 0i32;
            for l in 0..16 {
                let q = ((q2[q_off + l] >> shift) & 3) as i32;
                isuml += b.qs[q8_off + l] as i32 * q;
            }
            isum += d_sc * isuml;
            is += 1;

            let d_sc2 = (sc[is] & 0x0f) as i32;
            let mut isuml2 = 0i32;
            for l in 16..32 {
                let q = ((q2[q_off + l] >> shift) & 3) as i32;
                isuml2 += b.qs[q8_off + l] as i32 * q;
            }
            isum += d_sc2 * isuml2;
            is += 1;
            shift += 2;
            q8_off += 32;
        }
        q_off += 32;
    }
    dall * isum as f32 - dmin * summs as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4_k::Q8KView;

    fn make_q8k(input: &[f32; 256]) -> (Vec<i8>, Vec<i16>, f32) {
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
    fn dequantize_zero_block_is_zero() {
        let b = Q2_KBlock::new_zero();
        let mut out = [99.0f32; 256];
        dequantize(&b, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn quantize_then_dequantize_is_close() {
        // Synthetic signal with both positive and negative values.
        let input: [f32; 256] = std::array::from_fn(|i| {
            let x = (i as f32) / 128.0 - 1.0;
            x * 0.5
        });
        let q = quantize(&input);
        let mut out = [0.0f32; 256];
        dequantize(&q, &mut out);
        for (i, (&orig, &dq)) in input.iter().zip(out.iter()).enumerate() {
            let diff = (orig - dq).abs();
            assert!(diff < 1.5, "i={} orig={} dq={} diff={}", i, orig, dq, diff);
        }
    }

    #[test]
    fn dot_block_zero_is_zero() {
        let input = [0.0f32; 256];
        let (qs, bsums, d) = make_q8k(&input);
        let qa = Q2_KBlock::new_zero();
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
        // Build a Q2_K block with a single non-zero scale and q value,
        // and a Q8_K with a matching non-zero activation.
        let mut qa = Q2_KBlock::new_zero();
        qa.d = 1.0;
        qa.dmin = 0.0;
        // Sub-block 0: scale=1 (low nibble), min=0 (high nibble).
        qa.scales[0] = 1;
        // First 2-bit field of qs[0] = 3 (max). With d=1 and scale=1
        // the dequantized value is 1*1*3 - 0 = 3.0.
        qa.qs[0] = 3;

        let mut qs8 = [0i8; 256];
        qs8[0] = 5;
        let bsums = [0i16; 16];
        let view = Q8KView {
            d: 1.0,
            qs: &qs8,
            bsums: &bsums,
        };

        let mut out = [0.0f32];
        vec_dot(256, &[qa], &[view], &mut out);
        // 3.0 * 5 = 15.0
        assert!((out[0] - 15.0).abs() < 1e-3, "got {}", out[0]);
    }
}
