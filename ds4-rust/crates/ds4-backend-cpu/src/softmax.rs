// DS4 (DwarfStar) — numerically stable softmax.
//
// Both kernels subtract the max before exponentiation, so the
// largest `exp` argument is 0 and the smaller ones are
// non-positive — no overflow risk, no catastrophic cancellation
// in the normalization step.

/// Numerically stable softmax: `x' = exp(x - max(x)) / sum(...)`,
/// applied in-place.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    // 1. max
    let mut max_val = x[0];
    for &v in &x[1..] {
        if v > max_val {
            max_val = v;
        }
    }
    // 2. exp(x - max), accumulating the sum
    let mut sum = 0.0f32;
    for slot in x.iter_mut() {
        let e = (*slot - max_val).exp();
        *slot = e;
        sum += e;
    }
    // 3. normalize
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for slot in x.iter_mut() {
        *slot *= inv;
    }
}

/// Numerically stable softmax with a temperature. The input is
/// divided by `temperature` before exponentiating; `temperature`
/// values in (0, 1] sharpen the distribution, > 1 softens it.
pub fn softmax_with_temperature(x: &mut [f32], temperature: f32) {
    assert!(
        temperature > 0.0 && temperature.is_finite(),
        "softmax_with_temperature: temperature must be > 0 and finite (got {temperature})"
    );
    if x.is_empty() {
        return;
    }
    let inv_t = 1.0f32 / temperature;
    let mut max_val = x[0] * inv_t;
    for &v in &x[1..] {
        let v = v * inv_t;
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for slot in x.iter_mut() {
        let e = (*slot * inv_t - max_val).exp();
        *slot = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for slot in x.iter_mut() {
        *slot *= inv;
    }
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
    fn softmax_subtract_max_pattern() {
        // [1, 2, 3] -> subtract max(3) -> [-2, -1, 0] -> exp -> [e^-2, e^-1, 1]
        // -> normalize by s = e^-2 + e^-1 + 1. The largest output
        // belongs to the largest input (3), as expected.
        let mut x = [1.0f32, 2.0, 3.0];
        softmax(&mut x);
        let s = (-2.0f32).exp() + (-1.0f32).exp() + 1.0;
        let expected = [(-2.0f32).exp() / s, (-1.0f32).exp() / s, 1.0 / s];
        assert!(
            vec_close(&x, &expected, 1e-6),
            "got {x:?} expected {expected:?}"
        );
        // Sum must be 1.
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_uniform_input_is_uniform() {
        // All equal inputs => all equal outputs.
        let mut x = [4.0f32, 4.0, 4.0, 4.0];
        softmax(&mut x);
        for &v in &x {
            assert!(approx_eq(v, 0.25, 1e-6));
        }
    }

    #[test]
    fn softmax_large_inputs_no_overflow() {
        // Input values near 1000 must not overflow: subtracting the
        // max before exp() keeps the largest argument at 0.
        let mut x = [1000.0f32, 1001.0, 1002.0];
        softmax(&mut x);
        for &v in &x {
            assert!(v.is_finite(), "softmax produced non-finite value: {v}");
            assert!((0.0..=1.0).contains(&v), "softmax produced bad value: {v}");
        }
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_temperature_low_sharpens() {
        // With temperature=0.25 the effective logits are 4x larger,
        // which sharpens the distribution. argmax must still
        // dominate.
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_with_temperature(&mut x, 0.25);
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_temperature_high_softens() {
        // With temperature=4.0 the logits are 0.25x larger; the
        // output distribution should be closer to uniform than
        // the default-temperature softmax.
        let x_default = {
            let mut v = [1.0f32, 2.0, 3.0];
            softmax(&mut v);
            v
        };
        let mut x_high = [1.0f32, 2.0, 3.0];
        softmax_with_temperature(&mut x_high, 4.0);
        let default_entropy: f32 = x_default.iter().map(|p| -p * p.ln()).sum();
        let high_entropy: f32 = x_high.iter().map(|p| -p * p.ln()).sum();
        assert!(
            high_entropy > default_entropy,
            "high-T entropy ({high_entropy}) should exceed default-T entropy ({default_entropy})"
        );
    }

    #[test]
    fn softmax_temperature_one_matches_default() {
        let mut a = [1.0f32, 2.0, 3.0];
        let mut b = [1.0f32, 2.0, 3.0];
        softmax(&mut a);
        softmax_with_temperature(&mut b, 1.0);
        assert!(vec_close(&a, &b, 1e-6));
    }
}
