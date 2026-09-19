//! The matrix exponential and the adjoint of its Fréchet derivative.

use super::mat::{CMat, solve};
use crate::autodiff::{C, Scalar};
use crate::error::{Error, Result};

/// Padé-13 coefficients (Higham, "The scaling and squaring method for the matrix exponential revisited", 2005).
const PADE13: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// Largest 1-norm for which Padé-13 needs no scaling.
const THETA13: f64 = 5.371_920_351_148_152;

/// `exp(A)` by Padé-13 scaling and squaring.
///
/// The scaling exponent is chosen from the value parts alone, so a `Dual` evaluation follows exactly the branch the
/// `f64` one does and its derivative is the derivative of the same algorithm.
pub fn expm<T: Scalar>(a: &CMat<T>) -> Result<CMat<T>> {
    if !a.is_square() {
        return Err(Error::Dimension(format!("expm of a {}x{} matrix", a.rows, a.cols)));
    }
    let norm = a.norm1();
    if !norm.is_finite() {
        return Err(Error::Numerical("expm of a matrix with non-finite entries".into()));
    }
    let n = a.rows;
    if n == 0 {
        return Ok(a.clone());
    }
    let s = if norm > THETA13 {
        (norm / THETA13).log2().ceil().max(0.0) as i32
    } else {
        0
    };
    let a = a.scale_re(T::from_f64(0.5f64.powi(s)));

    let id = CMat::<T>::identity(n);
    let a2 = a.matmul(&a)?;
    let a4 = a2.matmul(&a2)?;
    let a6 = a4.matmul(&a2)?;
    let b = |i: usize| T::from_f64(PADE13[i]);

    let mut inner_u = a6.scale_re(b(13));
    inner_u.axpy_re(b(11), &a4);
    inner_u.axpy_re(b(9), &a2);
    let mut u = a6.matmul(&inner_u)?;
    u.axpy_re(b(7), &a6);
    u.axpy_re(b(5), &a4);
    u.axpy_re(b(3), &a2);
    u.axpy_re(b(1), &id);
    let u = a.matmul(&u)?;

    let mut inner_v = a6.scale_re(b(12));
    inner_v.axpy_re(b(10), &a4);
    inner_v.axpy_re(b(8), &a2);
    let mut v = a6.matmul(&inner_v)?;
    v.axpy_re(b(6), &a6);
    v.axpy_re(b(4), &a4);
    v.axpy_re(b(2), &a2);
    v.axpy_re(b(0), &id);

    let mut r = solve(&v.sub(&u)?, &v.add(&u)?)?;
    for _ in 0..s {
        r = r.matmul(&r)?;
    }
    Ok(r)
}

/// `exp(−i·dt·H)` for a Hermitian `H`.
///
/// Two-level systems use the closed form: with `c = (H₀₀ + H₁₁)/2` and the traceless part `K = H − c·I`,
/// `K² = ω²·I` where `ω² = ((H₀₀ − H₁₁)/2)² + H₀₁·H₁₀`, so
/// `exp(−i·dt·H) = exp(−i·c·dt)·[cos(ω·dt)·I − i·dt·sinc(ω·dt)·K]`.
/// Larger systems use [`expm`].
pub fn expm_mi_dt<T: Scalar>(h: &CMat<T>, dt: T) -> Result<CMat<T>> {
    if h.rows == 2 && h.cols == 2 {
        let zero = T::zero();
        let two = T::from_f64(2.0);
        let (h00, h01, h10, h11) = (h.get(0, 0), h.get(0, 1), h.get(1, 0), h.get(1, 1));
        let c = (h00.re + h11.re) / two;
        let h3 = (h00.re - h11.re) / two;
        let offdiag = h01 * h10;
        let omega2 = h3 * h3 + offdiag.re;
        let omega = if omega2.re() > 0.0 { omega2.sqrt() } else { zero };
        let x = omega * dt;
        let cos = C::new(x.cos(), zero);
        // −i·dt·sinc(ω·dt)
        let s = C::new(zero, -(dt * x.sinc()));
        let phase = C::new((c * dt).cos(), -(c * dt).sin());
        let k00 = C::new(h3, h00.im);
        let k11 = C::new(-h3, h11.im);
        let data = vec![
            phase * (cos + s * k00),
            phase * (s * h01),
            phase * (s * h10),
            phase * (cos + s * k11),
        ];
        return CMat::from_vec(2, 2, data);
    }
    let minus_i_dt = C::new(T::zero(), -dt);
    expm(&h.scale(minus_i_dt))
}

/// The adjoint of the Fréchet derivative of `exp` at `A`, applied to `G`: `L_exp(Aᴴ, G)`.
///
/// If `U = exp(A)` and `G = ∂L/∂U` in the conjugate convention, this is `∂L/∂A`.  It is the upper-right block of
/// `exp([[Aᴴ, G], [0, Aᴴ]])`.
pub fn expm_frechet_adjoint<T: Scalar>(a: &CMat<T>, g: &CMat<T>) -> Result<CMat<T>> {
    let n = a.rows;
    if !a.is_square() || g.shape() != a.shape() {
        return Err(Error::Dimension(format!(
            "Fréchet adjoint of a {}x{} exponential in a {}x{} direction",
            a.rows, a.cols, g.rows, g.cols
        )));
    }
    let ah = a.adjoint();
    let mut block = CMat::<T>::zeros(2 * n, 2 * n);
    for r in 0..n {
        for c in 0..n {
            let v = ah.get(r, c);
            block.set(r, c, v);
            block.set(n + r, n + c, v);
            block.set(r, n + c, g.get(r, c));
        }
    }
    let e = expm(&block)?;
    Ok(CMat::from_fn(n, n, |r, c| e.get(r, n + c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Mat;

    /// A deterministic pseudo-random complex matrix scaled to 1-norm `norm`.
    fn random(n: usize, seed: u64, norm: f64) -> CMat<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let m = Mat::from_fn(n, n, |_, _| C::new(next(), next()));
        let scale = norm / m.norm1();
        m.scale_re(scale)
    }

    fn hermitian(n: usize, seed: u64, norm: f64) -> CMat<f64> {
        let m = random(n, seed, norm);
        m.add(&m.adjoint()).unwrap().scale_re(0.5)
    }

    /// Taylor series with squaring, as an independent reference.
    fn taylor(a: &CMat<f64>) -> CMat<f64> {
        let s = (a.norm1().log2().ceil().max(0.0) as i32) + 4;
        let a = a.scale_re(0.5f64.powi(s));
        let mut term = CMat::identity(a.rows);
        let mut sum = term.clone();
        for k in 1..40 {
            term = term.matmul(&a).unwrap().scale_re(1.0 / k as f64);
            sum = sum.add(&term).unwrap();
        }
        for _ in 0..s {
            sum = sum.matmul(&sum).unwrap();
        }
        sum
    }

    #[test]
    fn pade_matches_taylor() {
        for (n, seed) in [(2, 1), (4, 2), (8, 3), (27, 4)] {
            for norm in [0.1, 1.0, 7.0, 50.0] {
                let a = random(n, seed, norm);
                let got = expm(&a).unwrap();
                let want = taylor(&a);
                let scale = want.data.iter().map(|z| z.norm()).fold(0.0, f64::max);
                assert!(got.max_abs_diff(&want) <= 1e-11 * scale, "n={n} norm={norm}");
            }
        }
    }

    #[test]
    fn closed_form_matches_pade_and_is_unitary() {
        for seed in 0..20 {
            let h = hermitian(2, seed, 3.0 + seed as f64);
            let dt = 0.37;
            let closed = expm_mi_dt(&h, dt).unwrap();
            let pade = expm(&h.scale(C::new(0.0, -dt))).unwrap();
            assert!(closed.max_abs_diff(&pade) < 1e-13, "seed {seed}");
            let uu = closed.matmul(&closed.adjoint()).unwrap();
            assert!(uu.max_abs_diff(&CMat::identity(2)) < 1e-13);
        }
        // Zero Hamiltonian: the identity, no division by zero.
        let z = expm_mi_dt(&CMat::<f64>::zeros(2, 2), 1.0).unwrap();
        assert!(z.max_abs_diff(&CMat::identity(2)) < 1e-15);
    }

    #[test]
    fn frechet_adjoint_matches_finite_differences() {
        for (n, seed) in [(2, 5), (3, 6), (4, 7)] {
            let a = random(n, seed, 2.0);
            let g = random(n, seed + 100, 1.0);
            let e = random(n, seed + 200, 1.0);
            let adj = expm_frechet_adjoint(&a, &g).unwrap();
            let h = 1e-6;
            let at = |s: f64| {
                let mut m = a.clone();
                m.axpy_re(s, &e);
                g.inner(&expm(&m).unwrap()).re
            };
            let fd = (at(h) - at(-h)) / (2.0 * h);
            let an = adj.inner(&e).re;
            assert!(
                (fd - an).abs() < 1e-7 * an.abs().max(1.0),
                "n={n}: fd {fd} vs adjoint {an}"
            );
        }
    }

    #[test]
    fn dual_expm_carries_the_derivative() {
        use crate::autodiff::Dual;
        let h = hermitian(4, 9, 3.0);
        let e = hermitian(4, 10, 1.0);
        let dt = 0.2;
        // d/ds exp(-i dt (H + s E)) at s = 0 via duals, against finite differences.
        let hd: CMat<Dual> = Mat::from_fn(4, 4, |r, c| {
            let (a, b) = (h.get(r, c), e.get(r, c));
            C::new(Dual::new(a.re, b.re), Dual::new(a.im, b.im))
        });
        let ud = expm_mi_dt(&hd, Dual::constant(dt)).unwrap();
        let step = 1e-6;
        let at = |s: f64| {
            let mut m = h.clone();
            m.axpy_re(s, &e);
            expm_mi_dt(&m, dt).unwrap()
        };
        let (plus, minus) = (at(step), at(-step));
        for i in 0..16 {
            let fd = (plus.data[i] - minus.data[i]) / (2.0 * step);
            let ad = C::new(ud.data[i].re.eps, ud.data[i].im.eps);
            assert!((fd - ad).norm() < 1e-7, "entry {i}");
        }
        // Same through the 2x2 closed form.
        let h2 = hermitian(2, 11, 4.0);
        let e2 = hermitian(2, 12, 1.0);
        let hd2: CMat<Dual> = Mat::from_fn(2, 2, |r, c| {
            let (a, b) = (h2.get(r, c), e2.get(r, c));
            C::new(Dual::new(a.re, b.re), Dual::new(a.im, b.im))
        });
        let ud2 = expm_mi_dt(&hd2, Dual::constant(dt)).unwrap();
        let at2 = |s: f64| {
            let mut m = h2.clone();
            m.axpy_re(s, &e2);
            expm_mi_dt(&m, dt).unwrap()
        };
        let (p2, m2) = (at2(step), at2(-step));
        for i in 0..4 {
            let fd = (p2.data[i] - m2.data[i]) / (2.0 * step);
            let ad = C::new(ud2.data[i].re.eps, ud2.data[i].im.eps);
            assert!((fd - ad).norm() < 1e-7, "2x2 entry {i}");
        }
    }
}
