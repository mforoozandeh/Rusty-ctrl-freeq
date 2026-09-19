//! The polynomial and polynomial-like bases.

use rand::RngExt;

use super::Basis;
use crate::linalg::RMat;
use crate::setup::Rng;

/// Fill an `x.len() × ncols` matrix column by column with a three-term recurrence
/// `p_i = a(i, x)·p_{i−1} − b(i)·p_{i−2}`, starting from `p_0 = 1` and `p_1 = first(x)`.
fn recurrence(
    x: &[f64],
    ncols: usize,
    first: impl Fn(f64) -> f64,
    step: impl Fn(usize, f64, f64, f64) -> f64,
) -> RMat<f64> {
    let mut m = RMat::<f64>::zeros(x.len(), ncols);
    for (r, &xv) in x.iter().enumerate() {
        if ncols > 0 {
            m.set(r, 0, 1.0);
        }
        if ncols > 1 {
            m.set(r, 1, first(xv));
        }
        for i in 2..ncols {
            let v = step(i, xv, m.get(r, i - 1), m.get(r, i - 2));
            m.set(r, i, v);
        }
    }
    m
}

/// Chebyshev polynomials of the first kind, `T_i(x)`.
pub(super) struct Chebyshev;

impl Basis for Chebyshev {
    fn name(&self) -> &'static str {
        "cheb"
    }
    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        recurrence(x, ncols, |x| x, |_, x, p1, p2| 2.0 * x * p1 - p2)
    }
}

/// Legendre polynomials `P_i(x)`.
pub(super) struct Legendre;

impl Basis for Legendre {
    fn name(&self) -> &'static str {
        "leg"
    }
    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        recurrence(
            x,
            ncols,
            |x| x,
            |i, x, p1, p2| {
                let i = i as f64;
                ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p2) / i
            },
        )
    }
}

/// Gegenbauer polynomials `C_i^(λ)(x)`.
pub(super) struct Gegenbauer {
    pub(super) lambda: f64,
}

impl Basis for Gegenbauer {
    fn name(&self) -> &'static str {
        "gegen"
    }
    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        let l = self.lambda;
        recurrence(
            x,
            ncols,
            |x| 2.0 * l * x,
            |i, x, p1, p2| {
                let i = i as f64;
                (2.0 * (i + l - 1.0) * x * p1 - (i + 2.0 * l - 2.0) * p2) / i
            },
        )
    }
}

/// Probabilists' Hermite polynomials `He_i(x)`.
pub(super) struct Hermite;

impl Basis for Hermite {
    fn name(&self) -> &'static str {
        "hermite"
    }
    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        recurrence(x, ncols, |x| x, |i, x, p1, p2| x * p1 - (i - 1) as f64 * p2)
    }
}

/// Monomials `x^i`.
pub(super) struct Monomial;

impl Basis for Monomial {
    fn name(&self) -> &'static str {
        "poly"
    }
    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        RMat::from_fn(x.len(), ncols, |r, c| x[r].powi(c as i32))
    }
}

/// Chirps `cos(2π·i·(x/2)²)`.
pub(super) struct Chirp;

impl Basis for Chirp {
    fn name(&self) -> &'static str {
        "chirp"
    }
    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        RMat::from_fn(x.len(), ncols, |r, c| {
            let h = 0.5 * x[r];
            (2.0 * std::f64::consts::PI * c as f64 * h * h).cos()
        })
    }
}

/// Uniform random numbers in `[0, 1)`, one column at a time.
pub(super) struct Random;

impl Basis for Random {
    fn name(&self) -> &'static str {
        "random"
    }
    fn matrix(&self, x: &[f64], ncols: usize, rng: &mut Rng) -> RMat<f64> {
        let mut m = RMat::<f64>::zeros(x.len(), ncols);
        for c in 0..ncols {
            for r in 0..x.len() {
                m.set(r, c, rng.random::<f64>());
            }
        }
        m
    }
}
