//! First-order dual numbers: a value and one directional derivative.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use num_traits::{Num, One, Zero};

use super::scalar::{Scalar, sinc_derivative_f64, sinc_f64};

/// `re + eps·ε` with `ε² = 0`.  Arithmetic on duals carries the derivative of every intermediate along a single
/// direction, which is what the Hessian-vector product needs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dual {
    /// The value.
    pub re: f64,
    /// The derivative along the seeded direction.
    pub eps: f64,
}

impl Dual {
    /// A dual with the given value and derivative.
    pub const fn new(re: f64, eps: f64) -> Self {
        Dual { re, eps }
    }

    /// A constant: derivative zero.
    pub const fn constant(re: f64) -> Self {
        Dual { re, eps: 0.0 }
    }

    /// Apply a scalar function with value `f` and derivative `df` at `self.re`.
    #[inline]
    fn chain(self, f: f64, df: f64) -> Self {
        Dual {
            re: f,
            eps: df * self.eps,
        }
    }
}

impl PartialOrd for Dual {
    /// Duals order by value alone, so pivoting and `max` choose the same branch as the `f64` computation.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(self, o: Dual) -> Dual {
        Dual::new(self.re + o.re, self.eps + o.eps)
    }
}

impl Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, o: Dual) -> Dual {
        Dual::new(self.re - o.re, self.eps - o.eps)
    }
}

impl Mul for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        Dual::new(self.re * o.re, self.re * o.eps + self.eps * o.re)
    }
}

impl Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: Dual) -> Dual {
        let q = self.re / o.re;
        Dual::new(q, (self.eps - q * o.eps) / o.re)
    }
}

impl Rem for Dual {
    type Output = Dual;
    /// `a - trunc(a / b)·b`, differentiated with the truncation held constant.
    fn rem(self, o: Dual) -> Dual {
        let k = (self.re / o.re).trunc();
        Dual::new(self.re % o.re, self.eps - k * o.eps)
    }
}

impl Neg for Dual {
    type Output = Dual;
    #[inline]
    fn neg(self) -> Dual {
        Dual::new(-self.re, -self.eps)
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, o: Dual) {
        *self = *self + o;
    }
}

impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, o: Dual) {
        *self = *self - o;
    }
}

impl MulAssign for Dual {
    #[inline]
    fn mul_assign(&mut self, o: Dual) {
        *self = *self * o;
    }
}

impl DivAssign for Dual {
    #[inline]
    fn div_assign(&mut self, o: Dual) {
        *self = *self / o;
    }
}

impl RemAssign for Dual {
    fn rem_assign(&mut self, o: Dual) {
        *self = *self % o;
    }
}

impl Zero for Dual {
    fn zero() -> Self {
        Dual::constant(0.0)
    }
    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.eps == 0.0
    }
}

impl One for Dual {
    fn one() -> Self {
        Dual::constant(1.0)
    }
}

impl Num for Dual {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(s, radix).map(Dual::constant)
    }
}

impl Scalar for Dual {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Dual::constant(v)
    }
    #[inline]
    fn re(self) -> f64 {
        self.re
    }
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        let d = if s == 0.0 { 0.0 } else { 0.5 / s };
        self.chain(s, d)
    }
    fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin())
    }
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }
    fn ln(self) -> Self {
        self.chain(self.re.ln(), 1.0 / self.re)
    }
    fn atan2(self, x: Self) -> Self {
        let r2 = x.re * x.re + self.re * self.re;
        let eps = if r2 == 0.0 {
            0.0
        } else {
            (x.re * self.eps - self.re * x.eps) / r2
        };
        Dual::new(self.re.atan2(x.re), eps)
    }
    fn abs(self) -> Self {
        let sign = if self.re > 0.0 {
            1.0
        } else if self.re < 0.0 {
            -1.0
        } else {
            0.0
        };
        self.chain(self.re.abs(), sign)
    }
    fn powi(self, n: i32) -> Self {
        let d = if n == 0 { 0.0 } else { n as f64 * self.re.powi(n - 1) };
        self.chain(self.re.powi(n), d)
    }
    fn sinc(self) -> Self {
        self.chain(sinc_f64(self.re), sinc_derivative_f64(self.re))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::C;

    /// The dual derivative of `f` at `x` against a central difference of the `f64` version.
    fn check(name: &str, f: impl Fn(Dual) -> Dual, g: impl Fn(f64) -> f64, x: f64) {
        let h = 1e-6;
        let fd = (g(x + h) - g(x - h)) / (2.0 * h);
        let ad = f(Dual::new(x, 1.0));
        assert!(
            (ad.re - g(x)).abs() <= 1e-15 * g(x).abs().max(1.0),
            "{name} value at {x}"
        );
        let scale = fd.abs().max(1.0);
        assert!(
            (ad.eps - fd).abs() <= 1e-7 * scale,
            "{name}'({x}): dual {} vs fd {fd}",
            ad.eps
        );
    }

    #[test]
    fn elementary_functions_match_finite_differences() {
        for &x in &[0.3, -1.7, 2.5] {
            check("sin", |v| v.sin(), f64::sin, x);
            check("cos", |v| v.cos(), f64::cos, x);
            check("exp", |v| v.exp(), f64::exp, x);
            check("abs", |v| v.abs(), f64::abs, x);
            check("powi3", |v| v.powi(3), |v| v.powi(3), x);
            check("sinc", |v| v.sinc(), sinc_f64, x);
            check("atan2", |v| v.atan2(Dual::constant(0.7)), |v| v.atan2(0.7), x);
            check("atan2x", |v| Dual::constant(0.7).atan2(v), |v| 0.7f64.atan2(v), x);
            check(
                "div",
                |v| Dual::constant(1.3) / (v * v + Dual::constant(1.0)),
                |v| 1.3 / (v * v + 1.0),
                x,
            );
        }
        for &x in &[0.3, 2.5] {
            check("sqrt", |v| v.sqrt(), f64::sqrt, x);
            check("ln", |v| v.ln(), f64::ln, x);
        }
    }

    #[test]
    fn sinc_is_smooth_through_zero() {
        for &x in &[0.0, 1e-5, -1e-5, 9e-5, 1.1e-4] {
            let h = 1e-7;
            let fd = (sinc_f64(x + h) - sinc_f64(x - h)) / (2.0 * h);
            let ad = Dual::new(x, 1.0).sinc();
            assert!((ad.eps - fd).abs() < 1e-7, "sinc'({x})");
            assert!((ad.re - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn sqrt_has_zero_derivative_at_zero() {
        assert_eq!(Dual::new(0.0, 1.0).sqrt(), Dual::new(0.0, 0.0));
    }

    /// Complex arithmetic over duals is the product rule applied to real and imaginary parts.
    #[test]
    fn complex_duals_follow_the_product_rule() {
        let t = 0.4;
        let z = |s: f64| C::new(1.0 + s, -0.5 * s);
        let w = C::new(0.3, 2.0);
        let f64_at = |s: f64| (z(s) * w * z(s)).norm_sqr();
        let dz = C::new(Dual::new(1.0 + t, 1.0), Dual::new(-0.5 * t, -0.5));
        let dw = C::new(Dual::constant(0.3), Dual::constant(2.0));
        let prod = dz * dw * dz;
        let dual = prod.re * prod.re + prod.im * prod.im;
        let h = 1e-6;
        let fd = (f64_at(t + h) - f64_at(t - h)) / (2.0 * h);
        assert!((dual.eps - fd).abs() < 1e-6 * fd.abs().max(1.0));
    }
}
