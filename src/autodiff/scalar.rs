//! The real scalar type every numerical kernel is written over.
//!
//! Kernels and tape operations are generic over [`Scalar`] so the same code runs on plain `f64` - the value and
//! gradient path - and on [`Dual`](super::Dual) numbers, which carry a directional derivative alongside each value.
//! Running the reverse pass on duals seeded with a direction `v` gives the exact Hessian-vector product `H v`
//! (forward-over-reverse differentiation) without a separate second-order implementation.

use std::fmt::Debug;
use std::ops::Neg;

use num_traits::NumAssign;

/// A complex number over a [`Scalar`].
pub type C<T> = num_complex::Complex<T>;

/// A real number the crate's kernels can compute with.
pub trait Scalar:
    Copy + Send + Sync + Debug + Default + PartialEq + PartialOrd + 'static + NumAssign + Neg<Output = Self>
{
    /// A constant.
    fn from_f64(v: f64) -> Self;
    /// The value part, used for comparisons, pivoting and norms.
    fn re(self) -> f64;
    /// Square root.  Its derivative at zero is taken as zero, the subgradient the penalty relies on.
    fn sqrt(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Exponential.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Four-quadrant arctangent of `self / x`.
    fn atan2(self, x: Self) -> Self;
    /// Absolute value; derivative zero at zero.
    fn abs(self) -> Self;
    /// Integer power.
    fn powi(self, n: i32) -> Self;
    /// `sin(x) / x`, continuous at zero.
    fn sinc(self) -> Self;
}

/// Below this magnitude `sinc` and its derivative are evaluated from their Taylor series.
pub(crate) const SINC_SERIES_BELOW: f64 = 1e-4;

/// `sin(x) / x` for `f64`.
pub(crate) fn sinc_f64(x: f64) -> f64 {
    if x.abs() < SINC_SERIES_BELOW {
        let x2 = x * x;
        1.0 - x2 / 6.0 + x2 * x2 / 120.0
    } else {
        x.sin() / x
    }
}

/// Derivative of `sin(x) / x`.
pub(crate) fn sinc_derivative_f64(x: f64) -> f64 {
    if x.abs() < SINC_SERIES_BELOW {
        let x2 = x * x;
        -x / 3.0 + x * x2 / 30.0
    } else {
        (x * x.cos() - x.sin()) / (x * x)
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn re(self) -> f64 {
        self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline]
    fn sinc(self) -> Self {
        sinc_f64(self)
    }
}
