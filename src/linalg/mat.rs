//! A small dense row-major matrix, generic over its element type.
//!
//! The Hilbert spaces here are small - 2 to 81 dimensions - so plain loops over a flat `Vec` beat anything
//! cleverer, and one generic type serves real and complex matrices over `f64` and `Dual` alike.

use crate::autodiff::{C, Scalar};
use crate::error::{Error, Result};

/// A `rows × cols` matrix stored row-major.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Mat<E> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Entries, row by row.
    pub data: Vec<E>,
}

/// A real matrix.
pub type RMat<T> = Mat<T>;
/// A complex matrix.
pub type CMat<T> = Mat<C<T>>;

impl<E: Copy> Mat<E> {
    /// A matrix from row-major data.  Fails if the length does not match the shape.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<E>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::Dimension(format!(
                "{} entries for a {rows}x{cols} matrix",
                data.len()
            )));
        }
        Ok(Mat { rows, cols, data })
    }

    /// A matrix whose entry `(r, c)` is `f(r, c)`.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> E) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Mat { rows, cols, data }
    }

    /// A column vector.
    pub fn column(data: Vec<E>) -> Self {
        Mat {
            rows: data.len(),
            cols: 1,
            data,
        }
    }

    /// Entry `(r, c)`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> E {
        self.data[r * self.cols + c]
    }

    /// Set entry `(r, c)`.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: E) {
        self.data[r * self.cols + c] = v;
    }

    /// Row `r` as a slice.
    #[inline]
    pub fn row(&self, r: usize) -> &[E] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Whether the matrix is square.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Apply `f` to every entry.
    pub fn map<F2>(&self, f: impl Fn(E) -> F2) -> Mat<F2> {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    /// Column `c` as a vector.
    pub fn col(&self, c: usize) -> Vec<E> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// The transpose.
    pub fn transpose(&self) -> Self {
        Mat::from_fn(self.cols, self.rows, |r, c| self.get(c, r))
    }
}

impl<T: Scalar> RMat<T> {
    /// A real matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// Real matrix product.
    pub fn matmul(&self, o: &Self) -> Result<Self> {
        check_product(self.shape(), o.shape())?;
        let mut out = Self::zeros(self.rows, o.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == T::zero() {
                    continue;
                }
                let orow = o.row(k);
                let dst = &mut out.data[i * o.cols..(i + 1) * o.cols];
                for (d, &b) in dst.iter_mut().zip(orow) {
                    *d += a * b;
                }
            }
        }
        Ok(out)
    }

    /// Lift an `f64` real matrix to this scalar type.
    pub fn lift(m: &RMat<f64>) -> Self {
        m.map(T::from_f64)
    }
}

impl<T: Scalar> CMat<T> {
    /// A complex matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![C::new(T::zero(), T::zero()); rows * cols],
        }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, C::new(T::one(), T::zero()));
        }
        m
    }

    /// Lift an `f64` complex matrix to this scalar type.
    pub fn lift(m: &CMat<f64>) -> Self {
        m.map(|z| C::new(T::from_f64(z.re), T::from_f64(z.im)))
    }

    /// A complex matrix from a real one.
    pub fn from_real(m: &RMat<T>) -> Self {
        m.map(|v| C::new(v, T::zero()))
    }

    /// Complex matrix product.
    pub fn matmul(&self, o: &Self) -> Result<Self> {
        check_product(self.shape(), o.shape())?;
        let mut out = Self::zeros(self.rows, o.cols);
        let zero = C::new(T::zero(), T::zero());
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == zero {
                    continue;
                }
                let orow = o.row(k);
                let dst = &mut out.data[i * o.cols..(i + 1) * o.cols];
                for (d, &b) in dst.iter_mut().zip(orow) {
                    *d += a * b;
                }
            }
        }
        Ok(out)
    }

    /// The conjugate transpose.
    pub fn adjoint(&self) -> Self {
        Mat::from_fn(self.cols, self.rows, |r, c| self.get(c, r).conj())
    }

    /// The entrywise conjugate.
    pub fn conj(&self) -> Self {
        self.map(|z| z.conj())
    }

    /// Every entry times `s`.
    pub fn scale(&self, s: C<T>) -> Self {
        self.map(|z| z * s)
    }

    /// Every entry times the real `s`.
    pub fn scale_re(&self, s: T) -> Self {
        self.map(|z| C::new(z.re * s, z.im * s))
    }

    /// `self + o`.
    pub fn add(&self, o: &Self) -> Result<Self> {
        check_same(self.shape(), o.shape())?;
        Ok(Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(&o.data).map(|(&a, &b)| a + b).collect(),
        })
    }

    /// `self - o`.
    pub fn sub(&self, o: &Self) -> Result<Self> {
        check_same(self.shape(), o.shape())?;
        Ok(Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(&o.data).map(|(&a, &b)| a - b).collect(),
        })
    }

    /// `self += s·o`, shapes assumed equal.
    pub fn axpy(&mut self, s: C<T>, o: &Self) {
        debug_assert_eq!(self.shape(), o.shape());
        for (d, &b) in self.data.iter_mut().zip(&o.data) {
            *d += s * b;
        }
    }

    /// `self += s·o` for a real `s`, shapes assumed equal.
    pub fn axpy_re(&mut self, s: T, o: &Self) {
        debug_assert_eq!(self.shape(), o.shape());
        for (d, &b) in self.data.iter_mut().zip(&o.data) {
            d.re += s * b.re;
            d.im += s * b.im;
        }
    }

    /// The Kronecker product `self ⊗ o`.
    pub fn kron(&self, o: &Self) -> Self {
        Mat::from_fn(self.rows * o.rows, self.cols * o.cols, |r, c| {
            self.get(r / o.rows, c / o.cols) * o.get(r % o.rows, c % o.cols)
        })
    }

    /// The trace.
    pub fn trace(&self) -> C<T> {
        (0..self.rows.min(self.cols)).fold(C::new(T::zero(), T::zero()), |acc, i| acc + self.get(i, i))
    }

    /// `Σ conj(self_ij)·o_ij`, the Frobenius inner product.
    pub fn inner(&self, o: &Self) -> C<T> {
        self.data
            .iter()
            .zip(&o.data)
            .fold(C::new(T::zero(), T::zero()), |acc, (&a, &b)| acc + a.conj() * b)
    }

    /// The 1-norm (largest column sum of magnitudes) of the value parts.
    pub fn norm1(&self) -> f64 {
        (0..self.cols)
            .map(|c| {
                (0..self.rows)
                    .map(|r| {
                        let z = self.get(r, c);
                        z.re.re().hypot(z.im.re())
                    })
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }
}

impl CMat<f64> {
    /// Largest entrywise magnitude of `self - o`.
    pub fn max_abs_diff(&self, o: &Self) -> f64 {
        self.data
            .iter()
            .zip(&o.data)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max)
    }

    /// Whether the matrix equals its conjugate transpose to within `tol`.
    pub fn is_hermitian(&self, tol: f64) -> bool {
        self.is_square() && self.max_abs_diff(&self.adjoint()) <= tol
    }
}

fn check_product(a: (usize, usize), b: (usize, usize)) -> Result<()> {
    if a.1 != b.0 {
        return Err(Error::Dimension(format!(
            "cannot multiply {}x{} by {}x{}",
            a.0, a.1, b.0, b.1
        )));
    }
    Ok(())
}

fn check_same(a: (usize, usize), b: (usize, usize)) -> Result<()> {
    if a != b {
        return Err(Error::Dimension(format!(
            "shapes {}x{} and {}x{} differ",
            a.0, a.1, b.0, b.1
        )));
    }
    Ok(())
}

/// Solve `A X = B` by LU decomposition with partial pivoting on the value parts.
pub fn solve<T: Scalar>(a: &CMat<T>, b: &CMat<T>) -> Result<CMat<T>> {
    let n = a.rows;
    if !a.is_square() || b.rows != n {
        return Err(Error::Dimension(format!(
            "cannot solve a {}x{} system for {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let mut lu = a.clone();
    let mut x = b.clone();
    let m = b.cols;
    let mag = |z: C<T>| z.re.re().hypot(z.im.re());
    for k in 0..n {
        let (p, best) = (k..n)
            .map(|r| (r, mag(lu.get(r, k))))
            .fold((k, -1.0), |acc, cur| if cur.1 > acc.1 { cur } else { acc });
        if best == 0.0 || !best.is_finite() {
            return Err(Error::Numerical("singular matrix in linear solve".into()));
        }
        if p != k {
            for c in 0..n {
                lu.data.swap(k * n + c, p * n + c);
            }
            for c in 0..m {
                x.data.swap(k * m + c, p * m + c);
            }
        }
        let pivot = lu.get(k, k);
        for r in k + 1..n {
            let factor = lu.get(r, k) / pivot;
            if factor == C::new(T::zero(), T::zero()) {
                continue;
            }
            for c in k..n {
                let v = lu.get(r, c) - factor * lu.get(k, c);
                lu.set(r, c, v);
            }
            for c in 0..m {
                let v = x.get(r, c) - factor * x.get(k, c);
                x.set(r, c, v);
            }
        }
    }
    for k in (0..n).rev() {
        let pivot = lu.get(k, k);
        for c in 0..m {
            let mut v = x.get(k, c);
            for j in k + 1..n {
                v -= lu.get(k, j) * x.get(j, c);
            }
            x.set(k, c, v / pivot);
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C<f64> {
        C::new(re, im)
    }

    #[test]
    fn solve_recovers_a_known_solution() {
        let a = Mat::from_vec(
            3,
            3,
            vec![
                c(0.0, 1.0),
                c(2.0, 0.0),
                c(1.0, -1.0),
                c(3.0, 0.0),
                c(1.0, 1.0),
                c(0.0, 0.0),
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(2.0, 2.0),
            ],
        )
        .unwrap();
        let x = Mat::from_vec(
            3,
            2,
            vec![
                c(1.0, 0.0),
                c(0.0, 1.0),
                c(-2.0, 0.5),
                c(1.0, 1.0),
                c(0.3, 0.0),
                c(2.0, -1.0),
            ],
        )
        .unwrap();
        let b = a.matmul(&x).unwrap();
        let got = solve(&a, &b).unwrap();
        assert!(got.max_abs_diff(&x) < 1e-13);
    }

    #[test]
    fn singular_systems_are_reported() {
        let a = CMat::<f64>::zeros(2, 2);
        assert!(matches!(solve(&a, &CMat::identity(2)), Err(Error::Numerical(_))));
    }

    #[test]
    fn kron_orders_the_first_factor_as_most_significant() {
        let x = Mat::from_vec(2, 2, vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]).unwrap();
        let id = CMat::<f64>::identity(2);
        let k = x.kron(&id);
        // X ⊗ I maps |00> (index 0) to |10> (index 2).
        assert_eq!(k.get(2, 0), c(1.0, 0.0));
        assert_eq!(k.get(1, 0), c(0.0, 0.0));
    }

    #[test]
    fn shape_mismatches_are_errors() {
        let a = CMat::<f64>::zeros(2, 3);
        assert!(matches!(a.matmul(&a), Err(Error::Dimension(_))));
        assert!(matches!(Mat::from_vec(2, 2, vec![1.0; 3]), Err(Error::Dimension(_))));
    }
}
