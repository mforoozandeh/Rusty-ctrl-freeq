//! The interpolation set and the linear model of the residuals it determines (spec 5.5, 5.8).

use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, StandardNormal};

use crate::optim::residual_difference;
use crate::setup::Rng;

/// Below this fraction of the largest, a diagonal entry of `R` marks the set as degenerate.
const RANK_TOL: f64 = 1e-12;

/// A standard normal draw.
pub(super) fn normal(rng: &mut Rng) -> f64 {
    StandardNormal.sample(rng)
}

/// `v` without its component in the span of the orthonormal columns of `basis`.
pub(super) fn project_out(basis: &DMatrix<f64>, v: &DVector<f64>) -> DVector<f64> {
    if basis.ncols() == 0 {
        return v.clone();
    }
    v - basis * basis.tr_mul(v)
}

/// The length of `v` outside the span of the orthonormal columns of `basis`.
pub(super) fn outside(basis: &DMatrix<f64>, v: &DVector<f64>) -> f64 {
    project_out(basis, v).norm()
}

/// Up to `count` orthonormal vectors orthogonal to the orthonormal columns of `range`: first the normalised part of
/// `r` outside them, then coordinate vectors, each made orthogonal to everything before it (Gram–Schmidt, twice).
fn left_vectors(range: &DMatrix<f64>, r: &DVector<f64>, count: usize) -> DMatrix<f64> {
    let m = range.nrows();
    let coordinates = (0..m).map(|i| {
        let mut e = DVector::zeros(m);
        e[i] = 1.0;
        e
    });
    let mut basis: Vec<DVector<f64>> = Vec::with_capacity(count);
    for candidate in std::iter::once(r.clone()).chain(coordinates) {
        if basis.len() == count {
            break;
        }
        let scale = candidate.norm();
        if scale == 0.0 {
            continue;
        }
        let mut v = candidate;
        for _ in 0..2 {
            v = project_out(range, &v);
            for b in &basis {
                v -= b * b.dot(&v);
            }
        }
        let length = v.norm();
        if length > 1e-8 * scale {
            basis.push(v / length);
        }
    }
    DMatrix::from_columns(&basis)
}

/// `count` seeded random orthonormal vectors orthogonal to the orthonormal columns of `span`.
fn random_complement(span: &DMatrix<f64>, count: usize, rng: &mut Rng) -> DMatrix<f64> {
    let n = span.nrows();
    let mut g = DMatrix::from_fn(n, count, |_, _| normal(rng));
    if span.ncols() > 0 {
        for _ in 0..2 {
            g -= span * span.tr_mul(&g);
        }
    }
    g.qr().q()
}

/// One evaluated point.
#[derive(Clone, Debug)]
pub(super) struct Point {
    /// The point exactly as evaluated.
    pub x: DVector<f64>,
    /// Its residuals.
    pub r: DVector<f64>,
    /// `‖r‖²`.
    pub f: f64,
}

impl Point {
    pub fn new(x: DVector<f64>, r: Vec<f64>) -> Point {
        let r = DVector::from_vec(r);
        let f = r.norm_squared();
        Point { x, r, f }
    }
}

/// The interpolation set: `x_k`, the best point, and up to `n` others, stored as evaluated.  Offsets `x_t − x_k`
/// are formed when the model is built; subtraction of nearby coordinates is exact or off by `ε·|x|`, so the model
/// uses exactly the points that were evaluated.
pub(super) struct Set {
    pub points: Vec<Point>,
    /// Index of `x_k`.
    pub k: usize,
}

/// The model a set determines, or the points that make it degenerate.
pub(super) enum Build {
    Model(Model),
    /// Set indices of the points beyond the numerical rank.
    Unpoised(Vec<usize>),
}

/// The linear model `r(x_k + s) ≈ r_k + J·s` and the set's Lagrange functions.
pub(super) struct Model {
    /// `m × n`.
    pub j: DMatrix<f64>,
    /// `r(x_k)`.
    pub r: DVector<f64>,
    /// An orthonormal basis of the offsets' span, `n × p`.
    pub span: DMatrix<f64>,
    /// `S/α` with `W·S = I`: column `c` is the gradient of the Lagrange function of point `others[c]`.
    lagrange: DMatrix<f64>,
    /// Set indices of the points other than `x_k`, in column order.
    others: Vec<usize>,
    k: usize,
}

impl Set {
    pub fn new(first: Point) -> Set {
        Set {
            points: vec![first],
            k: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn xk(&self) -> &DVector<f64> {
        &self.points[self.k].x
    }

    pub fn best(&self) -> &Point {
        &self.points[self.k]
    }

    /// Whether `p` is strictly better than `x_k`, by [`residual_difference`]; a tie is not.
    pub fn beats(&self, p: &Point) -> bool {
        residual_difference(self.best().r.as_slice(), p.r.as_slice()) > 0.0
    }

    /// Add a point, moving `x_k` to it if it is better.
    pub fn push(&mut self, p: Point) {
        let better = self.beats(&p);
        self.points.push(p);
        if better {
            self.k = self.points.len() - 1;
        }
    }

    /// Put `p` in place of point `t`, moving `x_k` to it if it is better.  `x_k` itself is only ever replaced by a
    /// better point.
    pub fn replace(&mut self, t: usize, p: Point) {
        let better = self.beats(&p);
        debug_assert!(better || t != self.k, "x_k is replaced only by a better point");
        self.points[t] = p;
        if better {
            self.k = t;
        }
    }

    /// Drop the points at the indices `drop`, which never include `x_k`.
    pub fn remove(&mut self, drop: &[usize]) {
        debug_assert!(!drop.contains(&self.k));
        let below = drop.iter().filter(|&&t| t < self.k).count();
        let mut t = 0;
        self.points.retain(|_| {
            let keep = !drop.contains(&t);
            t += 1;
            keep
        });
        self.k -= below;
    }

    /// `x_t − x_k`.
    pub fn offset(&self, t: usize) -> DVector<f64> {
        &self.points[t].x - self.xk()
    }

    /// The point farthest from `x_k`, and its distance.
    pub fn farthest(&self) -> (usize, f64) {
        (0..self.len())
            .map(|t| (t, self.offset(t).norm()))
            .fold((self.k, 0.0), |a, b| if b.1 > a.1 { b } else { a })
    }

    /// An orthonormal basis of the span of the offsets, `n × (len − 1)`.
    pub fn span(&self) -> DMatrix<f64> {
        let offsets: Vec<DVector<f64>> = (0..self.len())
            .filter(|&t| t != self.k)
            .map(|t| self.offset(t))
            .collect();
        if offsets.is_empty() {
            return DMatrix::zeros(self.xk().len(), 0);
        }
        DMatrix::from_columns(&offsets).qr().q()
    }

    /// The model: `J` from a QR with column pivoting of `Wᵀ`, the matrix of offsets scaled by the largest, so that
    /// `Wᵀ·P = Q·R` and `S = Q·R⁻ᵀ·Pᵀ` satisfies `W·S = I`.  `J = (S·ΔR)ᵀ/α` is the exact interpolant for a full
    /// set and the minimum-norm one for a smaller set.
    pub fn model(&self) -> Build {
        let n = self.xk().len();
        let others: Vec<usize> = (0..self.len()).filter(|&t| t != self.k).collect();
        let p = others.len();
        let r = self.best().r.clone();
        if p == 0 {
            return Build::Model(Model {
                j: DMatrix::zeros(r.len(), n),
                r,
                span: DMatrix::zeros(n, 0),
                lagrange: DMatrix::zeros(n, 0),
                others,
                k: self.k,
            });
        }
        let offsets: Vec<DVector<f64>> = others.iter().map(|&t| self.offset(t)).collect();
        let alpha = offsets.iter().map(|d| d.norm()).fold(0.0, f64::max);
        if alpha == 0.0 {
            return Build::Unpoised(others);
        }
        let qr = (DMatrix::from_columns(&offsets) / alpha).col_piv_qr();
        let rr = qr.r();
        let largest = (0..p).map(|i| rr[(i, i)].abs()).fold(0.0, f64::max);
        // Which original column sits at each pivoted position.
        let mut order = DMatrix::from_fn(1, p, |_, c| c as f64);
        qr.p().permute_columns(&mut order);
        let dependent: Vec<usize> = (0..p)
            .filter(|&i| rr[(i, i)].abs() <= RANK_TOL * largest)
            .map(|i| others[order[(0, i)] as usize])
            .collect();
        if !dependent.is_empty() {
            return Build::Unpoised(dependent);
        }
        let q = qr.q();
        let rinv = rr
            .solve_upper_triangular(&DMatrix::identity(p, p))
            .expect("R has no zero on its diagonal");
        let mut s = &q * rinv.transpose();
        qr.p().inv_permute_columns(&mut s);
        let lagrange = s / alpha;
        let dr = DMatrix::from_fn(p, r.len(), |row, c| self.points[others[row]].r[c] - r[c]);
        let j = (&lagrange * dr).transpose();
        Build::Model(Model {
            j,
            r,
            span: q,
            lagrange,
            others,
            k: self.k,
        })
    }

    /// The point `y = x_k + s` should replace: the `t ≠ keep` maximising `|Λ_t(y)|·max(‖y_t − x_k‖⁴/Δ⁴, 1)`
    /// (DFO-GN's criterion).  `keep` is `x_k` when `y` will not take its place.
    pub fn replacement(&self, model: &Model, s: &DVector<f64>, delta: f64, keep: Option<usize>) -> usize {
        (0..self.len())
            .filter(|&t| Some(t) != keep)
            .map(|t| {
                let distance = self.offset(t).norm() / delta;
                (t, model.lagrange(t, s).abs() * distance.powi(4).max(1.0))
            })
            .fold((usize::MAX, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a })
            .0
    }
}

impl Model {
    /// The gradient of point `t`'s Lagrange function: `Λ_t(x_k + s) = δ_tk + gᵀs`.
    pub fn lagrange_gradient(&self, t: usize) -> DVector<f64> {
        match self.others.iter().position(|&o| o == t) {
            Some(c) => self.lagrange.column(c).into_owned(),
            None => -self.lagrange.column_sum(),
        }
    }

    /// `Λ_t(x_k + s)`.
    pub fn lagrange(&self, t: usize, s: &DVector<f64>) -> f64 {
        let at_xk = if t == self.k { 1.0 } else { 0.0 };
        at_xk + self.lagrange_gradient(t).dot(s)
    }

    /// Whether the offset `d` has at least `fraction` of its length outside the explored span.
    pub fn adds_direction(&self, d: &DVector<f64>, fraction: f64) -> bool {
        let length = d.norm();
        length > 0.0 && outside(&self.span, d) >= fraction * length
    }

    /// The Jacobian of a set with fewer than `n + 1` points, given slope outside the explored span (spec 5.5).
    ///
    /// `σ_q`, the smallest nonzero singular value of `J`, is put along a seeded random orthonormal basis `v_j` of
    /// the unexplored directions, paired with orthonormal `u_j` orthogonal to `range J`, the first being the part of
    /// `r_k` the model cannot explain.  `J` is unchanged on the explored span, where a zero slope is what the data
    /// says.  The result has rank `q + min(n − p, m − q)`.
    pub fn filled(&self, rng: &mut Rng) -> DMatrix<f64> {
        let (m, n) = self.j.shape();
        let p = self.span.ncols();
        let svd = self.j.clone().svd(true, false);
        let largest = svd.singular_values.max();
        if p >= n || largest == 0.0 {
            return self.j.clone();
        }
        let u = svd.u.as_ref().expect("U was asked for");
        let kept: Vec<usize> = (0..svd.singular_values.len())
            .filter(|&i| svd.singular_values[i] > RANK_TOL * largest)
            .collect();
        let sigma = kept
            .iter()
            .map(|&i| svd.singular_values[i])
            .fold(f64::INFINITY, f64::min);
        let count = (n - p).min(m - kept.len());
        if count == 0 {
            return self.j.clone();
        }
        let range = DMatrix::from_columns(&kept.iter().map(|&i| u.column(i).into_owned()).collect::<Vec<_>>());
        let left = left_vectors(&range, &self.r, count);
        let right = random_complement(&self.span, left.ncols(), rng);
        &self.j + sigma * left * right.transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn point(x: &[f64], r: impl Fn(&[f64]) -> Vec<f64>) -> Point {
        Point::new(DVector::from_column_slice(x), r(x))
    }

    /// `r(x) = x₀ + 2x₁ + 3`: `J` must be `[1, 2]`.
    fn affine(x: &[f64]) -> Vec<f64> {
        vec![x[0] + 2.0 * x[1] + 3.0]
    }

    fn model(set: &Set) -> Model {
        match set.model() {
            Build::Model(m) => m,
            Build::Unpoised(d) => panic!("unpoised: {d:?}"),
        }
    }

    /// A short offset listed before a long one, so the column pivoting swaps them.
    #[test]
    fn the_jacobian_is_exact_for_affine_residuals_whatever_the_pivoting() {
        let mut set = Set::new(point(&[0.0, 0.0], affine));
        set.points.push(point(&[0.1, 0.0], affine));
        set.points.push(point(&[0.0, 1.0], affine));
        let m = model(&set);
        assert!(
            (m.j[(0, 0)] - 1.0).abs() < 1e-12 && (m.j[(0, 1)] - 2.0).abs() < 1e-12,
            "{}",
            m.j
        );
        for t in 0..3 {
            for u in 0..3 {
                let want = if t == u { 1.0 } else { 0.0 };
                let got = m.lagrange(t, &set.offset(u));
                assert!((got - want).abs() < 1e-12, "Λ_{t}(y_{u}) = {got}");
            }
        }
    }

    #[test]
    fn a_small_set_gives_the_minimum_norm_jacobian() {
        let r = |x: &[f64]| vec![x[0] + 2.0 * x[1] + 3.0 * x[2]];
        let mut set = Set::new(point(&[0.0, 0.0, 0.0], r));
        set.points.push(point(&[0.5, 0.0, 0.0], r));
        let m = model(&set);
        assert!((m.j[(0, 0)] - 1.0).abs() < 1e-12, "{}", m.j);
        assert!(m.j[(0, 1)].abs() < 1e-12 && m.j[(0, 2)].abs() < 1e-12, "{}", m.j);
        assert_eq!(m.span.ncols(), 1);
    }

    #[test]
    fn dependent_points_are_reported() {
        let mut set = Set::new(point(&[0.0, 0.0], affine));
        set.points.push(point(&[1.0, 1.0], affine));
        set.points.push(point(&[2.0, 2.0], affine));
        match set.model() {
            Build::Unpoised(d) => assert_eq!(d, vec![1]),
            Build::Model(_) => panic!("collinear offsets cannot be poised"),
        }
    }

    /// `y = (0.5, 0.5)` against points at `(1, 0)` and `(0, 10)`: `Λ₁ = 0.5`, `Λ₂ = 0.05`.  At `Δ = 1` the far point's
    /// distance weighs it by `10⁴`; at `Δ = 100` distances stop counting and `Λ` decides.
    #[test]
    fn the_replacement_weighs_lagrange_values_by_distance() {
        let mut set = Set::new(point(&[0.0, 0.0], affine));
        set.points.push(point(&[1.0, 0.0], affine));
        set.points.push(point(&[0.0, 10.0], affine));
        let m = model(&set);
        let s = DVector::from_column_slice(&[0.5, 0.5]);
        assert_eq!(set.replacement(&m, &s, 1.0, Some(0)), 2);
        assert_eq!(set.replacement(&m, &s, 100.0, Some(0)), 1);
    }

    #[test]
    fn x_k_moves_only_to_a_strictly_better_point() {
        let r = |x: &[f64]| vec![1e8, x[0]];
        let mut set = Set::new(point(&[1.0], r));
        set.push(point(&[-1.0], r));
        assert_eq!(set.k, 0, "a tie keeps the incumbent");
        set.push(point(&[0.9], r));
        assert_eq!(set.k, 2, "0.9 beats 1 although ‖r‖² rounds equal");
        set.remove(&[1]);
        assert_eq!((set.len(), set.k), (2, 1));
    }

    fn singular_values(j: &DMatrix<f64>) -> DVector<f64> {
        j.clone().svd(false, false).singular_values
    }

    fn rank(s: &DVector<f64>) -> usize {
        s.iter().filter(|&&v| v > 1e-10 * s.max()).count()
    }

    /// `r(x) = A·x` with `A` 6 × 4 of full rank, explored along two directions: `q = p = 2` and `m ≥ n`, so the
    /// fill gives full rank, with `σ_q` along the new directions and `J` unchanged on the explored span.
    #[test]
    fn the_fill_completes_the_jacobian_outside_the_span() {
        let a = DMatrix::from_row_slice(
            6,
            4,
            &[
                1.0, 0.5, 0.0, -0.3, 0.2, 1.0, 0.4, 0.0, 0.0, -0.6, 1.0, 0.1, 0.3, 0.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.5, -0.5, 0.25, -0.25,
            ],
        );
        let r = |x: &[f64]| (&a * DVector::from_column_slice(x)).as_slice().to_vec();
        let mut set = Set::new(point(&[0.3, 0.2, 0.1, 0.0], r));
        set.points.push(point(&[0.8, 0.2, 0.1, 0.0], r));
        set.points.push(point(&[0.3, 0.2, 0.6, 0.0], r));
        let m = model(&set);
        let filled = m.filled(&mut Rng::seed_from_u64(1));
        let (before, after) = (singular_values(&m.j), singular_values(&filled));
        assert_eq!((rank(&before), rank(&after)), (2, 4));
        let sigma_q = before
            .iter()
            .copied()
            .filter(|&v| v > 1e-10 * before.max())
            .fold(f64::INFINITY, f64::min);
        assert!(
            (after.min() - sigma_q).abs() < 1e-10 * sigma_q,
            "{after} vs σ_q = {sigma_q}"
        );
        assert!(
            ((&filled - &m.j) * &m.span).norm() < 1e-12,
            "J changed on the explored span"
        );
    }

    /// Two identical residuals: `J` has rank `q = 1` though `p = 2` directions were explored (`n = 4`, `m = 2`).  The
    /// fill adds `min(n − p, m − q) = 1` direction, for rank 2, and leaves the explored span alone.
    #[test]
    fn a_rank_deficient_model_is_filled_only_outside_the_span() {
        let r = |x: &[f64]| {
            let v = x[0] + 2.0 * x[1] + 0.5;
            vec![v, v]
        };
        let mut set = Set::new(point(&[0.0; 4], r));
        set.points.push(point(&[0.5, 0.0, 0.0, 0.0], r));
        set.points.push(point(&[0.0, 0.5, 0.0, 0.0], r));
        let m = model(&set);
        let filled = m.filled(&mut Rng::seed_from_u64(1));
        assert_eq!((rank(&singular_values(&m.j)), rank(&singular_values(&filled))), (1, 2));
        assert!(
            ((&filled - &m.j) * &m.span).norm() < 1e-12,
            "J changed on the explored span"
        );
    }
}
