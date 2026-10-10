//! The two subproblems, both over a ball and a box with bounds relative to the centre (`lower ≤ 0 ≤ upper`): the
//! trust-region step on the Gauss–Newton model, and the step that moves a linear function the most.

use nalgebra::{DMatrix, DVector};

/// `‖r‖² − ‖r + J·s‖²`, written as `−2·rᵀ(J·s) − ‖J·s‖²` so that it does not cancel when `‖r‖` is large.
pub(super) fn predicted_decrease(j: &DMatrix<f64>, r: &DVector<f64>, s: &DVector<f64>) -> f64 {
    let js = j * s;
    -2.0 * r.dot(&js) - js.norm_squared()
}

/// The largest `t ≥ 0` with `‖s + t·d‖ = delta`, for `‖s‖ ≤ delta` and `d ≠ 0`.
fn to_sphere(s: &DVector<f64>, d: &DVector<f64>, delta: f64) -> f64 {
    let a = d.norm_squared();
    let b = 2.0 * s.dot(d);
    let c = (s.norm_squared() - delta * delta).min(0.0);
    let root = (b * b - 4.0 * a * c).max(0.0).sqrt();
    if b >= 0.0 {
        if b + root == 0.0 { 0.0 } else { -2.0 * c / (b + root) }
    } else {
        (root - b) / (2.0 * a)
    }
}

/// The nearest bound along `d` among the free coordinates: its step length and index.  `None` when `d` never
/// leaves the box.
fn to_bound(
    s: &DVector<f64>,
    d: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
    free: &[bool],
) -> Option<(f64, usize)> {
    let mut nearest: Option<(f64, usize)> = None;
    for i in 0..s.len() {
        if !free[i] || d[i] == 0.0 {
            continue;
        }
        let room = if d[i] > 0.0 { upper[i] - s[i] } else { lower[i] - s[i] };
        let t = (room / d[i]).max(0.0);
        if nearest.is_none_or(|(b, _)| t < b) {
            nearest = Some((t, i));
        }
    }
    nearest
}

/// Zero the fixed coordinates.
fn mask(v: &mut DVector<f64>, free: &[bool]) {
    for (x, &f) in v.iter_mut().zip(free) {
        if !f {
            *x = 0.0;
        }
    }
}

/// Clamp `s` into the box against rounding.
fn inside(mut s: DVector<f64>, lower: &DVector<f64>, upper: &DVector<f64>) -> DVector<f64> {
    for i in 0..s.len() {
        s[i] = s[i].clamp(lower[i], upper[i]);
    }
    s
}

/// The step minimising `‖r + J·s‖²` over `‖s‖ ≤ delta` and `lower ≤ s ≤ upper`, by truncated conjugate gradients
/// on the Gauss–Newton model: a variable that reaches a bound is fixed there and CG restarts on the others;
/// reaching the sphere ends the step.  The first step is steepest descent cut short by the region, so the Cauchy
/// decrease holds.  Powell's TRSBOX goes on to improve the step along the sphere; that refinement is not done.
pub(super) fn trust_region_step(
    j: &DMatrix<f64>,
    r: &DVector<f64>,
    delta: f64,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
) -> DVector<f64> {
    let n = j.ncols();
    let mut s = DVector::zeros(n);
    let g0 = 2.0 * j.tr_mul(r);
    // A variable on a bound with the gradient pushing it out stays there.
    let mut free: Vec<bool> = (0..n)
        .map(|i| !((lower[i] >= 0.0 && g0[i] > 0.0) || (upper[i] <= 0.0 && g0[i] < 0.0)))
        .collect();
    let mut steps = 0;
    'restart: loop {
        let mut g = 2.0 * j.tr_mul(&(r + j * &s));
        mask(&mut g, &free);
        // Relative to the free gradient where this run of CG starts, never the full one: a large gradient on a
        // coordinate held at its bound must not hide the free coordinates' step.
        let tol = 1e-10 * g.norm();
        if g.norm() == 0.0 {
            return inside(s, lower, upper);
        }
        let mut d = -&g;
        loop {
            if steps == n {
                return inside(s, lower, upper);
            }
            steps += 1;
            let mut hd = 2.0 * j.tr_mul(&(j * &d));
            mask(&mut hd, &free);
            let curvature = d.dot(&hd);
            let t_cg = if curvature > 0.0 {
                g.norm_squared() / curvature
            } else {
                f64::INFINITY
            };
            let t_sphere = to_sphere(&s, &d, delta);
            let bound = to_bound(&s, &d, lower, upper, &free);
            let t_bound = bound.map_or(f64::INFINITY, |b| b.0);
            if t_sphere <= t_cg && t_sphere <= t_bound {
                s += t_sphere * &d;
                return inside(s, lower, upper);
            }
            if let Some((t, i)) = bound.filter(|_| t_bound < t_cg) {
                s += t * &d;
                s[i] = if d[i] > 0.0 { upper[i] } else { lower[i] };
                free[i] = false;
                continue 'restart;
            }
            s += t_cg * &d;
            let g_new = &g + t_cg * &hd;
            if g_new.norm() <= tol {
                return inside(s, lower, upper);
            }
            let beta = g_new.norm_squared() / g.norm_squared();
            d = -&g_new + beta * &d;
            g = g_new;
        }
    }
}

/// The step minimising `gᵀs` over the ball and the box, by DFO-GN's Algorithm 3 (Cartis & Roberts): head along
/// `−g` to the sphere; if a bound comes first, stop there, fix that coordinate and carry on along the rest of `−g`.
fn minimise_linear(g: &DVector<f64>, delta: f64, lower: &DVector<f64>, upper: &DVector<f64>) -> DVector<f64> {
    let n = g.len();
    let mut s = DVector::zeros(n);
    let mut d = -g;
    let mut free = vec![true; n];
    for _ in 0..n {
        if d.norm() == 0.0 {
            return s;
        }
        let t = to_sphere(&s, &d, delta);
        match to_bound(&s, &d, lower, upper, &free) {
            Some((tb, i)) if tb < t => {
                s += tb * &d;
                s[i] = if d[i] > 0.0 { upper[i] } else { lower[i] };
                free[i] = false;
                d[i] = 0.0;
            }
            _ => {
                s += t * &d;
                return inside(s, lower, upper);
            }
        }
    }
    s
}

/// The step maximising `|gᵀs|` over the ball and the box: the better of minimising `gᵀs` and `−gᵀs`.
pub(super) fn maximise_linear(
    g: &DVector<f64>,
    delta: f64,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
) -> DVector<f64> {
    let down = minimise_linear(g, delta, lower, upper);
    let up = minimise_linear(&-g, delta, lower, upper);
    if g.dot(&up).abs() > g.dot(&down).abs() {
        up
    } else {
        down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: &[f64]) -> DVector<f64> {
        DVector::from_column_slice(x)
    }

    /// Brute force in two dimensions: the largest `|gᵀs|` over a grid of the ball ∩ box.  A lower bound on the
    /// true maximum.
    fn brute_max(g: &DVector<f64>, delta: f64, lower: &DVector<f64>, upper: &DVector<f64>) -> f64 {
        let n = 1001;
        let at = |i: usize| delta * (2.0 * i as f64 / (n - 1) as f64 - 1.0);
        let mut best = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                let s = v(&[at(a), at(b)]);
                if s.norm() <= delta && (0..2).all(|i| lower[i] <= s[i] && s[i] <= upper[i]) {
                    best = best.max(g.dot(&s).abs());
                }
            }
        }
        best
    }

    #[test]
    fn the_linear_step_maximises_over_ball_and_box() {
        let g = v(&[0.6, -0.8]);
        for (lower, upper) in [
            (v(&[-10.0, -10.0]), v(&[10.0, 10.0])), // the box plays no part
            (v(&[-0.3, -1.0]), v(&[0.2, 0.5])),     // both coordinates cut
            (v(&[0.0, 0.0]), v(&[1.0, 1.0])),       // the centre on a corner of the box
        ] {
            let s = maximise_linear(&g, 1.0, &lower, &upper);
            assert!(s.norm() <= 1.0 + 1e-12);
            assert!((0..2).all(|i| lower[i] <= s[i] && s[i] <= upper[i]), "{s}");
            let best = brute_max(&g, 1.0, &lower, &upper);
            assert!(g.dot(&s).abs() >= best - 1e-12, "{} < {best}", g.dot(&s).abs());
        }
    }

    /// At the upper corner of the box with the explored span along (1, 1), neither ±(1, −1) is feasible, yet the
    /// linear step finds a feasible one with at least 1/√2 of its length outside the span.
    #[test]
    fn a_new_direction_exists_at_a_corner_of_the_box() {
        let q = v(&[1.0, -1.0]) / 2f64.sqrt();
        let s = maximise_linear(&q, 0.5, &v(&[-200.0, -200.0]), &v(&[0.0, 0.0]));
        assert!(s.iter().all(|&x| x <= 0.0) && s.norm() > 0.0, "{s}");
        let span = v(&[1.0, 1.0]) / 2f64.sqrt();
        let outside = (&s - &span * span.dot(&s)).norm();
        assert!(outside >= s.norm() / 2f64.sqrt() - 1e-12, "{s}");
    }

    fn tall() -> (DMatrix<f64>, DVector<f64>) {
        let j = DMatrix::from_row_slice(
            5,
            3,
            &[
                1.0, 0.2, -0.3, 0.1, 2.0, 0.4, -0.5, 0.3, 1.5, 0.7, -0.1, 0.2, 0.0, 0.6, -0.9,
            ],
        );
        (j, v(&[1.0, -2.0, 0.5, 3.0, -1.0]))
    }

    #[test]
    fn a_large_region_gives_the_gauss_newton_step() {
        let (j, r) = tall();
        let big = DVector::from_element(3, 1e6);
        let s = trust_region_step(&j, &r, 1e6, &-&big, &big);
        let want = j.clone().svd(true, true).solve(&-&r, 1e-14).unwrap();
        assert!((&s - &want).norm() < 1e-8 * want.norm(), "{s} vs {want}");
    }

    #[test]
    fn the_step_stays_in_the_region_and_beats_the_cauchy_point() {
        let (j, r) = tall();
        for (delta, lower, upper) in [
            (0.1, v(&[-1.0; 3]), v(&[1.0; 3])),
            (1.0, v(&[-0.05, -1.0, -0.2]), v(&[0.3, 0.01, 1.0])),
        ] {
            let s = trust_region_step(&j, &r, delta, &lower, &upper);
            assert!(s.norm() <= delta * (1.0 + 1e-12));
            assert!((0..3).all(|i| lower[i] <= s[i] && s[i] <= upper[i]), "{s}");
            // The Cauchy point: steepest descent as far as the model, the sphere and the box allow.
            let g = 2.0 * j.tr_mul(&r);
            let d = -&g;
            let zero = DVector::zeros(3);
            let t = (g.norm_squared() / (2.0 * (&j * &d).norm_squared()))
                .min(to_sphere(&zero, &d, delta))
                .min(to_bound(&zero, &d, &lower, &upper, &[true; 3]).map_or(f64::INFINITY, |b| b.0));
            let cauchy = t * &d;
            assert!(predicted_decrease(&j, &r, &s) >= predicted_decrease(&j, &r, &cauchy) - 1e-12);
        }
    }

    #[test]
    fn the_predicted_decrease_does_not_cancel() {
        let j = DMatrix::identity(2, 2);
        let r = v(&[1e8, 1.0]);
        let s = v(&[0.0, -1.0]);
        assert_eq!(predicted_decrease(&j, &r, &s), 1.0);
        let naive = r.norm_squared() - (&r + &j * &s).norm_squared();
        assert_eq!(naive, 0.0, "the difference of norms loses it");
    }

    /// `r = [10⁶(x₁ − 200), x₂ − 5]` near the upper bound `x₁ ≤ 100`: the huge gradient in the first coordinate points
    /// out of the box, and must not drown the second coordinate's step, whether the first starts on its bound or
    /// reaches it during CG.
    #[test]
    fn a_large_gradient_at_a_bound_does_not_hide_the_free_step() {
        let j = DMatrix::from_row_slice(2, 2, &[1e6, 0.0, 0.0, 1.0]);
        for (x1, want) in [(100.0, [0.0, 5.0]), (99.0, [1.0, 5.0])] {
            let r = v(&[1e6 * (x1 - 200.0), -5.0]);
            let (lower, upper) = (v(&[-100.0 - x1, -100.0]), v(&[100.0 - x1, 100.0]));
            let s = trust_region_step(&j, &r, 10.0, &lower, &upper);
            assert!((&s - v(&want)).norm() < 1e-6, "x₁ = {x1}: s = {s}");
            assert!(predicted_decrease(&j, &r, &s) > 0.0);
        }
    }
}
