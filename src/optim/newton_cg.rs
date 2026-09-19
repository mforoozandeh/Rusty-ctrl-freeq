//! Truncated Newton: Newton steps from conjugate gradients on Hessian-vector products.
//!
//! This is the line-search Newton-CG of Nocedal and Wright (Algorithm 7.1), as torchmin and scipy implement it: the
//! Newton system `H·p = −g` is solved by conjugate gradients only as accurately as the forcing term
//! `min(0.5, √‖g‖)·‖g‖` asks, stopping early on negative curvature, and each step is found by backtracking until
//! the Armijo condition holds.  It needs only Hessian-vector products, which the objective computes exactly, so a
//! step costs a few gradient evaluations rather than a whole Hessian.

use super::{Derivatives, Eval, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl};
use crate::error::{Error, Result};

/// Sufficient-decrease constant of the Armijo condition.
const ARMIJO: f64 = 1e-4;
/// Gradient norm at which a run has converged.
const TOL_GRAD: f64 = 1e-12;
/// Smallest step fraction tried before the line search gives up.
const MIN_STEP: f64 = 1e-12;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Line-search Newton-CG with exact Hessian-vector products.
#[derive(Clone, Copy, Debug, Default)]
pub struct NewtonCg;

impl Optimizer for NewtonCg {
    fn name(&self) -> &'static str {
        "newton-cg"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::Hessian
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        let mut monitor = Monitor::new(ctl);
        let mut iterations = 0;
        let outcome = iterate(obj, x0, &mut monitor, &mut iterations);
        match outcome {
            Ok(exit) => monitor.finish(iterations, exit),
            Err(e) if monitor.best.is_some() => monitor.finish(iterations, Exit::Failed(e.to_string())),
            Err(e) => Err(e),
        }
    }
}

/// The iterations; returns the exit to use when no stop condition fired first.
fn iterate(obj: &dyn Objective, x0: &[f64], monitor: &mut Monitor, iterations: &mut usize) -> Result<Exit> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let (mut e, mut g) = obj.gradient(&x)?;
    monitor.evaluated(&x, e);
    monitor.report(0, e, Some(norm(&g)));
    while monitor.exit.is_none() {
        let gnorm = norm(&g);
        if gnorm < TOL_GRAD {
            return Ok(Exit::Converged);
        }
        let mut p = newton_direction(obj, &x, &g, n)?;
        let mut slope = dot(&g, &p);
        if slope.is_nan() || slope >= 0.0 {
            p = g.iter().map(|v| -v).collect();
            slope = -gnorm * gnorm;
        }
        let Some((xn, en)) = backtrack(obj, &x, &p, &e, slope, monitor)? else {
            return Ok(Exit::Converged);
        };
        let (en2, gn) = obj.gradient(&xn)?;
        debug_assert_eq!(en.cost, en2.cost);
        x = xn;
        e = en2;
        g = gn;
        *iterations += 1;
        monitor.report(*iterations, e, Some(norm(&g)));
    }
    Ok(Exit::Converged)
}

/// Approximately solve `H·p = −g` by conjugate gradients, truncated by the forcing term and at negative curvature.
fn newton_direction(obj: &dyn Objective, x: &[f64], g: &[f64], n: usize) -> Result<Vec<f64>> {
    let gnorm = norm(g);
    let tol = 0.5f64.min(gnorm.sqrt()) * gnorm;
    let mut z = vec![0.0; n];
    let mut r = g.to_vec();
    let mut d: Vec<f64> = g.iter().map(|v| -v).collect();
    let mut rr = dot(&r, &r);
    for j in 0..(2 * n).max(20) {
        let hd = obj.hvp(x, &d)?;
        if hd.len() != n || hd.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("the Hessian-vector product is not finite".into()));
        }
        let curvature = dot(&d, &hd);
        if curvature <= 0.0 {
            return Ok(if j == 0 { d } else { z });
        }
        let alpha = rr / curvature;
        for i in 0..n {
            z[i] += alpha * d[i];
            r[i] += alpha * hd[i];
        }
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() < tol {
            return Ok(z);
        }
        let beta = rr_new / rr;
        for i in 0..n {
            d[i] = -r[i] + beta * d[i];
        }
        rr = rr_new;
    }
    Ok(z)
}

/// Halve the step from 1 until the Armijo condition holds.  `None` if no step decreases the cost.
fn backtrack(
    obj: &dyn Objective,
    x: &[f64],
    p: &[f64],
    e: &Eval,
    slope: f64,
    monitor: &mut Monitor,
) -> Result<Option<(Vec<f64>, Eval)>> {
    let mut step = 1.0;
    while step >= MIN_STEP {
        let xn: Vec<f64> = x.iter().zip(p).map(|(a, b)| a + step * b).collect();
        match obj.value(&xn) {
            Ok(en) => {
                monitor.evaluated(&xn, en);
                if en.cost <= e.cost + ARMIJO * step * slope {
                    return Ok(Some((xn, en)));
                }
            }
            // A step into a region where the objective fails is treated as too long.
            Err(Error::Numerical(_)) => {}
            Err(other) => return Err(other),
        }
        step *= 0.5;
    }
    Ok(None)
}
