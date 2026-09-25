//! Nelder-Mead, the downhill simplex method.

use super::{bounds, clamp, finish};
use crate::error::{Error, Result};
use crate::optim::{Derivatives, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl};

/// Reflection, expansion, contraction and shrink coefficients: the standard ones.
const ALPHA: f64 = 1.0;
const GAMMA: f64 = 2.0;
const RHO: f64 = 0.5;
const SIGMA: f64 = 0.5;

/// The initial simplex moves each coordinate of the starting point by this fraction, or to [`ZERO_STEP`] where
/// the coordinate is zero.  Both are scipy's.
const STEP: f64 = 0.05;
const ZERO_STEP: f64 = 0.00025;

/// Converged once the simplex is this small in both the parameters and the objective.
const XTOL: f64 = 1e-8;
const FTOL: f64 = 1e-8;

/// Nelder-Mead simplex search.  The simplex starts as the scipy one — the starting point plus one vertex per
/// coordinate, each moved 5% — and the run stops once the simplex has shrunk below `1e-8` in the parameters and
/// in the objective.
///
/// `tests/nelder_mead_matches_argmin.rs` holds this against argmin's implementation from the same simplex: the
/// two evaluate the same points in the same order.  They are not bit-identical only because the centroid is
/// summed and divided once here, where argmin multiplies by a rounded `1/n`; this way rounds once instead of
/// twice.  Only the stopping rule really differs — argmin stops on the standard deviation of the vertex costs,
/// this stops on the spread of the simplex itself, and runs a few per cent longer for it.
#[derive(Clone, Copy, Debug, Default)]
pub struct NelderMead;

impl Optimizer for NelderMead {
    fn name(&self) -> &'static str {
        "nelder-mead"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::None
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        if x0.is_empty() {
            return Err(Error::NotSupported("Nelder-Mead needs at least one parameter".into()));
        }
        finish(Monitor::new(ctl), |m| search(obj, x0, m))
    }
}

/// The simplex and the objective at each vertex, kept sorted best first.
struct Simplex {
    vertices: Vec<Vec<f64>>,
    values: Vec<f64>,
}

impl Simplex {
    /// The worst vertex is the last one.
    fn worst(&self) -> usize {
        self.vertices.len() - 1
    }

    fn sort(&mut self) {
        let mut order: Vec<usize> = (0..self.values.len()).collect();
        order.sort_by(|&a, &b| self.values[a].total_cmp(&self.values[b]));
        self.vertices = order.iter().map(|&i| self.vertices[i].clone()).collect();
        self.values = order.iter().map(|&i| self.values[i]).collect();
    }

    /// Centroid of every vertex but the worst.
    fn centroid(&self) -> Vec<f64> {
        let n = self.worst();
        let mut c = vec![0.0; self.vertices[0].len()];
        for v in &self.vertices[..n] {
            for (c, v) in c.iter_mut().zip(v) {
                *c += v;
            }
        }
        // Summed first and divided once: n roundings fewer than scaling each vertex on the way in.
        c.iter_mut().for_each(|c| *c /= n as f64);
        c
    }

    /// The simplex has collapsed: every vertex is within [`XTOL`] of the best one and their values within
    /// [`FTOL`].
    fn collapsed(&self) -> bool {
        let n = self.worst();
        let spread = self.vertices[1..]
            .iter()
            .flat_map(|v| v.iter().zip(&self.vertices[0]).map(|(a, b)| (a - b).abs()))
            .fold(0.0f64, f64::max);
        spread <= XTOL && (self.values[n] - self.values[0]).abs() <= FTOL
    }

    fn replace_worst(&mut self, x: Vec<f64>, f: f64) {
        let w = self.worst();
        self.vertices[w] = x;
        self.values[w] = f;
    }
}

fn search(obj: &dyn Objective, x0: &[f64], m: &mut Monitor) -> Result<Exit> {
    let n = x0.len();
    let (lower, upper) = bounds(x0);
    // `point` walks from `from` towards `to` by `t`, which is how every reflection, expansion and contraction
    // below is expressed.
    let point = |from: &[f64], to: &[f64], t: f64| -> Vec<f64> {
        let mut p: Vec<f64> = from.iter().zip(to).map(|(a, b)| a + t * (b - a)).collect();
        clamp(&mut p, &lower, &upper);
        p
    };

    let mut simplex = Simplex {
        vertices: Vec::with_capacity(n + 1),
        values: Vec::with_capacity(n + 1),
    };
    simplex.vertices.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] = if v[i] == 0.0 { ZERO_STEP } else { v[i] * (1.0 + STEP) };
        clamp(&mut v, &lower, &upper);
        simplex.vertices.push(v);
    }
    for i in 0..=n {
        let f = m.evaluate(obj, &simplex.vertices[i])?;
        simplex.values.push(f);
        if m.exit.is_some() {
            return Ok(Exit::Converged);
        }
    }

    loop {
        simplex.sort();
        if simplex.collapsed() {
            return Ok(Exit::Converged);
        }
        let c = simplex.centroid();
        let worst = simplex.vertices[simplex.worst()].clone();
        let (best, second_worst) = (simplex.values[0], simplex.values[simplex.worst() - 1]);

        let mut reflected: Vec<f64> = c.iter().zip(&worst).map(|(c, w)| c + ALPHA * (c - w)).collect();
        clamp(&mut reflected, &lower, &upper);
        let f_reflected = m.evaluate(obj, &reflected)?;
        if m.exit.is_some() {
            return Ok(Exit::Converged);
        }

        if f_reflected < best {
            // Reflection was the new best: try going further in the same direction.
            let expanded = point(&c, &reflected, GAMMA);
            let f_expanded = m.evaluate(obj, &expanded)?;
            if m.exit.is_some() {
                return Ok(Exit::Converged);
            }
            if f_expanded < f_reflected {
                simplex.replace_worst(expanded, f_expanded);
            } else {
                simplex.replace_worst(reflected, f_reflected);
            }
        } else if f_reflected < second_worst {
            simplex.replace_worst(reflected, f_reflected);
        } else {
            // Contract, on the reflected side if reflection at least beat the worst vertex, on the inside if
            // it did not.  A contraction no better than what it replaces shrinks the whole simplex.
            let f_worst = simplex.values[simplex.worst()];
            let outside = f_reflected < f_worst;
            let contracted = if outside {
                point(&c, &reflected, RHO)
            } else {
                point(&c, &worst, RHO)
            };
            let f_contracted = m.evaluate(obj, &contracted)?;
            if m.exit.is_some() {
                return Ok(Exit::Converged);
            }
            let accepted = if outside {
                f_contracted <= f_reflected
            } else {
                f_contracted < f_worst
            };
            if accepted {
                simplex.replace_worst(contracted, f_contracted);
            } else {
                let best_vertex = simplex.vertices[0].clone();
                for i in 1..=n {
                    simplex.vertices[i] = point(&best_vertex, &simplex.vertices[i], SIGMA);
                    simplex.values[i] = m.evaluate(obj, &simplex.vertices[i])?;
                    if m.exit.is_some() {
                        return Ok(Exit::Converged);
                    }
                }
            }
        }
    }
}
