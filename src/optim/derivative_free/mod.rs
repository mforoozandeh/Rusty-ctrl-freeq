//! Derivative-free optimisers: COBYLA, BOBYQA, Nelder-Mead, SPSA and CMA-ES.
//!
//! None of them asks the objective for a derivative, all of them count one iteration per function evaluation, as
//! in the Python package, and all of them keep the parameters inside one box: a solver with no gradient has
//! nothing else holding it near the start, and a pulse far outside the box is not worth propagating.
//!
//! [`Monitor::evaluate`](super::Monitor::evaluate) does the evaluating, the reporting and the stop checks, so an
//! algorithm here only has to look at [`Monitor::exit`](super::Monitor::exit) after each evaluation and stop when
//! it is set.

mod cma_es;
mod nelder_mead;
mod powell;
mod spsa;

pub use cma_es::CmaEs;
pub use nelder_mead::NelderMead;
pub use powell::{Bobyqa, Cobyla};
pub use spsa::Spsa;

use super::{Exit, Monitor, OptimResult};
use crate::error::{Error, Result};

/// Half-width of the box the parameters are kept in.  Basis coefficients stay far inside it; derivative-free
/// solvers need a finite box.
const BOX: f64 = 100.0;

fn bounds(x0: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let half = x0.iter().fold(BOX, |m, v| m.max(2.0 * v.abs()));
    (vec![-half; x0.len()], vec![half; x0.len()])
}

/// Bring `x` back inside the box, and replace anything non-finite by the bound it ran off.
fn clamp(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for i in 0..x.len() {
        x[i] = if x[i].is_nan() {
            0.0
        } else {
            x[i].clamp(lower[i], upper[i])
        };
    }
}

/// The result after the objective failed: the best point so far if there is one, the error otherwise.
fn failed(monitor: Monitor, error: Error) -> Result<OptimResult> {
    let n = monitor.evaluations;
    if monitor.best.is_some() {
        monitor.finish(n, Exit::Failed(error.to_string()))
    } else {
        Err(error)
    }
}

/// Run `search` under `monitor` and turn what it returns into a result.
///
/// `search` reports its own convergence; a stop the monitor decided on overrides it, and an objective failure
/// still returns the best point reached.
fn finish(mut monitor: Monitor, search: impl FnOnce(&mut Monitor) -> Result<Exit>) -> Result<OptimResult> {
    match search(&mut monitor) {
        Ok(exit) => {
            let n = monitor.evaluations;
            monitor.finish(n, exit)
        }
        Err(e) => failed(monitor, e),
    }
}
