//! SPSA, simultaneous perturbation stochastic approximation.

use rand::{RngExt, SeedableRng};

use super::{bounds, clamp, finish};
use crate::error::Result;
use crate::optim::{Derivatives, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl};
use crate::setup::Rng;

/// Gain-sequence exponents: Spall's practical recommendations.  The step size decays as `(A + k + 1)^-ALPHA`
/// and the perturbation as `(k + 1)^-GAMMA`.
const ALPHA: f64 = 0.602;
const GAMMA: f64 = 0.101;

/// Fraction of the iterations the stability constant `A` covers.  Qiskit leaves `A` at zero; Spall recommends
/// about a tenth of the run, which holds the step size near its calibrated value instead of decaying from the
/// first iteration.  The first step is the same either way, because `a` is rescaled by `(A + 1)^ALPHA`.
const STABILITY: f64 = 0.1;

/// Perturbation size at the first iteration.  Qiskit uses 0.2, sized for the shot noise of real hardware; this
/// objective is deterministic, so a smaller difference gives a sharper gradient at no cost.
const C: f64 = 0.01;

/// Below this the calibrated step size is meaningless, and Qiskit falls back to the target step itself.
const CALIBRATION_FLOOR: f64 = 1e-10;

/// Iterations spent measuring the objective's scale before the first step.
const CALIBRATION_STEPS: usize = 25;

/// How far the first step should move, in parameter units.  The calibrated step size is whatever turns the
/// measured gradient magnitude into a move this big; the value is Qiskit's.
const TARGET_STEP: f64 = std::f64::consts::TAU / 10.0;

/// Evaluations one iteration costs: the two probes and the trial point.
const PER_ITERATION: usize = 3;

/// SPSA.  Every iteration estimates the gradient from two evaluations at `x ± c_k Δ`, with `Δ` drawn uniformly
/// from `±1` per parameter, so the cost of an iteration does not grow with the number of parameters.  The step
/// size is calibrated against the objective's own scale over the first 25 iterations, as in Qiskit; those
/// evaluations count like any other.
///
/// A third evaluation per iteration accepts the step only when it lowers the objective — Qiskit calls this
/// blocking, and switches it off by default.  Here it is always on: without it a calibrated step can land
/// somewhere far steeper than the point it was measured at and the iterate runs away.  Qiskit's companion
/// setting, an allowed increase of twice the objective's standard deviation at the starting point, is zero for
/// a deterministic objective, so the 25 evaluations it would take to measure are not spent.
///
/// SPSA is built for objectives too noisy to differentiate, and pays for that with a slow rate: expect it to
/// locate a basin rather than to converge inside one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spsa;

impl Optimizer for Spsa {
    fn name(&self) -> &'static str {
        "spsa"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::None
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        finish(Monitor::new(ctl), |m| search(obj, x0, m))
    }
}

/// A `±1` perturbation, one entry per parameter.
fn perturbation(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 }).collect()
}

fn search(obj: &dyn Objective, x0: &[f64], m: &mut Monitor) -> Result<Exit> {
    let n = x0.len();
    let (lower, upper) = bounds(x0);
    let mut rng = Rng::seed_from_u64(m.seed);
    let mut x = x0.to_vec();
    clamp(&mut x, &lower, &upper);

    // `A` spans a tenth of the iterations the budget leaves after calibration.
    let iterations = m.max_iter.saturating_sub(2 * CALIBRATION_STEPS + 1) / PER_ITERATION;
    let stability = STABILITY * iterations as f64;

    // The two evaluations an iteration's gradient estimate needs, at `x ± c Δ`.  `None` means the monitor
    // stopped the run in the middle of them.
    let probe = |m: &mut Monitor, x: &[f64], delta: &[f64], c: f64| -> Result<Option<f64>> {
        let mut plus: Vec<f64> = x.iter().zip(delta).map(|(x, d)| x + c * d).collect();
        let mut minus: Vec<f64> = x.iter().zip(delta).map(|(x, d)| x - c * d).collect();
        clamp(&mut plus, &lower, &upper);
        clamp(&mut minus, &lower, &upper);
        let f_plus = m.evaluate(obj, &plus)?;
        if m.exit.is_some() {
            return Ok(None);
        }
        let f_minus = m.evaluate(obj, &minus)?;
        if m.exit.is_some() {
            return Ok(None);
        }
        // With Δ drawn from ±1 its elementwise inverse is itself, so this scalar times Δ is the estimate.
        Ok(Some((f_plus - f_minus) / (2.0 * c)))
    };

    // Calibration: the mean magnitude of the finite difference at the starting point sets the step size, so
    // that the first step moves the parameters by about `TARGET_STEP` whatever the objective's units are.
    let mut magnitude = 0.0;
    for _ in 0..CALIBRATION_STEPS {
        let delta = perturbation(&mut rng, n);
        let Some(estimate) = probe(m, &x, &delta, C)? else {
            return Ok(Exit::Converged);
        };
        magnitude += estimate.abs() / CALIBRATION_STEPS as f64;
    }
    // A flat objective gives nothing to calibrate against, and one too steep to measure gives a step of no
    // use either; both fall back to the target step, as Qiskit does.
    let calibrated = TARGET_STEP / magnitude;
    let a = if calibrated >= CALIBRATION_FLOOR && calibrated.is_finite() {
        calibrated
    } else {
        TARGET_STEP
    } * (stability + 1.0).powf(ALPHA);

    let mut f_x = m.evaluate(obj, &x)?;
    let mut k = 0u64;
    while m.exit.is_none() {
        let a_k = a / (stability + k as f64 + 1.0).powf(ALPHA);
        let c_k = C / (k as f64 + 1.0).powf(GAMMA);
        k += 1;
        let delta = perturbation(&mut rng, n);
        let Some(estimate) = probe(m, &x, &delta, c_k)? else {
            return Ok(Exit::Converged);
        };
        let mut trial: Vec<f64> = x.iter().zip(&delta).map(|(x, d)| x - a_k * estimate * d).collect();
        clamp(&mut trial, &lower, &upper);
        let f_trial = m.evaluate(obj, &trial)?;
        if f_trial < f_x {
            x = trial;
            f_x = f_trial;
        }
    }
    Ok(Exit::Converged)
}
