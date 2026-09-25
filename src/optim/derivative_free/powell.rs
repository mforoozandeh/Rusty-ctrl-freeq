//! COBYLA and BOBYQA, Powell's trust-region methods.
//!
//! Both run to completion inside a single call, so the stop checks live in the objective wrapper: once a stop is
//! due, COBYLA's force-stop flag is raised, and BOBYQA is handed a value below its target, which it treats as
//! success.

use std::cell::{Cell, RefCell};

use super::{bounds, failed};
use crate::error::{Error, Result};
use crate::optim::{Derivatives, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl};

/// Evaluate, record and report one point; `Err` is remembered and ends the run.
fn step(monitor: &mut Monitor, error: &mut Option<Error>, obj: &dyn Objective, x: &[f64]) -> Option<f64> {
    match monitor.evaluate(obj, x) {
        Ok(cost) => Some(cost),
        Err(err) => {
            *error = Some(err);
            None
        }
    }
}

/// COBYLA, Powell's linear-approximation trust-region method.  Initial trust radius 1.0, as in scipy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cobyla;

impl Optimizer for Cobyla {
    fn name(&self) -> &'static str {
        "cobyla"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::None
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        let (lower, upper) = bounds(x0);
        let stop = Cell::new(0);
        let state = RefCell::new((Monitor::new(ctl), None::<Error>));
        let (status, _, _) = crate::optim::cobyla::minimize(
            |x| {
                let mut guard = state.borrow_mut();
                let (monitor, error) = &mut *guard;
                let value = step(monitor, error, obj, x);
                if monitor.exit.is_some() || error.is_some() {
                    stop.set(1);
                }
                value.unwrap_or(f64::MAX)
            },
            x0,
            &lower,
            &upper,
            1.0,
            usize::MAX,
            &stop,
        );
        let (monitor, error) = state.into_inner();
        if let Some(e) = error {
            return failed(monitor, e);
        }
        let n = monitor.evaluations;
        let exit = match status {
            crate::optim::cobyla::Status::Failed(m) => Exit::Failed(m.into()),
            crate::optim::cobyla::Status::MaxEvalReached => Exit::MaxIterations,
            _ => Exit::Converged,
        };
        monitor.finish(n, exit)
    }
}

/// BOBYQA, Powell's bound-constrained quadratic-model trust-region method.  Initial trust radius 0.5.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bobyqa;

impl Optimizer for Bobyqa {
    fn name(&self) -> &'static str {
        "bobyqa"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::None
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        let n = x0.len();
        if n < 2 {
            return Err(Error::NotSupported("BOBYQA needs at least two parameters".into()));
        }
        let (lower, upper) = bounds(x0);
        let f_target = -ctl.target_fidelity;
        let mut config = bobyqa::Config::new(n);
        config.rho_begin = 0.5;
        config.rho_end = 1e-8;
        config.f_target = f_target;
        // The monitor ends the run at `max_iter` evaluations; BOBYQA only needs a budget above its `npt`
        // initial points, and sizes its history by it.
        config.max_fun = (ctl.max_iter + 1).max(config.npt + 1);
        let mut solver = bobyqa::Bobyqa::new(n, config)
            .map_err(|s| Error::Numerical(format!("BOBYQA refused its settings: {s}")))?;
        let mut monitor = Monitor::new(ctl);
        let mut error = None;
        let mut x = x0.to_vec();
        let outcome = solver.minimize(
            |x| {
                if monitor.exit.is_some() || error.is_some() {
                    return f_target - 1.0;
                }
                let value = step(&mut monitor, &mut error, obj, x);
                if monitor.exit.is_some() || error.is_some() {
                    f_target - 1.0
                } else {
                    value.unwrap_or(f64::MAX)
                }
            },
            &mut x,
            &lower,
            &upper,
        );
        if let Some(e) = error {
            return failed(monitor, e);
        }
        let iterations = monitor.evaluations;
        let exit = match outcome.status {
            bobyqa::Status::MaxFunReached => Exit::MaxIterations,
            bobyqa::Status::InvalidArgs | bobyqa::Status::AllocationFailed => {
                Exit::Failed(format!("BOBYQA: {}", outcome.status))
            }
            _ => Exit::Converged,
        };
        monitor.finish(iterations, exit)
    }
}
