//! L-BFGS and the trust-region Newton method, from [argmin](https://argmin-rs.org).
//!
//! argmin runs a solver one iteration at a time and asks it whether to stop, so a thin [`Monitored`] wrapper around
//! any argmin solver gives it progress reports, the target check and cancellation - the pattern Rusty-QOALA uses.
//! [`Adapter`] presents an [`Objective`] to argmin and records every evaluation, so the best point ever evaluated,
//! line-search trials included, is what a run returns.

use std::cell::RefCell;

use argmin::core::{
    CostFunction, Error as ArgminError, Executor, Gradient, Hessian, IterState, KV, Problem, Solver, State,
    TerminationReason, TerminationStatus,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};

use super::{Derivatives, Eval, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl};
use crate::error::{Error, Result};

/// L-BFGS history length.
const LBFGS_MEMORY: usize = 10;

type Iterate<H> = IterState<DVector<f64>, DVector<f64>, (), H, (), f64>;
type LineSearch = MoreThuenteLineSearch<DVector<f64>, DVector<f64>, f64>;

/// An [`Objective`] as an argmin problem.
struct Adapter<'a, 's> {
    obj: &'a dyn Objective,
    monitor: &'a RefCell<Monitor<'s>>,
}

fn argmin_error(e: Error) -> ArgminError {
    ArgminError::msg(e.to_string())
}

impl Adapter<'_, '_> {
    fn record(&self, x: &DVector<f64>, e: Eval) {
        self.monitor.borrow_mut().evaluated(x.as_slice(), e);
    }
}

impl CostFunction for Adapter<'_, '_> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> std::result::Result<f64, ArgminError> {
        let e = self.obj.value(x.as_slice()).map_err(argmin_error)?;
        self.record(x, e);
        Ok(e.cost)
    }
}

impl Gradient for Adapter<'_, '_> {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, x: &Self::Param) -> std::result::Result<DVector<f64>, ArgminError> {
        let (e, g) = self.obj.gradient(x.as_slice()).map_err(argmin_error)?;
        self.record(x, e);
        Ok(DVector::from_vec(g))
    }
}

impl Hessian for Adapter<'_, '_> {
    type Param = DVector<f64>;
    type Hessian = DMatrix<f64>;

    fn hessian(&self, x: &Self::Param) -> std::result::Result<DMatrix<f64>, ArgminError> {
        let (e, _, h) = self.obj.hessian(x.as_slice()).map_err(argmin_error)?;
        self.record(x, e);
        Ok(h)
    }
}

/// Any argmin solver with reporting and the shared stop checks around it.
struct Monitored<'a, 's, S> {
    inner: S,
    monitor: &'a RefCell<Monitor<'s>>,
    iteration: usize,
    failure: Option<String>,
}

impl<'a, 's, S> Monitored<'a, 's, S> {
    /// Report the state's current point as iteration `self.iteration`.
    fn report<H>(
        &mut self,
        problem: &Problem<Adapter<'a, 's>>,
        state: &Iterate<H>,
    ) -> std::result::Result<(), ArgminError> {
        let (Some(adapter), Some(x)) = (problem.problem.as_ref(), state.get_param()) else {
            return Ok(());
        };
        // Usually the point just evaluated, so a cache hit in the objective.
        let e = adapter.obj.value(x.as_slice()).map_err(argmin_error)?;
        let grad_norm = state.get_gradient().map(|g| g.norm());
        self.monitor.borrow_mut().report(self.iteration, e, grad_norm);
        Ok(())
    }
}

impl<'a, 's, S, H> Solver<Adapter<'a, 's>, Iterate<H>> for Monitored<'a, 's, S>
where
    S: Solver<Adapter<'a, 's>, Iterate<H>>,
    H: Clone,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn init(
        &mut self,
        problem: &mut Problem<Adapter<'a, 's>>,
        state: Iterate<H>,
    ) -> std::result::Result<(Iterate<H>, Option<KV>), ArgminError> {
        let (state, kv) = self.inner.init(problem, state)?;
        self.report(problem, &state)?;
        Ok((state, kv))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Adapter<'a, 's>>,
        state: Iterate<H>,
    ) -> std::result::Result<(Iterate<H>, Option<KV>), ArgminError> {
        // A solver that fails mid-iteration consumes the state; keep a copy so the run ends on the last good point.
        let fallback = state.clone();
        match self.inner.next_iter(problem, state) {
            Ok((state, kv)) => {
                self.iteration += 1;
                self.report(problem, &state)?;
                Ok((state, kv))
            }
            Err(e) => {
                let message = e.to_string();
                self.failure = Some(message.clone());
                Ok((fallback.terminate_with(TerminationReason::SolverExit(message)), None))
            }
        }
    }

    fn terminate(&mut self, state: &Iterate<H>) -> TerminationStatus {
        if self.monitor.borrow().exit.is_some() {
            return TerminationStatus::Terminated(TerminationReason::SolverExit("monitor".into()));
        }
        self.inner.terminate(state)
    }
}

/// Run `solver` on `obj` from `x0` with the shared monitoring.
fn run<S, H>(solver: S, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult>
where
    for<'a, 's> S: Solver<Adapter<'a, 's>, Iterate<H>>,
    H: Clone,
{
    let max_iters = ctl.max_iter as u64;
    let monitor = RefCell::new(Monitor::new(ctl));
    let adapter = Adapter { obj, monitor: &monitor };
    let wrapped = Monitored {
        inner: solver,
        monitor: &monitor,
        iteration: 0,
        failure: None,
    };
    let outcome = Executor::new(adapter, wrapped)
        .configure(|state| state.param(DVector::from_column_slice(x0)).max_iters(max_iters))
        .run();
    let (iterations, exit) = match outcome {
        Ok(result) => {
            let failure = result.solver.failure.clone();
            let iterations = result.solver.iteration;
            let exit = match (failure, result.state.get_termination_reason()) {
                (Some(m), _) => Exit::Failed(m),
                (None, Some(TerminationReason::MaxItersReached)) => Exit::MaxIterations,
                (None, Some(TerminationReason::Interrupt)) => Exit::Cancelled,
                _ => Exit::Converged,
            };
            (iterations, exit)
        }
        Err(e) => (0, Exit::Failed(e.to_string())),
    };
    monitor.into_inner().finish(iterations, exit)
}

/// Limited-memory BFGS with a Moré–Thuente line search.
#[derive(Clone, Copy, Debug, Default)]
pub struct LBfgs;

impl Optimizer for LBfgs {
    fn name(&self) -> &'static str {
        "l-bfgs"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::Gradient
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        let solver: LBFGS<LineSearch, DVector<f64>, DVector<f64>, f64> =
            LBFGS::new(MoreThuenteLineSearch::new(), LBFGS_MEMORY);
        run::<_, ()>(solver, obj, x0, ctl)
    }
}

/// Newton's method in a trust region on the exact Hessian, with the subproblem solved by Steihaug's truncated
/// conjugate gradients, so indefinite Hessians are handled.
#[derive(Clone, Copy, Debug, Default)]
pub struct NewtonExact;

impl Optimizer for NewtonExact {
    fn name(&self) -> &'static str {
        "newton-exact"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::Hessian
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        let solver: TrustRegion<Steihaug<DVector<f64>, f64>, f64> = TrustRegion::new(Steihaug::new());
        run::<_, DMatrix<f64>>(solver, obj, x0, ctl)
    }
}
