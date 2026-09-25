//! Optimisation: the objective and optimiser interfaces, the optimiser registry, and the algorithms.
//!
//! Every optimiser minimises an [`Objective`] through the same [`Optimizer`] trait and reports to the same
//! [`ProgressSink`], which can also cancel the run.  All of them stop when the target fidelity is reached, return the
//! best point they evaluated, and never see threads: parallelism lives inside the objective.
//!
//! A new algorithm is one type implementing [`Optimizer`] plus one entry in [`optimizer`] and [`optimizer_names`];
//! [`Monitor`] gives it progress reports, cancellation, the target check and best-point tracking.

mod cobyla;
mod derivative_free;
mod gradient;
mod newton_cg;

pub use derivative_free::{Bobyqa, CmaEs, Cobyla, NelderMead, Spsa};
pub use gradient::{LBfgs, NewtonExact};
pub use newton_cg::NewtonCg;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::time::Instant;

/// One evaluation of the objective.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Eval {
    /// What is minimised: `penalty − fidelity`.
    pub cost: f64,
    /// Mean fidelity over the batch.
    pub fidelity: f64,
    /// Amplitude penalty.
    pub penalty: f64,
}

impl Eval {
    /// `fidelity − penalty`, the quantity compared with the target fidelity.
    pub fn score(&self) -> f64 {
        self.fidelity - self.penalty
    }
}

/// A function to minimise.
pub trait Objective: Sync {
    /// Number of parameters.
    fn dim(&self) -> usize;
    /// The objective at `x`.
    fn value(&self, x: &[f64]) -> Result<Eval>;
    /// The objective and its gradient at `x`.
    fn gradient(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)>;
    /// The objective, its gradient and its Hessian at `x`.
    fn hessian(&self, x: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)>;
    /// The Hessian at `x` applied to `v`.  The default forms the whole Hessian; objectives that can do better
    /// override it.
    fn hvp(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>> {
        let (_, _, h) = self.hessian(x)?;
        Ok((h * nalgebra::DVector::from_column_slice(v)).as_slice().to_vec())
    }
}

/// What an optimiser asks the objective for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivatives {
    /// Values only.
    None,
    /// Values and gradients.
    Gradient,
    /// Values, gradients and Hessians.
    Hessian,
}

/// One row of progress.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterationReport {
    /// Iteration number; `0` is the starting point for gradient methods.  For derivative-free methods an
    /// iteration is one function evaluation, counted from 1.
    pub iteration: usize,
    /// Mean fidelity.
    pub fidelity: f64,
    /// Amplitude penalty.
    pub penalty: f64,
    /// `penalty − fidelity`.
    pub cost: f64,
    /// Gradient norm, for gradient methods.
    pub grad_norm: Option<f64>,
    /// Objective evaluations so far.
    pub evaluations: usize,
    /// Seconds since the optimiser started.
    pub elapsed_s: f64,
}

/// Something that watches an optimisation and can stop it.
pub trait ProgressSink {
    /// Called once per iteration.
    fn on_iteration(&mut self, report: &IterationReport);
    /// Asked after every report.  Returning `true` ends the run with [`Exit::Cancelled`], keeping the best point.
    fn should_cancel(&self) -> bool {
        false
    }
}

/// A sink that ignores everything.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn on_iteration(&mut self, _: &IterationReport) {}
}

/// A sink that keeps every report.
#[derive(Clone, Debug, Default)]
pub struct RecordingSink {
    /// Every report, in order.
    pub reports: Vec<IterationReport>,
}

impl ProgressSink for RecordingSink {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.reports.push(report.clone());
    }
}

/// A sink that prints an iteration table to standard output, every `every`-th row.
#[derive(Clone, Copy, Debug)]
pub struct ConsoleSink {
    /// Print one row in this many; 1 prints them all.
    pub every: usize,
    header: bool,
}

impl ConsoleSink {
    /// Print every `every`-th iteration.
    pub fn new(every: usize) -> Self {
        ConsoleSink {
            every: every.max(1),
            header: false,
        }
    }
}

impl ProgressSink for ConsoleSink {
    fn on_iteration(&mut self, r: &IterationReport) {
        if !self.header {
            println!(
                "{:>9}  {:>10}  {:>10}  {:>10}  {:>9}",
                "iteration", "fidelity", "penalty", "|grad|", "time (s)"
            );
            self.header = true;
        }
        if r.iteration.is_multiple_of(self.every) {
            let g = r.grad_norm.map_or("-".to_string(), |g| format!("{g:.3e}"));
            println!(
                "{:>9}  {:>10.6}  {:>10.3e}  {:>10}  {:>9.2}",
                r.iteration, r.fidelity, r.penalty, g, r.elapsed_s
            );
        }
    }
}

/// Why an optimisation ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Exit {
    /// Fidelity minus penalty reached the target.
    TargetReached,
    /// The optimiser met its own convergence test.
    Converged,
    /// The iteration limit was reached.
    MaxIterations,
    /// The progress sink asked to stop.
    Cancelled,
    /// The optimiser or the objective failed; the best point so far is kept.
    Failed(String),
}

impl Exit {
    /// A short sentence for display.
    pub fn message(&self) -> String {
        match self {
            Exit::TargetReached => "target fidelity reached".into(),
            Exit::Converged => "converged".into(),
            Exit::MaxIterations => "iteration limit reached".into(),
            Exit::Cancelled => "cancelled".into(),
            Exit::Failed(m) => format!("stopped: {m}"),
        }
    }
}

/// What an optimisation produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimResult {
    /// The best parameters evaluated.
    pub x: Vec<f64>,
    /// The objective there.
    pub eval: Eval,
    /// Iterations completed.
    pub iterations: usize,
    /// Objective evaluations.
    pub evaluations: usize,
    /// Why the run ended.
    pub exit: Exit,
}

/// Run settings shared by every optimiser.
pub struct RunControl<'a> {
    /// Iteration limit (function evaluations for derivative-free methods).
    pub max_iter: usize,
    /// Stop once fidelity minus penalty reaches this.
    pub target_fidelity: f64,
    /// Seed for the optimisers that draw random numbers, so their runs repeat exactly.  The deterministic
    /// optimisers ignore it.
    pub seed: u64,
    /// Where progress goes, and where cancellation comes from.
    pub sink: &'a mut dyn ProgressSink,
}

/// An optimisation algorithm.
pub trait Optimizer: Send + Sync {
    /// The name the configuration uses.
    fn name(&self) -> &'static str;
    /// What it needs from the objective.
    fn derivatives(&self) -> Derivatives;
    /// Minimise `obj` from `x0`.
    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult>;
}

/// The optimiser with configuration name `name`.
pub fn optimizer(name: &str) -> Result<Box<dyn Optimizer>> {
    Ok(match name {
        "l-bfgs" => Box::new(LBfgs),
        "newton-cg" => Box::new(NewtonCg),
        "newton-exact" => Box::new(NewtonExact),
        "cobyla" => Box::new(Cobyla),
        "bobyqa" => Box::new(Bobyqa),
        "nelder-mead" => Box::new(NelderMead),
        "spsa" => Box::new(Spsa),
        "cma-es" => Box::new(CmaEs),
        _ => {
            return Err(Error::NotSupported(format!(
                "algorithm \"{name}\" is not available; choose one of {}",
                optimizer_names().join(", ")
            )));
        }
    })
}

/// Every optimiser name, in the order the interface lists them.
pub fn optimizer_names() -> &'static [&'static str] {
    &[
        "l-bfgs",
        "newton-cg",
        "newton-exact",
        "cobyla",
        "bobyqa",
        "nelder-mead",
        "spsa",
        "cma-es",
    ]
}

/// Bookkeeping every optimiser shares: progress reports, the stop checks and the best point.
pub struct Monitor<'s> {
    sink: &'s mut dyn ProgressSink,
    target: f64,
    started: Instant,
    /// The evaluation budget, for optimisers that have to divide it up in advance.
    pub max_iter: usize,
    /// The run's seed, for optimisers that draw random numbers.
    pub seed: u64,
    /// Objective evaluations so far.
    pub evaluations: usize,
    /// The best point evaluated and its value.
    pub best: Option<(Vec<f64>, Eval)>,
    /// Set once a stop condition has fired.
    pub exit: Option<Exit>,
}

impl<'s> Monitor<'s> {
    /// A monitor for `ctl`'s limits and sink.
    pub fn new(ctl: &'s mut RunControl) -> Monitor<'s> {
        Monitor {
            target: ctl.target_fidelity,
            max_iter: ctl.max_iter,
            seed: ctl.seed,
            sink: &mut *ctl.sink,
            started: Instant::now(),
            evaluations: 0,
            best: None,
            exit: None,
        }
    }

    /// Count an evaluation and keep it if it is the best so far.
    pub fn evaluated(&mut self, x: &[f64], e: Eval) {
        self.evaluations += 1;
        if self.best.as_ref().is_none_or(|(_, b)| e.cost < b.cost) {
            self.best = Some((x.to_vec(), e));
        }
    }

    /// Evaluate `obj` at `x`, count it, keep it if it is the best, and report it as one iteration.  This is the
    /// derivative-free convention: one iteration is one function evaluation.  The caller stops as soon as
    /// [`exit`](Self::exit) is set.
    pub fn evaluate(&mut self, obj: &dyn Objective, x: &[f64]) -> Result<f64> {
        let e = obj.value(x)?;
        self.evaluated(x, e);
        let n = self.evaluations;
        self.report(n, e, None);
        Ok(e.cost)
    }

    /// Send a report, then check the stop conditions.  `iteration` is compared with the limit.
    pub fn report(&mut self, iteration: usize, e: Eval, grad_norm: Option<f64>) {
        self.sink.on_iteration(&IterationReport {
            iteration,
            fidelity: e.fidelity,
            penalty: e.penalty,
            cost: e.cost,
            grad_norm,
            evaluations: self.evaluations,
            elapsed_s: self.started.elapsed().as_secs_f64(),
        });
        if self.exit.is_some() {
            return;
        }
        if e.score() >= self.target {
            self.exit = Some(Exit::TargetReached);
        } else if self.sink.should_cancel() {
            self.exit = Some(Exit::Cancelled);
        } else if iteration >= self.max_iter {
            self.exit = Some(Exit::MaxIterations);
        }
    }

    /// The result: the best point, with `exit` unless a stop condition fired first.
    pub fn finish(self, iterations: usize, exit: Exit) -> Result<OptimResult> {
        let exit = self.exit.unwrap_or(exit);
        match self.best {
            Some((x, eval)) => Ok(OptimResult {
                x,
                eval,
                iterations,
                evaluations: self.evaluations,
                exit,
            }),
            None => Err(match exit {
                Exit::Failed(m) => Error::Numerical(m),
                other => Error::Numerical(format!(
                    "the optimiser stopped ({}) before any evaluation",
                    other.message()
                )),
            }),
        }
    }
}
