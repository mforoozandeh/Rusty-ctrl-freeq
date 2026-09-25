//! Every optimiser on the Rosenbrock function, and the stop controls they share.
//!
//! The test objective reports `cost = f` and `fidelity = 1 − f`, so reaching a target fidelity means driving the
//! Rosenbrock value below `1 − target`.

use ctrl_freeq::Result;
use ctrl_freeq::optim::{
    Eval, Exit, IterationReport, Monitor, NoProgress, Objective, ProgressSink, RecordingSink, RunControl, optimizer,
    optimizer_names,
};
use nalgebra::DMatrix;
use std::sync::Mutex;

struct Rosenbrock;

fn rosen(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
}

fn eval(f: f64) -> Eval {
    Eval {
        cost: f,
        fidelity: 1.0 - f,
        penalty: 0.0,
    }
}

impl Objective for Rosenbrock {
    fn dim(&self) -> usize {
        2
    }
    fn value(&self, x: &[f64]) -> Result<Eval> {
        Ok(eval(rosen(x)))
    }
    fn gradient(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        let (a, b) = (x[0], x[1]);
        let g = vec![-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)];
        Ok((eval(rosen(x)), g))
    }
    fn hessian(&self, x: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
        let (e, g) = self.gradient(x)?;
        let (a, b) = (x[0], x[1]);
        let h = DMatrix::from_row_slice(2, 2, &[2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a, -400.0 * a, 200.0]);
        Ok((e, g, h))
    }
}

const START: [f64; 2] = [-1.2, 1.0];

fn minimise(name: &str, max_iter: usize, target: f64, sink: &mut dyn ProgressSink) -> ctrl_freeq::optim::OptimResult {
    let opt = optimizer(name).unwrap();
    let mut ctl = RunControl {
        max_iter,
        target_fidelity: target,
        seed: 11,
        sink,
    };
    opt.minimize(&Rosenbrock, &START, &mut ctl).unwrap()
}

/// How accurate an optimiser can be expected to get on Rosenbrock, and how much budget it needs: the
/// evaluation limit, the tolerance on the objective and the tolerance on the first coordinate.
///
/// SPSA has a tier of its own.  It estimates the gradient from two evaluations along one random direction and
/// damps the step on a fixed schedule, which buys robustness to noise at the cost of a slow rate; on a curved,
/// ill-conditioned valley it settles around `5e-2` and stays there.  That is the method, not the port: a sweep
/// of the gain schedule over the ranges Spall and Qiskit recommend does not move it.
fn accuracy(name: &str) -> (usize, f64, f64) {
    match name {
        // COBYLA's linear models crawl along the Rosenbrock valley, so the derivative-free bar is lower.
        "cobyla" | "bobyqa" | "nelder-mead" | "cma-es" => (5000, 1e-3, 1e-1),
        "spsa" => (5000, 1e-1, 3.5e-1),
        _ => (500, 1e-8, 1e-3),
    }
}

/// The best fidelity the target-stop test can ask of an optimiser; see [`accuracy`].
fn reachable_target(name: &str) -> f64 {
    if name == "spsa" { 0.9 } else { 0.99 }
}

#[test]
fn every_optimiser_solves_rosenbrock() {
    for &name in optimizer_names() {
        let (budget, tol, xtol) = accuracy(name);
        let r = minimise(name, budget, 1.0, &mut RecordingSink::default());
        assert!(
            r.eval.cost < tol,
            "{name}: f = {} after {} iterations ({:?})",
            r.eval.cost,
            r.iterations,
            r.exit
        );
        assert!(
            (r.x[0] - 1.0).abs() < xtol && (r.x[1] - 1.0).abs() < 2.0 * xtol,
            "{name}: x = {:?}",
            r.x
        );
    }
}

#[test]
fn every_optimiser_stops_at_the_target() {
    for &name in optimizer_names() {
        let mut sink = RecordingSink::default();
        let target = reachable_target(name);
        let r = minimise(name, 5000, target, &mut sink);
        assert_eq!(r.exit, Exit::TargetReached, "{name}");
        assert!(r.eval.score() >= target, "{name}");
        let last = sink.reports.last().unwrap();
        assert!(
            last.fidelity >= target,
            "{name}: stopped right after the report that reached the target"
        );
    }
}

/// `cost = |x|^2`, so the origin is already a perfect solution.  Records every point it is asked about.
#[derive(Default)]
struct Quadratic {
    seen: Mutex<Vec<Vec<f64>>>,
}

impl Objective for Quadratic {
    fn dim(&self) -> usize {
        2
    }
    fn value(&self, x: &[f64]) -> Result<Eval> {
        self.seen.lock().unwrap().push(x.to_vec());
        Ok(eval(x.iter().map(|v| v * v).sum()))
    }
    fn gradient(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        Ok((self.value(x)?, x.iter().map(|v| 2.0 * v).collect()))
    }
    fn hessian(&self, x: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
        let (e, g) = self.gradient(x)?;
        Ok((e, g, DMatrix::identity(2, 2) * 2.0))
    }
}

/// A starting point that already meets the target ends the run where it stands.  Nothing may sample around it
/// first: a stochastic optimiser that calibrates or draws a population before looking at what it was given
/// throws away a solution it was handed.
#[test]
fn every_optimiser_stops_at_a_starting_point_that_already_meets_the_target() {
    for &name in optimizer_names() {
        let mut sink = RecordingSink::default();
        let opt = optimizer(name).unwrap();
        let mut ctl = RunControl {
            max_iter: 20,
            target_fidelity: 1.0,
            seed: 11,
            sink: &mut sink,
        };
        let obj = Quadratic::default();
        let r = opt.minimize(&obj, &[0.0, 0.0], &mut ctl).unwrap();
        assert_eq!(r.exit, Exit::TargetReached, "{name}: fidelity {}", r.eval.fidelity);
        assert_eq!(r.x, vec![0.0, 0.0], "{name}: came back with a point it was not given");
        assert_eq!(sink.reports.len(), 1, "{name}: no reports after the target");
        // How many evaluations a solver spends on one point is its own business - argmin's initialisation
        // asks for the value and the gradient separately.  What matters is that none of them went elsewhere.
        let seen = obj.seen.into_inner().unwrap();
        assert!(
            seen.iter().all(|x| x == &[0.0, 0.0]),
            "{name}: evaluated something other than the starting point: {seen:?}"
        );
    }
}

/// Cancelled at its very first report, a run still comes back with the point it started from.
#[test]
fn every_optimiser_cancels_on_its_first_report() {
    for &name in optimizer_names() {
        let mut sink = CancelAfter { n: 1, seen: Vec::new() };
        let r = minimise(name, 5000, 2.0, &mut sink);
        assert_eq!(r.exit, Exit::Cancelled, "{name}");
        assert_eq!(sink.seen.len(), 1, "{name}: no reports after the cancel");
        assert_eq!(r.x, START, "{name}: the first report was not the starting point");
        assert_eq!(
            r.eval.cost,
            rosen(&r.x),
            "{name}: the returned value belongs to the returned point"
        );
    }
}

/// An optimiser with nothing to vary says so, rather than failing somewhere inside its own linear algebra.
#[test]
fn the_derivative_free_optimisers_refuse_an_empty_start() {
    for name in ["nelder-mead", "spsa", "cma-es"] {
        let opt = optimizer(name).unwrap();
        let mut ctl = RunControl {
            max_iter: 20,
            target_fidelity: 1.0,
            seed: 11,
            sink: &mut NoProgress,
        };
        let err = opt
            .minimize(&Quadratic::default(), &[], &mut ctl)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one parameter"), "{name}: {err}");
    }
}

/// Cancels after a set number of reports.
struct CancelAfter {
    n: usize,
    seen: Vec<IterationReport>,
}

impl ProgressSink for CancelAfter {
    fn on_iteration(&mut self, r: &IterationReport) {
        self.seen.push(r.clone());
    }
    fn should_cancel(&self) -> bool {
        self.seen.len() >= self.n
    }
}

#[test]
fn every_optimiser_cancels_and_keeps_its_best_point() {
    for &name in optimizer_names() {
        let mut sink = CancelAfter { n: 5, seen: Vec::new() };
        let r = minimise(name, 5000, 1.0, &mut sink);
        assert_eq!(r.exit, Exit::Cancelled, "{name}");
        assert_eq!(sink.seen.len(), 5, "{name}: no reports after the cancel");
        assert!(r.eval.cost <= rosen(&START), "{name}");
        assert_eq!(
            r.eval.cost,
            rosen(&r.x),
            "{name}: the returned value belongs to the returned point"
        );
    }
}

#[test]
fn every_optimiser_respects_the_iteration_limit() {
    for &name in optimizer_names() {
        let mut sink = RecordingSink::default();
        let r = minimise(name, 20, 1.0, &mut sink);
        assert_eq!(r.exit, Exit::MaxIterations, "{name}");
        let last = sink.reports.last().unwrap().iteration;
        assert_eq!(last, 20, "{name}");
    }
}

/// A run that stops on its own convergence test says it fell short, rather than reporting plain success.
#[test]
fn stopping_short_of_the_target_says_so() {
    // Nothing on the Rosenbrock scores above 1, so no optimiser can reach this target.
    let mut converged = 0;
    for &name in optimizer_names() {
        let r = minimise(name, 5000, 1.5, &mut RecordingSink::default());
        assert!(r.eval.score() < 1.5, "{name}");
        if r.exit == Exit::Converged {
            converged += 1;
            let m = r.exit.message();
            assert!(m.contains("short of the target"), "{name}: {m}");
        } else {
            assert_eq!(r.exit, Exit::MaxIterations, "{name}");
        }
    }
    assert!(converged > 0, "no optimiser exercised the converged-short wording");
}

/// The best point can come from a line-search trial no iteration reported, so the exit follows the point.
#[test]
fn a_best_point_that_meets_the_target_is_not_called_converged() {
    let mut sink = NoProgress;
    let mut ctl = RunControl {
        max_iter: 10,
        target_fidelity: 0.5,
        seed: 0,
        sink: &mut sink,
    };
    let mut monitor = Monitor::new(&mut ctl);
    monitor.evaluated(&[0.0, 0.0], eval(0.25));
    let r = monitor.finish(1, Exit::Converged).unwrap();
    assert_eq!(r.exit, Exit::TargetReached);
    assert_eq!(r.exit.message(), "target fidelity reached");
}

/// The optimisers that draw random numbers take them from the run's seed, so a run repeats exactly and a
/// different seed is a different run.
#[test]
fn the_seed_decides_a_stochastic_run() {
    for name in ["spsa", "cma-es"] {
        let run = |seed| {
            let opt = optimizer(name).unwrap();
            let mut sink = RecordingSink::default();
            let mut ctl = RunControl {
                max_iter: 200,
                target_fidelity: 1.0,
                seed,
                sink: &mut sink,
            };
            let r = opt.minimize(&Rosenbrock, &START, &mut ctl).unwrap();
            (r.x, sink.reports.iter().map(|r| r.cost).collect::<Vec<_>>())
        };
        assert_eq!(run(3), run(3), "{name}: the same seed gives the same run");
        assert_ne!(run(3), run(4), "{name}: a different seed gives a different run");
    }
}

#[test]
fn unknown_names_list_the_supported_ones() {
    let err = optimizer("qiskit-spsa").err().unwrap().to_string();
    assert!(
        err.contains("l-bfgs, newton-cg, newton-exact, cobyla, bobyqa, nelder-mead, spsa, cma-es"),
        "{err}"
    );
}
