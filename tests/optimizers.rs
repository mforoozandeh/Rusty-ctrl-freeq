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
        sink,
    };
    opt.minimize(&Rosenbrock, &START, &mut ctl).unwrap()
}

fn is_gradient_based(name: &str) -> bool {
    !matches!(name, "cobyla" | "bobyqa")
}

#[test]
fn every_optimiser_solves_rosenbrock() {
    for &name in optimizer_names() {
        // COBYLA's linear models crawl along the Rosenbrock valley, so the derivative-free bar is lower.
        let (budget, tol, xtol) = if is_gradient_based(name) {
            (500, 1e-8, 1e-3)
        } else {
            (5000, 1e-3, 1e-1)
        };
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
        let r = minimise(name, 5000, 0.99, &mut sink);
        assert_eq!(r.exit, Exit::TargetReached, "{name}");
        assert!(r.eval.score() >= 0.99, "{name}");
        let last = sink.reports.last().unwrap();
        assert!(
            last.fidelity >= 0.99,
            "{name}: stopped right after the report that reached the target"
        );
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
        sink: &mut sink,
    };
    let mut monitor = Monitor::new(&mut ctl);
    monitor.evaluated(&[0.0, 0.0], eval(0.25));
    let r = monitor.finish(1, Exit::Converged).unwrap();
    assert_eq!(r.exit, Exit::TargetReached);
    assert_eq!(r.exit.message(), "target fidelity reached");
}

#[test]
fn unknown_names_list_the_supported_ones() {
    let err = optimizer("qiskit-spsa").err().unwrap().to_string();
    assert!(err.contains("l-bfgs, newton-cg, newton-exact, cobyla, bobyqa"), "{err}");
}
