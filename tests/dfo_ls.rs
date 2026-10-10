//! DFO-LS on least-squares problems with known answers.

use std::sync::Mutex;

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::optim::{
    DfoLs, Eval, Exit, NoProgress, Objective, OptimResult, Optimizer, OptimizerOptions, ProgressSink, RecordingSink,
    RunControl, optimizer,
};
use ctrl_freeq::run;
use ctrl_freeq::{Error, Result};
use nalgebra::DMatrix;

/// A least-squares problem: its residuals, the cost reported with them, and every point it was asked about.
struct LeastSquares {
    n: usize,
    r: fn(&[f64]) -> Vec<f64>,
    /// The cost reported for `x` and its residuals; `‖r‖²` unless a test says otherwise.
    cost: fn(&[f64], &[f64]) -> f64,
    seen: Mutex<Vec<Vec<f64>>>,
}

fn squared_norm(_: &[f64], r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum()
}

impl LeastSquares {
    fn new(n: usize, r: fn(&[f64]) -> Vec<f64>) -> Self {
        LeastSquares {
            n,
            r,
            cost: squared_norm,
            seen: Mutex::default(),
        }
    }
}

impl Objective for LeastSquares {
    fn dim(&self) -> usize {
        self.n
    }
    fn value(&self, x: &[f64]) -> Result<Eval> {
        Ok(self.residuals(x)?.0)
    }
    fn gradient(&self, _: &[f64]) -> Result<(Eval, Vec<f64>)> {
        Err(Error::NotSupported("residuals only".into()))
    }
    fn hessian(&self, _: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
        Err(Error::NotSupported("residuals only".into()))
    }
    fn residuals(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        self.seen.lock().unwrap().push(x.to_vec());
        let r = (self.r)(x);
        let f = (self.cost)(x, &r);
        Ok((
            Eval {
                cost: f,
                fidelity: 1.0 - f,
                penalty: 0.0,
            },
            r,
        ))
    }
}

/// Run dfo-ls with an unreachable target, so it stops on its own tests or the budget.
fn solve(obj: &dyn Objective, x0: &[f64], max_iter: usize, sink: &mut dyn ProgressSink) -> OptimResult {
    let mut ctl = RunControl {
        max_iter,
        target_fidelity: 2.0,
        seed: 5,
        sink,
    };
    optimizer("dfo-ls").unwrap().minimize(obj, x0, &mut ctl).unwrap()
}

/// `A·(x − x*)` with `A` 6 × 4 of full rank and `x* = (1, −1, 0.5, 2)`.
fn affine(x: &[f64]) -> Vec<f64> {
    const A: [[f64; 4]; 6] = [
        [1.0, 0.5, 0.0, -0.3],
        [0.2, 1.0, 0.4, 0.0],
        [0.0, -0.6, 1.0, 0.1],
        [0.3, 0.0, 0.2, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, -0.5, 0.25, -0.25],
    ];
    const X: [f64; 4] = [1.0, -1.0, 0.5, 2.0];
    A.iter()
        .map(|row| row.iter().zip(x).zip(&X).map(|((a, x), s)| a * (x - s)).sum())
        .collect()
}

/// The initial set costs `n + 1` evaluations; with the solution inside the first trust region, the first model
/// step, evaluation `n + 2`, lands on it.
#[test]
fn an_affine_problem_is_solved_by_the_first_model_step() {
    let obj = LeastSquares::new(4, affine);
    // Δ0 = 0.1·max(‖x0‖∞, 1) = 0.195, and the solution is 0.1 away.
    let x0 = [1.05, -1.05, 0.55, 1.95];
    let mut sink = RecordingSink::default();
    let r = solve(&obj, &x0, 100, &mut sink);
    let costs: Vec<f64> = sink.reports.iter().map(|r| r.cost).collect();
    assert!(costs[5] <= 1e-12, "evaluation n + 2: {costs:?}");
    assert_eq!(r.exit, Exit::Converged);
}

/// Kowalik and Osborne (Moré, Garbow and Hillstrom, problem 15): a nonzero-residual minimum.
fn kowalik_osborne(x: &[f64]) -> Vec<f64> {
    const Y: [f64; 11] = [
        0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ];
    const U: [f64; 11] = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625];
    Y.iter()
        .zip(&U)
        .map(|(y, u)| y - x[0] * (u * u + u * x[1]) / (u * u + u * x[2] + x[3]))
        .collect()
}

#[test]
fn a_nonzero_residual_problem_reaches_its_minimum() {
    let obj = LeastSquares::new(4, kowalik_osborne);
    let r = solve(&obj, &[0.25, 0.39, 0.415, 0.39], 2000, &mut NoProgress);
    let f_star = 3.0750560385e-4;
    assert!(
        (r.eval.cost - f_star).abs() <= 1e-6 * f_star,
        "f = {} after {} evaluations ({:?})",
        r.eval.cost,
        r.evaluations,
        r.exit
    );
}

#[test]
fn an_objective_without_residuals_is_refused_before_any_evaluation() {
    struct Plain;
    impl Objective for Plain {
        fn dim(&self) -> usize {
            2
        }
        fn value(&self, x: &[f64]) -> Result<Eval> {
            let f = x[0] * x[0] + x[1] * x[1];
            Ok(Eval {
                cost: f,
                fidelity: 1.0 - f,
                penalty: 0.0,
            })
        }
        fn gradient(&self, _: &[f64]) -> Result<(Eval, Vec<f64>)> {
            Err(Error::NotSupported("values only".into()))
        }
        fn hessian(&self, _: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
            Err(Error::NotSupported("values only".into()))
        }
    }
    let mut sink = RecordingSink::default();
    let mut ctl = RunControl {
        max_iter: 10,
        target_fidelity: 2.0,
        seed: 0,
        sink: &mut sink,
    };
    let err = optimizer("dfo-ls")
        .unwrap()
        .minimize(&Plain, &[0.0, 0.0], &mut ctl)
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported(_)), "{err}");
    assert!(sink.reports.is_empty());
}

/// `r(x) = 1 + x²(x² − 0.01)` from `x0 = 0`: the first two points, `0` and `±0.1`, have equal residuals whichever
/// way the first direction points, so the first model is flat.  The safety branch must improve the geometry rather
/// than shrink ρ to the end with no further evaluation.
#[test]
fn a_flat_first_model_does_not_end_the_run() {
    let obj = LeastSquares::new(1, |x| vec![1.0 + x[0] * x[0] * (x[0] * x[0] - 0.01)]);
    let r = solve(&obj, &[0.0], 500, &mut NoProgress);
    assert!(obj.seen.lock().unwrap().len() > 2);
    assert!(
        (r.x[0].abs() - 0.005f64.sqrt()).abs() < 1e-6,
        "x = {} ({:?})",
        r.x[0],
        r.exit
    );
}

fn big(x: &[f64]) -> Vec<f64> {
    vec![1e8, x[0]]
}

/// Residuals `[1e8, x]` with cost `x²`: `‖r‖² = cost + 1e16`, so differences of squared norms would round to 0.
#[test]
fn a_large_constant_residual_does_not_hide_progress() {
    let mut obj = LeastSquares::new(1, big);
    obj.cost = |x, _| x[0] * x[0];
    let r = solve(&obj, &[1.0], 1000, &mut NoProgress);
    assert_eq!(r.exit, Exit::Converged);
    assert!(r.x[0].abs() <= 1e-12, "x = {}", r.x[0]);
    let seen = obj.seen.into_inner().unwrap();
    assert!(seen.len() < 50, "{} evaluations", seen.len());
    for (i, a) in seen.iter().enumerate() {
        assert!(!seen[..i].contains(a), "{a:?} evaluated twice");
    }
}

/// The same residuals with cost `‖r‖²`, which rounds to `1e16` for every `|x| ≤ 1`: the reported costs tie, yet the
/// returned parameters are the minimiser.
#[test]
fn the_returned_point_is_the_best_by_residuals_when_costs_tie() {
    let obj = LeastSquares::new(1, big);
    let r = solve(&obj, &[1.0], 1000, &mut NoProgress);
    assert!(r.x[0].abs() <= 1e-12, "x = {}", r.x[0]);
}

/// Review focus: a propagation error after some evaluations ends the run with the best point so far.
#[test]
fn an_objective_failure_keeps_the_best_point() {
    struct Failing(Mutex<usize>);
    impl Objective for Failing {
        fn dim(&self) -> usize {
            2
        }
        fn value(&self, x: &[f64]) -> Result<Eval> {
            Ok(self.residuals(x)?.0)
        }
        fn gradient(&self, _: &[f64]) -> Result<(Eval, Vec<f64>)> {
            Err(Error::NotSupported("residuals only".into()))
        }
        fn hessian(&self, _: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
            Err(Error::NotSupported("residuals only".into()))
        }
        fn residuals(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
            let mut calls = self.0.lock().unwrap();
            *calls += 1;
            if *calls > 6 {
                return Err(Error::Numerical("the propagation failed".into()));
            }
            let r = vec![x[0] - 1.0, x[1] - 2.0];
            let f = r[0] * r[0] + r[1] * r[1];
            Ok((
                Eval {
                    cost: f,
                    fidelity: 1.0 - f,
                    penalty: 0.0,
                },
                r,
            ))
        }
    }
    let r = solve(&Failing(Mutex::new(0)), &[0.0, 0.0], 100, &mut NoProgress);
    assert!(
        matches!(r.exit, Exit::Failed(ref m) if m.contains("propagation failed")),
        "{:?}",
        r.exit
    );
    assert_eq!(r.evaluations, 6);
    assert!(r.eval.cost < 5.0, "the best point is not the start: {:?}", r.eval);
}

/// The extended Rosenbrock function in ten variables, as residuals.
fn extended_rosenbrock(x: &[f64]) -> Vec<f64> {
    x.chunks(2)
        .flat_map(|p| [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        .collect()
}

fn run_with(ndirs: Option<usize>, obj: &dyn Objective, x0: &[f64], max_iter: usize) -> OptimResult {
    let mut sink = NoProgress;
    let mut ctl = RunControl {
        max_iter,
        target_fidelity: 2.0,
        seed: 5,
        sink: &mut sink,
    };
    DfoLs { ndirs_initial: ndirs }.minimize(obj, x0, &mut ctl).unwrap()
}

#[test]
fn a_reduced_initial_set_reaches_the_full_sets_accuracy() {
    let x0: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    for ndirs in [None, Some(1)] {
        let r = run_with(ndirs, &LeastSquares::new(10, extended_rosenbrock), &x0, 3000);
        assert!(
            r.eval.cost <= 1e-8,
            "{ndirs:?}: f = {} after {} evaluations ({:?})",
            r.eval.cost,
            r.evaluations,
            r.exit
        );
    }
}

/// Review focus: `ndirs_initial` at or above the parameter count is the full set, evaluation for evaluation.
#[test]
fn a_large_ndirs_initial_is_the_full_set() {
    let x0 = [1.05, -1.05, 0.55, 1.95];
    let full = LeastSquares::new(4, affine);
    let large = LeastSquares::new(4, affine);
    run_with(None, &full, &x0, 50);
    run_with(Some(100), &large, &x0, 50);
    assert_eq!(full.seen.into_inner().unwrap(), large.seen.into_inner().unwrap());
}

fn single_qubit() -> Config {
    let mut cfg = Config::from_json(examples()[0].1).unwrap();
    cfg.seed = Some(7);
    cfg.optimization.algorithm = "dfo-ls".into();
    cfg
}

/// Review focus: one drift snapshot and 16 coefficients give 7 residuals for 16 parameters.
#[test]
fn fewer_residuals_than_parameters_still_converge() {
    let mut cfg = single_qubit();
    cfg.optimization.h0_snapshots = 1;
    cfg.optimization.max_iter = 400;
    let r = run(&cfg, &mut NoProgress).unwrap();
    assert_eq!(
        r.exit,
        Exit::TargetReached,
        "F = {} after {} evaluations",
        r.fidelity,
        r.evaluations
    );
}

/// Review focus: with a tenth of the Rabi frequency an inversion needs, the amplitude limit binds and the
/// non-smooth penalty residual is active; the run still ends normally.
#[test]
fn a_binding_amplitude_limit_ends_normally() {
    let mut cfg = single_qubit();
    cfg.parameters.omega_r_max = vec![0.25e6];
    cfg.optimization.h0_snapshots = 5;
    cfg.optimization.max_iter = 300;
    let mut sink = RecordingSink::default();
    let r = run(&cfg, &mut sink).unwrap();
    assert!(
        sink.reports.iter().any(|e| e.penalty > 0.0),
        "the limit never bound, so the penalty residual went untested"
    );
    assert!(!matches!(r.exit, Exit::Failed(_)), "{:?}", r.exit);
    assert!(r.fidelity.is_finite() && r.penalty.is_finite());
}

/// A reduced initial set on a problem whose residuals span fewer directions than the parameters: once the model
/// explains `r_k` inside the explored span, every step stays there.  Without a forced new direction after such a
/// failed step the run cycled through the same points to the budget (seeds 1, 3 and 5 with one direction, 2 and 5
/// with three).
#[test]
fn a_reduced_set_does_not_stall_inside_its_span() {
    for (ndirs, seed) in [(1, 1), (1, 3), (1, 5), (3, 2), (3, 5)] {
        let mut cfg = Config::from_json(
            examples()
                .iter()
                .find(|e| e.0 == "single_qubit_parameters_multiple_initial_targ")
                .unwrap()
                .1,
        )
        .unwrap();
        cfg.seed = Some(seed);
        cfg.optimization.algorithm = "dfo-ls".into();
        cfg.optimization.max_iter = 200;
        cfg.optimization.optimizer_options = Some(OptimizerOptions {
            ndirs_initial: Some(ndirs),
        });
        let r = run(&cfg, &mut NoProgress).unwrap();
        assert_eq!(
            r.exit,
            Exit::TargetReached,
            "ndirs {ndirs}, seed {seed}: F = {} after {} evaluations",
            r.fidelity,
            r.evaluations
        );
    }
}

/// `r(x) = [10⁶(x₀ − 200), x₁ − 5]` from the origin, in the box `[−100, 100]²`: the constrained minimum is
/// `(100, 5)`.  Once `x₀` sits on its bound, its gradient points out of the box and is a hundred million times the
/// other's; the trust-region step must still move `x₁`.
#[test]
fn a_large_gradient_at_a_bound_does_not_stop_the_other_coordinates() {
    for seed in 0..5 {
        let obj = LeastSquares::new(2, |x| vec![1e6 * (x[0] - 200.0), x[1] - 5.0]);
        let mut sink = NoProgress;
        let mut ctl = RunControl {
            max_iter: 2000,
            target_fidelity: 2.0,
            seed,
            sink: &mut sink,
        };
        let r = optimizer("dfo-ls")
            .unwrap()
            .minimize(&obj, &[0.0, 0.0], &mut ctl)
            .unwrap();
        assert!(
            (r.x[0] - 100.0).abs() < 1e-9 && (r.x[1] - 5.0).abs() < 1e-6,
            "seed {seed}: x = {:?} after {} evaluations ({:?})",
            r.x,
            r.evaluations,
            r.exit
        );
    }
}
