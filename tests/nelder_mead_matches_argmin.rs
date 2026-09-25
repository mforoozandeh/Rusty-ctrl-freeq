//! ctrl-freeq's Nelder-Mead against argmin's, from the same initial simplex.
//!
//! argmin ships a Nelder-Mead this crate could have used instead.  It was not used, because wiring it to the
//! shared monitoring would mean generalising `Adapter`/`Monitored` over argmin's gradient type and paying an
//! extra objective evaluation per iteration for the progress report.  This test is what makes that choice
//! safe: handed the same starting simplex, the two implementations walk the same path.
//!
//! They are not compared bit for bit.  The centroid here is summed and divided once, where argmin multiplies
//! by a rounded `1/n`, so the two drift by an ulp or so whenever `n` is not a power of two.

use std::sync::Mutex;

use argmin::core::{CostFunction, Error as ArgminError, Executor};
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use ctrl_freeq::Result;
use ctrl_freeq::optim::{Eval, NoProgress, Objective, RunControl, optimizer};
use nalgebra::{DMatrix, DVector};

/// An objective that records every point it is asked about, in order.
struct Recorder {
    f: fn(&[f64]) -> f64,
    n: usize,
    seen: Mutex<Vec<Vec<f64>>>,
}

impl Recorder {
    fn new(f: fn(&[f64]) -> f64, n: usize) -> Recorder {
        Recorder {
            f,
            n,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn call(&self, x: &[f64]) -> f64 {
        self.seen.lock().unwrap().push(x.to_vec());
        (self.f)(x)
    }

    fn into_seen(self) -> Vec<Vec<f64>> {
        self.seen.into_inner().unwrap()
    }
}

impl Objective for Recorder {
    fn dim(&self) -> usize {
        self.n
    }
    fn value(&self, x: &[f64]) -> Result<Eval> {
        let f = self.call(x);
        Ok(Eval {
            cost: f,
            fidelity: -f,
            penalty: 0.0,
        })
    }
    fn gradient(&self, _: &[f64]) -> Result<(Eval, Vec<f64>)> {
        unimplemented!("Nelder-Mead asks for values only")
    }
    fn hessian(&self, _: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
        unimplemented!("Nelder-Mead asks for values only")
    }
}

impl CostFunction for Recorder {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> std::result::Result<f64, ArgminError> {
        Ok(self.call(x.as_slice()))
    }
}

fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
}

fn beale(x: &[f64]) -> f64 {
    (1.5 - x[0] + x[0] * x[1]).powi(2)
        + (2.25 - x[0] + x[0] * x[1] * x[1]).powi(2)
        + (2.625 - x[0] + x[0] * x[1].powi(3)).powi(2)
}

/// Deliberately five-dimensional: the centroid then divides by five, which is where the two implementations
/// round differently.
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum()
}

fn powell(x: &[f64]) -> f64 {
    (x[0] + 10.0 * x[1]).powi(2)
        + 5.0 * (x[2] - x[3]).powi(2)
        + (x[1] - 2.0 * x[2]).powi(4)
        + 10.0 * (x[0] - x[3]).powi(4)
}

/// The initial simplex ctrl-freeq builds: each coordinate of the starting point moved 5%, or to `0.00025`
/// where it is zero.  argmin takes its simplex as an argument, so it can be handed this one.
fn simplex(x0: &[f64]) -> Vec<DVector<f64>> {
    let mut s = vec![DVector::from_column_slice(x0)];
    for i in 0..x0.len() {
        let mut v = x0.to_vec();
        v[i] = if v[i] == 0.0 { 0.00025 } else { v[i] * 1.05 };
        s.push(DVector::from_vec(v));
    }
    s
}

/// Two points are the same step of the same search if they agree to a relative `1e-9`.
fn same_point(a: &[f64], b: &[f64]) -> bool {
    a.iter()
        .zip(b)
        .all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())))
}

/// One case: a name, the function and where to start it.
type Case = (&'static str, fn(&[f64]) -> f64, &'static [f64]);

#[test]
fn the_two_implementations_walk_the_same_path() {
    let cases: [Case; 4] = [
        ("rosenbrock", rosenbrock, &[-1.2, 1.0]),
        ("beale", beale, &[1.0, 1.0]),
        ("sphere-5d", sphere, &[1.0, -2.0, 3.0, -4.0, 5.0]),
        ("powell-4d", powell, &[3.0, -1.0, 0.0, 1.0]),
    ];

    for (name, f, x0) in cases {
        let ours = Recorder::new(f, x0.len());
        // A target no value of these objectives can reach, so the run ends on its own convergence test.
        let mut ctl = RunControl {
            max_iter: 100_000,
            target_fidelity: 2.0,
            seed: 0,
            sink: &mut NoProgress,
        };
        let result = optimizer("nelder-mead").unwrap().minimize(&ours, x0, &mut ctl).unwrap();
        let ours = ours.into_seen();

        let theirs = Recorder::new(f, x0.len());
        let solver: ArgminNelderMead<DVector<f64>, f64> = ArgminNelderMead::new(simplex(x0));
        let outcome = Executor::new(theirs, solver)
            .configure(|s| s.max_iters(100_000))
            .run()
            .unwrap();
        let theirs_cost = outcome.state.best_cost;
        let theirs = outcome.problem.problem.unwrap().into_seen();

        // argmin stops on the standard deviation of the vertex costs and this crate on the spread of the
        // simplex, so one run outlasts the other; every evaluation they share must be the same one.
        let shared = ours.len().min(theirs.len());
        assert!(shared > 100, "{name}: only {shared} evaluations to compare");
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert!(
                same_point(a, b),
                "{name}: the searches part at evaluation {i}: {a:?} vs {b:?}"
            );
        }
        // Neither stopping rule should leave the other's answer out of reach.
        assert!(
            result.eval.cost <= theirs_cost.max(1e-14),
            "{name}: {} vs {theirs_cost}",
            result.eval.cost
        );
    }
}
