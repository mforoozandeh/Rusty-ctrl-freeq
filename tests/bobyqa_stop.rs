//! Regression tests for the early-stop mechanisms the derivative-free optimisers rely on.
//!
//! The `bobyqa` crate takes no callback that can end a run.  ctrl-freeq needs to end one when the target fidelity is
//! reached or the user cancels, so its wrapper steers the solver from inside the objective: once a stop is
//! requested the objective returns a value below `f_target`, which the solver treats as success and exits on.
//!
//! This test pins that behaviour against the crate version in `Cargo.lock`.  (COBYLA is vendored with its stop
//! flag exposed; its tests live beside it in `src/optim/cobyla`.)

/// A plain 10-D sphere, `sum (x_i - 0.3)^2`.
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum()
}

/// What the wrappers track.
#[derive(Default)]
struct Track {
    evals: usize,
    best_f: f64,
    best_x: Vec<f64>,
    frozen_at: Option<usize>,
}

impl Track {
    fn new() -> Self {
        Track {
            best_f: f64::INFINITY,
            ..Default::default()
        }
    }

    /// Record a genuine evaluation and decide whether the run should stop now.
    fn record(&mut self, x: &[f64], f: f64, stop_below: f64) {
        self.evals += 1;
        if f < self.best_f {
            self.best_f = f;
            self.best_x = x.to_vec();
        }
        if self.frozen_at.is_none() && f < stop_below {
            self.frozen_at = Some(self.evals);
        }
    }
}

#[test]
fn bobyqa_stops_below_its_target() {
    let n = 10;
    let mut track = Track::new();
    let f_target = -1.0;
    let mut config = bobyqa::Config::new(n);
    config.f_target = f_target;
    config.rho_begin = 0.5;
    let mut solver = bobyqa::Bobyqa::new(n, config).expect("valid configuration");
    let mut x = vec![1.0; n];
    let outcome = solver.minimize(
        |x| {
            if track.frozen_at.is_some() {
                track.evals += 1;
                return f_target - 1.0;
            }
            let f = sphere(x);
            track.record(x, f, 1e-2);
            f
        },
        &mut x,
        &vec![-100.0; n],
        &vec![100.0; n],
    );
    let frozen_at = track.frozen_at.expect("the sphere should reach 1e-2");
    let extra = track.evals - frozen_at;
    assert!(extra <= 1, "BOBYQA kept going for {extra} evaluations after the freeze");
    assert_eq!(outcome.status, bobyqa::Status::TargetReached);
    assert!(track.best_f < 1e-2);
}
