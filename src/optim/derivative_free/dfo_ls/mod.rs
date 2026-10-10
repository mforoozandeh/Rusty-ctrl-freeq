//! DFO-LS, a derivative-free Gauss–Newton trust-region method for nonlinear least squares.
//!
//! The method of C. Cartis, J. Fiala, B. Marteau and L. Roberts, "Improving the flexibility and robustness of
//! model-based derivative-free optimization solvers" (ACM TOMS 45(3), 2019), built on C. Cartis and L. Roberts,
//! "A derivative-free Gauss–Newton method" (Math. Prog. Comp. 11, 2019).  It keeps up to `n + 1` points, fits a
//! linear model of the residual vector through them, and takes Gauss–Newton steps in a trust region whose radius
//! Δ is floored by a second radius ρ that only shrinks once the points around `x_k` have been checked.
//!
//! This is a clean-room implementation from those published descriptions.  No source code of the DFO-LS or DFO-GN
//! packages (GPL-3.0) was consulted.  Where the papers leave a choice open, the choice here is its own:
//!
//! - the geometry threshold `2Δ`, the slow-progress window, threshold and count, and the new-direction threshold;
//! - points stored as evaluated, with no base point;
//! - in the growing phase, the left singular vectors of the fill: the part of `r_k` the model cannot explain first;
//! - the trust-region step without TRSBOX's refinement along the sphere;
//! - `f` compared through `(r_a − r_b)ᵀ(r_a + r_b)`, never by subtracting squared norms;
//! - geometry before ρ: ρ shrinks only on a set whose points are all within `2Δ` of `x_k`;
//! - a ratio of exactly 0 counts as a failed step.

mod model;
mod subproblem;

use std::collections::VecDeque;
use std::ops::ControlFlow::{self, Break, Continue};

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;

use self::model::{Build, Point, Set, normal, outside, project_out};
use self::subproblem::{maximise_linear, predicted_decrease, trust_region_step};
use super::{bounds, clamp, finish};
use crate::error::{Error, Result};
use crate::optim::{Derivatives, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl, residual_difference};
use crate::setup::Rng;

const ETA1: f64 = 0.1;
const ETA2: f64 = 0.7;
const GAMMA_DEC: f64 = 0.5;
const GAMMA_INC: f64 = 2.0;
const GAMMA_INC_BAR: f64 = 4.0;
const ALPHA1: f64 = 0.1;
const ALPHA2: f64 = 0.5;
const OMEGA_S: f64 = 0.1;
const GAMMA_S: f64 = 0.5;
const DELTA_MAX: f64 = 1e10;
const RHO_END: f64 = 1e-8;
/// The set is poor once a point is farther than this many Δ from `x_k`.
const POOR: f64 = 2.0;
/// A point adds a direction if at least this fraction of its offset lies outside the explored span.
const NEW_DIRECTION: f64 = 1e-2;
/// Random directions tried before giving up on finding one outside the span.
const DIRECTION_DRAWS: usize = 10;
const SLOW_WINDOW: usize = 20;
const SLOW_THRESHOLD: f64 = 1e-8;
const SLOW_COUNT: usize = 20;
/// Consecutive unsuccessful steps before ρ may shrink.
const UNSUCCESSFUL: usize = 3;

/// DFO-LS.  Needs an objective with [`residuals`](Objective::residuals).
#[derive(Clone, Copy, Debug, Default)]
pub struct DfoLs {
    /// Directions in the initial interpolation set.  `None`, or at least the number of parameters, is the full set
    /// of `n + 1` points; a smaller set starts stepping sooner and gains a point whenever a step, or an extra point
    /// after a failed step, opens a new direction.
    pub ndirs_initial: Option<usize>,
}

impl Optimizer for DfoLs {
    fn name(&self) -> &'static str {
        "dfo-ls"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::Residuals
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        if x0.is_empty() {
            return Err(Error::NotSupported("DFO-LS needs at least one parameter".into()));
        }
        finish(Monitor::new(ctl), |m| search(obj, x0, self.ndirs_initial, m))
    }
}

fn search(obj: &dyn Objective, x0: &[f64], ndirs: Option<usize>, monitor: &mut Monitor<'_>) -> Result<Exit> {
    let mut solver = match Solver::start(obj, x0, ndirs, monitor)? {
        Continue(solver) => solver,
        Break(exit) => return Ok(exit),
    };
    loop {
        if let Break(exit) = solver.iterate()? {
            return Ok(exit);
        }
    }
}

/// Evaluate `x`, clamped into the box.  `None` once the monitor has ended the run.
fn evaluate(
    obj: &dyn Objective,
    monitor: &mut Monitor<'_>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
    mut x: DVector<f64>,
) -> Result<Option<Point>> {
    clamp(x.as_mut_slice(), lower.as_slice(), upper.as_slice());
    let r = monitor.evaluate_residuals(obj, x.as_slice())?;
    if monitor.exit.is_some() {
        return Ok(None);
    }
    Ok(Some(Point::new(x, r)))
}

/// The monitor's own reason for ending the run.
fn monitor_exit(monitor: &Monitor<'_>) -> Exit {
    monitor.exit.clone().expect("the monitor ended the run")
}

fn no_direction() -> Exit {
    Exit::Failed("no direction outside the interpolation set's span could be found inside the box".into())
}

/// `p` random orthonormal directions in `n` dimensions, as columns.
fn random_orthonormal(n: usize, p: usize, rng: &mut Rng) -> DMatrix<f64> {
    DMatrix::from_fn(n, p, |_, _| normal(rng)).qr().q()
}

/// Whether some point is farther than `POOR·Δ` from `x_k`.
fn poor(set: &Set, delta: f64) -> bool {
    set.farthest().1 > POOR * delta
}

/// What follows an unsuccessful step or a safety step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Continue,
    Geometry,
    ReduceRho,
    /// A point along a new direction, for a growing set.
    Grow,
}

/// After an evaluated step with `ratio < η1`.  While the set is growing, a step that `added` no direction is
/// followed by a point along a new one: once the model explains `r_k` inside the explored span, its steps stay
/// there, and Δ never falls below ρ for the safety step to take over.  On a full set, a poor geometry is improved
/// first; only a set that is not poor may shrink ρ, and only once Δ has reached ρ, the step made `f` no better, and
/// enough steps in a row have failed.
fn after_step(full: bool, added: bool, poor: bool, ratio: f64, delta: f64, rho: f64, unsuccessful: usize) -> Action {
    if !full {
        if added { Action::Continue } else { Action::Grow }
    } else if poor {
        Action::Geometry
    } else if ratio <= 0.0 && delta == rho && unsuccessful >= UNSUCCESSFUL {
        Action::ReduceRho
    } else {
        Action::Continue
    }
}

/// After a step too short to evaluate, on a full set, once Δ has shrunk: geometry first, then ρ.
fn after_safety(poor: bool, delta: f64, rho: f64) -> Action {
    if poor {
        Action::Geometry
    } else if delta == rho {
        Action::ReduceRho
    } else {
        Action::Continue
    }
}

/// The trust-region radius after a step of length `step` with ratio `ratio`.
fn radius(delta: f64, rho: f64, ratio: f64, step: f64) -> f64 {
    if ratio >= ETA2 {
        (GAMMA_INC * delta).max(GAMMA_INC_BAR * step).min(DELTA_MAX)
    } else if ratio >= ETA1 {
        (GAMMA_DEC * delta).max(step).max(rho)
    } else {
        (GAMMA_DEC * delta).min(step).max(rho)
    }
}

/// ρ's next value: a tenth while far from `ρ_end`, then the geometric mean with it, then `ρ_end`.
fn next_rho(rho: f64) -> f64 {
    if rho > 250.0 * RHO_END {
        ALPHA1 * rho
    } else if rho > 16.0 * RHO_END {
        (rho * RHO_END).sqrt()
    } else {
        RHO_END
    }
}

/// The slow-progress test: over the last `SLOW_WINDOW` successful iterations `ln f` fell by less than
/// `SLOW_THRESHOLD` per iteration, `SLOW_COUNT` times in a row.
#[derive(Default)]
struct Slow {
    history: VecDeque<f64>,
    count: usize,
}

impl Slow {
    /// Record `f(x_k)` after a successful iteration; `true` once progress has been slow for long enough.
    fn record(&mut self, f: f64) -> bool {
        self.history.push_back(f.ln());
        if self.history.len() > SLOW_WINDOW + 1 {
            self.history.pop_front();
        }
        if self.history.len() == SLOW_WINDOW + 1 {
            let fall = (self.history[0] - self.history[SLOW_WINDOW]) / SLOW_WINDOW as f64;
            self.count = if fall < SLOW_THRESHOLD { self.count + 1 } else { 0 };
        }
        self.count >= SLOW_COUNT
    }
}

type Flow = Result<ControlFlow<Exit>>;

struct Solver<'a, 'm, 's> {
    obj: &'a dyn Objective,
    monitor: &'m mut Monitor<'s>,
    lower: DVector<f64>,
    upper: DVector<f64>,
    rng: Rng,
    set: Set,
    delta: f64,
    rho: f64,
    /// Consecutive unsuccessful iterations.
    unsuccessful: usize,
    slow: Slow,
    /// `f(x0)`, for the small-objective stop.
    f0: f64,
}

impl<'a, 'm, 's> Solver<'a, 'm, 's> {
    /// Evaluate `x0` and the initial directions.
    fn start(
        obj: &'a dyn Objective,
        x0: &[f64],
        ndirs: Option<usize>,
        monitor: &'m mut Monitor<'s>,
    ) -> Result<ControlFlow<Exit, Self>> {
        let n = x0.len();
        let (lower, upper) = bounds(x0);
        let (lower, upper) = (DVector::from_vec(lower), DVector::from_vec(upper));
        let delta = 0.1 * x0.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let mut rng = Rng::seed_from_u64(monitor.seed);
        let start = DVector::from_column_slice(x0);
        let Some(first) = evaluate(obj, monitor, &lower, &upper, start.clone())? else {
            return Ok(Break(monitor_exit(monitor)));
        };
        let f0 = first.f;
        let mut set = Set::new(first);
        let directions = random_orthonormal(n, ndirs.unwrap_or(n).clamp(1, n), &mut rng);
        for q in directions.column_iter() {
            let Some(point) = evaluate(obj, monitor, &lower, &upper, &start + q * delta)? else {
                return Ok(Break(monitor_exit(monitor)));
            };
            set.push(point);
        }
        Ok(Continue(Solver {
            obj,
            monitor,
            lower,
            upper,
            rng,
            set,
            delta,
            rho: delta,
            unsuccessful: 0,
            slow: Slow::default(),
            f0,
        }))
    }

    fn evaluate(&mut self, x: DVector<f64>) -> Result<Option<Point>> {
        evaluate(self.obj, &mut *self.monitor, &self.lower, &self.upper, x)
    }

    fn stopped(&self) -> Flow {
        Ok(Break(monitor_exit(&*self.monitor)))
    }

    fn full(&self) -> bool {
        self.set.len() == self.set.xk().len() + 1
    }

    /// The box relative to `x_k`.
    fn local_box(&self) -> (DVector<f64>, DVector<f64>) {
        (&self.lower - self.set.xk(), &self.upper - self.set.xk())
    }

    /// One iteration: a trust-region step, or a safety, geometry, repair or ρ step.
    fn iterate(&mut self) -> Flow {
        debug_assert_eq!(
            self.monitor.best.as_ref().map(|b| b.0.as_slice()),
            Some(self.set.xk().as_slice()),
            "x_k is the monitor's best point"
        );
        let model = match self.set.model() {
            Build::Model(model) => model,
            Build::Unpoised(dependent) => return self.repair(&dependent),
        };
        let full = self.full();
        let j = if full {
            model.j.clone()
        } else {
            model.filled(&mut self.rng)
        };
        let (lower, upper) = self.local_box();
        let s = trust_region_step(&j, &model.r, self.delta, &lower, &upper);
        let predicted = predicted_decrease(&j, &model.r, &s);
        if s.norm() < GAMMA_S * self.rho || !(predicted > 0.0 && predicted.is_finite()) {
            return if full { self.safety() } else { self.grow(&model.span) };
        }
        let xk = self.set.xk().clone();
        let Some(point) = self.evaluate(&xk + &s)? else {
            return self.stopped();
        };
        let ratio = residual_difference(model.r.as_slice(), point.r.as_slice()) / predicted;
        let d = &point.x - &xk;
        let added = !full && model.adds_direction(&d, NEW_DIRECTION);
        if added {
            self.set.push(point);
        } else {
            let keep = (!self.set.beats(&point)).then_some(self.set.k);
            let t = self.set.replacement(&model, &d, self.delta, keep);
            self.set.replace(t, point);
        }
        self.delta = radius(self.delta, self.rho, ratio, s.norm());
        if self.set.best().f <= 1e-12f64.max(1e-20 * self.f0) {
            return Ok(Break(Exit::Converged));
        }
        if ratio >= ETA1 {
            self.unsuccessful = 0;
            let slow = self.slow.record(self.set.best().f);
            return Ok(if slow { Break(Exit::Converged) } else { Continue(()) });
        }
        self.unsuccessful += 1;
        let poor = poor(&self.set, self.delta);
        match after_step(full, added, poor, ratio, self.delta, self.rho, self.unsuccessful) {
            Action::Geometry => self.geometry(),
            Action::ReduceRho => self.reduce_rho(),
            Action::Grow => {
                let span = self.set.span();
                self.grow(&span)
            }
            Action::Continue => Ok(Continue(())),
        }
    }

    /// A step too short to be worth evaluating, on a full set.
    fn safety(&mut self) -> Flow {
        self.delta = self.rho.max(OMEGA_S * self.delta);
        match after_safety(poor(&self.set, self.delta), self.delta, self.rho) {
            Action::Geometry => self.geometry(),
            Action::ReduceRho => self.reduce_rho(),
            Action::Grow | Action::Continue => Ok(Continue(())),
        }
    }

    /// Move the point farthest from `x_k` to where its Lagrange function is largest in the trust region.
    fn geometry(&mut self) -> Flow {
        let model = match self.set.model() {
            Build::Model(model) => model,
            Build::Unpoised(dependent) => return self.repair(&dependent),
        };
        let (t, _) = self.set.farthest();
        let (lower, upper) = self.local_box();
        let s = maximise_linear(&model.lagrange_gradient(t), self.delta, &lower, &upper);
        let x = self.set.xk() + s;
        let Some(point) = self.evaluate(x)? else {
            return self.stopped();
        };
        self.set.replace(t, point);
        Ok(Continue(()))
    }

    fn reduce_rho(&mut self) -> Flow {
        debug_assert!(!poor(&self.set, self.delta), "ρ shrinks only on a set that is not poor");
        if self.rho <= RHO_END {
            return Ok(Break(Exit::Converged));
        }
        let old = self.rho;
        self.rho = next_rho(old);
        self.delta = (ALPHA2 * old).max(self.rho);
        self.unsuccessful = 0;
        Ok(Continue(()))
    }

    /// The growing phase's safety step: a point along a new direction.
    fn grow(&mut self, span: &DMatrix<f64>) -> Flow {
        let Some(x) = self.new_direction(span) else {
            return Ok(Break(no_direction()));
        };
        let Some(point) = self.evaluate(x)? else {
            return self.stopped();
        };
        self.set.push(point);
        Ok(Continue(()))
    }

    /// A point `x_k + s` inside the box whose offset has at least `NEW_DIRECTION` of its length outside the span of
    /// the orthonormal columns `span`: `s` maximises `|qᵀs|` over the ball and the box for a random unit `q`
    /// orthogonal to the span.  The feasible region near `x_k` always holds an orthant, so `s` keeps at least
    /// `1/√2` of its length along `q`.
    fn new_direction(&mut self, span: &DMatrix<f64>) -> Option<DVector<f64>> {
        let n = self.set.xk().len();
        if span.ncols() >= n {
            return None;
        }
        let (lower, upper) = self.local_box();
        for _ in 0..DIRECTION_DRAWS {
            let g = DVector::from_fn(n, |_, _| normal(&mut self.rng));
            let q = project_out(span, &g);
            let length = q.norm();
            if length == 0.0 {
                continue;
            }
            let s = maximise_linear(&(q / length), self.delta, &lower, &upper);
            let mut x = self.set.xk() + s;
            clamp(x.as_mut_slice(), self.lower.as_slice(), self.upper.as_slice());
            let d = &x - self.set.xk();
            if d.norm() > 0.0 && outside(span, &d) >= NEW_DIRECTION * d.norm() {
                return Some(x);
            }
        }
        None
    }

    /// Replace the points that make the set degenerate by points along new directions, one at a time.
    fn repair(&mut self, dependent: &[usize]) -> Flow {
        self.set.remove(dependent);
        for _ in dependent {
            let span = self.set.span();
            let Some(x) = self.new_direction(&span) else {
                return Ok(Break(no_direction()));
            };
            let Some(point) = self.evaluate(x)? else {
                return self.stopped();
            };
            self.set.push(point);
        }
        if let Build::Unpoised(_) = self.set.model() {
            return Ok(Break(Exit::Failed(
                "the interpolation set stays degenerate after a repair".into(),
            )));
        }
        Ok(Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::{Eval, NoProgress};

    /// `r(x) = [x₀ − 1, x₁ − 2, x₂ − 3, 1]`: minimised at `(1, 2, 3)` with `f = 1`, so the run ends by shrinking ρ.
    struct Shifted;

    impl Objective for Shifted {
        fn dim(&self) -> usize {
            3
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
            let r = vec![x[0] - 1.0, x[1] - 2.0, x[2] - 3.0, 1.0];
            let f: f64 = r.iter().map(|v| v * v).sum();
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

    #[test]
    fn a_poor_set_is_improved_before_rho_shrinks() {
        assert_eq!(after_step(true, false, true, -1.0, 0.1, 0.1, 5), Action::Geometry);
        assert_eq!(after_step(true, false, false, -1.0, 0.1, 0.1, 5), Action::ReduceRho);
        assert_eq!(
            after_step(true, false, false, 0.0, 0.1, 0.1, 3),
            Action::ReduceRho,
            "a tie counts as failure"
        );
        assert_eq!(
            after_step(true, false, false, -1.0, 0.1, 0.1, 2),
            Action::Continue,
            "three failures first"
        );
        assert_eq!(
            after_step(true, false, false, -1.0, 0.2, 0.1, 5),
            Action::Continue,
            "Δ has not reached ρ"
        );
        assert_eq!(
            after_step(false, true, true, -1.0, 0.1, 0.1, 5),
            Action::Continue,
            "no geometry while growing"
        );
        assert_eq!(
            after_step(false, false, false, -1.0, 0.1, 0.1, 5),
            Action::Grow,
            "a growing set whose step stayed in its span grows"
        );
        assert_eq!(after_safety(true, 0.1, 0.1), Action::Geometry);
        assert_eq!(after_safety(false, 0.1, 0.1), Action::ReduceRho);
        assert_eq!(after_safety(false, 0.2, 0.1), Action::Continue);
    }

    /// Offsets `0.1`, `0.8` and `1.0` with `Δ = ρ = 0.1`: moving the farthest point in still leaves the `0.8` one
    /// beyond `2Δ`, so ρ waits for both.
    #[test]
    fn rho_waits_for_every_far_point() {
        let mut set = Set::new(Point::new(DVector::zeros(3), vec![0.0]));
        for (i, length) in [0.1, 0.8, 1.0].into_iter().enumerate() {
            let mut x = DVector::zeros(3);
            x[i] = length;
            set.points.push(Point::new(x, vec![1.0]));
        }
        let decide = |set: &Set| after_step(true, false, poor(set, 0.1), -1.0, 0.1, 0.1, 3);
        assert_eq!(decide(&set), Action::Geometry);
        set.points[3].x[2] = 0.05;
        assert_eq!(decide(&set), Action::Geometry, "the 0.8 offset is still beyond 2Δ");
        set.points[2].x[1] = 0.05;
        assert_eq!(decide(&set), Action::ReduceRho);
    }

    #[test]
    fn the_radius_follows_the_ratio() {
        assert_eq!(radius(1.0, 0.1, 0.8, 1.0), 4.0);
        assert_eq!(radius(1.0, 0.1, 0.5, 0.2), 0.5);
        assert_eq!(radius(1.0, 0.1, 0.05, 0.2), 0.2);
        assert_eq!(radius(1.0, 0.3, -1.0, 0.2), 0.3);
        assert_eq!(next_rho(1.0), 0.1);
        assert_eq!(next_rho(100.0 * RHO_END), (100.0 * RHO_END * RHO_END).sqrt());
        assert_eq!(next_rho(10.0 * RHO_END), RHO_END);
    }

    /// `r(x) = [3 − x₀ − 2x₁, x₀ − x₁]`, in two variables: smallest at the corner `(0, 0)` of the box below.
    struct Plane;

    impl Objective for Plane {
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
            let r = vec![3.0 - x[0] - 2.0 * x[1], x[0] - x[1]];
            let f: f64 = r.iter().map(|v| v * v).sum();
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

    /// `x_k` on the upper corner of the box, `(0, 0)`, with the other points along `(1, 1)`: the orthogonal
    /// direction leaves the box both ways, yet the growing safety step's new direction and the repair each find a
    /// feasible point outside the span, and the repaired set is poised.
    #[test]
    fn new_directions_work_at_a_corner_of_the_box() {
        let mut sink = NoProgress;
        let mut ctl = RunControl {
            max_iter: 100,
            target_fidelity: 2.0,
            seed: 1,
            sink: &mut sink,
        };
        let mut monitor = Monitor::new(&mut ctl);
        let (lower, upper) = (DVector::from_element(2, -100.0), DVector::zeros(2));
        // (0, 0) has f = 9 and the others more (10.89, 12.96), so it stays x_k.
        let mut points = [[0.0, 0.0], [-0.1, -0.1], [-0.2, -0.2]].into_iter().map(|x| {
            evaluate(&Plane, &mut monitor, &lower, &upper, DVector::from_column_slice(&x))
                .unwrap()
                .unwrap()
        });
        let mut set = Set::new(points.next().unwrap());
        points.for_each(|p| set.push(p));
        let f0 = set.points[0].f;
        let mut solver = Solver {
            obj: &Plane,
            monitor: &mut monitor,
            lower,
            upper,
            rng: Rng::seed_from_u64(1),
            set,
            delta: 0.5,
            rho: 0.5,
            unsuccessful: 0,
            slow: Slow::default(),
            f0,
        };
        assert_eq!(solver.set.k, 0);
        let along = DMatrix::from_column_slice(2, 1, &[1.0, 1.0]) / 2f64.sqrt();
        let x = solver.new_direction(&along).expect("a direction at the corner");
        assert!(x.iter().all(|&v| v <= 0.0), "{x}");
        assert!(outside(&along, &x) >= x.norm() / 2f64.sqrt() - 1e-12, "{x}");
        let Build::Unpoised(dependent) = solver.set.model() else {
            panic!("collinear offsets cannot be poised");
        };
        assert!(solver.repair(&dependent).unwrap().is_continue());
        assert!(
            matches!(solver.set.model(), Build::Model(_)),
            "the repaired set is poised"
        );
        assert!(solver.set.points.iter().all(|p| p.x.iter().all(|&v| v <= 0.0)));
    }

    /// A set built with points at `0.1Δ`, `8Δ` and `10Δ` runs to convergence; the debug assertion in `reduce_rho`
    /// checks along the way that ρ never shrinks before the far points have moved.
    #[test]
    fn a_run_from_a_set_with_far_points_converges() {
        let mut sink = NoProgress;
        let mut ctl = RunControl {
            max_iter: 1000,
            target_fidelity: 2.0,
            seed: 1,
            sink: &mut sink,
        };
        let mut monitor = Monitor::new(&mut ctl);
        let (lower, upper) = (DVector::from_element(3, -100.0), DVector::from_element(3, 100.0));
        let delta = 0.1;
        let mut points = [
            [0.0, 0.0, 0.0],
            [0.1 * delta, 0.0, 0.0],
            [0.0, 8.0 * delta, 0.0],
            [0.0, 0.0, 10.0 * delta],
        ]
        .into_iter()
        .map(|x| {
            evaluate(&Shifted, &mut monitor, &lower, &upper, DVector::from_column_slice(&x))
                .unwrap()
                .unwrap()
        });
        let mut set = Set::new(points.next().unwrap());
        points.for_each(|p| set.push(p));
        let f0 = set.points[0].f;
        let mut solver = Solver {
            obj: &Shifted,
            monitor: &mut monitor,
            lower,
            upper,
            rng: Rng::seed_from_u64(1),
            set,
            delta,
            rho: delta,
            unsuccessful: 0,
            slow: Slow::default(),
            f0,
        };
        let exit = loop {
            if let Break(exit) = solver.iterate().unwrap() {
                break exit;
            }
        };
        assert_eq!(exit, Exit::Converged);
        let want = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        assert!((solver.set.xk() - want).norm() < 1e-6, "{}", solver.set.xk());
    }

    #[test]
    fn one_initial_direction_costs_two_evaluations() {
        let mut sink = NoProgress;
        let mut ctl = RunControl {
            max_iter: 100,
            target_fidelity: 2.0,
            seed: 1,
            sink: &mut sink,
        };
        let mut monitor = Monitor::new(&mut ctl);
        let Continue(mut solver) = Solver::start(&Shifted, &[0.0; 3], Some(1), &mut monitor).unwrap() else {
            panic!("stopped during the initial set");
        };
        assert_eq!((solver.set.len(), solver.monitor.evaluations), (2, 2));
        assert!(solver.iterate().unwrap().is_continue());
        assert_eq!(solver.monitor.evaluations, 3);
    }

    /// A residual objective from a function, for the tests below.
    struct Residuals<F: Fn(&[f64]) -> Vec<f64> + Sync>(usize, F);

    impl<F: Fn(&[f64]) -> Vec<f64> + Sync> Objective for Residuals<F> {
        fn dim(&self) -> usize {
            self.0
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
            let r = (self.1)(x);
            let f: f64 = r.iter().map(|v| v * v).sum();
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

    /// A solver over the points `xs`, evaluated in order, with `Δ = ρ = delta` and a wide box.
    fn solver_over<'a, 'm, 's>(
        obj: &'a dyn Objective,
        monitor: &'m mut Monitor<'s>,
        xs: &[&[f64]],
        delta: f64,
    ) -> Solver<'a, 'm, 's> {
        let n = xs[0].len();
        let (lower, upper) = (DVector::from_element(n, -100.0), DVector::from_element(n, 100.0));
        let mut set: Option<Set> = None;
        for x in xs {
            let p = evaluate(obj, monitor, &lower, &upper, DVector::from_column_slice(x))
                .unwrap()
                .unwrap();
            match &mut set {
                None => set = Some(Set::new(p)),
                Some(set) => set.push(p),
            }
        }
        let set = set.unwrap();
        let f0 = set.points[0].f;
        Solver {
            obj,
            monitor,
            lower,
            upper,
            rng: Rng::seed_from_u64(1),
            set,
            delta,
            rho: delta,
            unsuccessful: 0,
            slow: Slow::default(),
            f0,
        }
    }

    fn quiet_monitor_control(sink: &mut NoProgress) -> RunControl<'_> {
        RunControl {
            max_iter: 100,
            target_fidelity: 2.0,
            seed: 1,
            sink,
        }
    }

    /// At the minimiser the model's step is zero: the safety branch shrinks ρ without evaluating anything.
    #[test]
    fn a_zero_step_goes_to_the_safety_branch_unevaluated() {
        let mut sink = NoProgress;
        let mut ctl = quiet_monitor_control(&mut sink);
        let mut monitor = Monitor::new(&mut ctl);
        let xs: [&[f64]; 4] = [
            &[1.0, 2.0, 3.0],
            &[1.05, 2.0, 3.0],
            &[1.0, 2.05, 3.0],
            &[1.0, 2.0, 3.05],
        ];
        let mut solver = solver_over(&Shifted, &mut monitor, &xs, 0.1);
        assert!(solver.iterate().unwrap().is_continue());
        assert_eq!(solver.monitor.evaluations, 4, "nothing was evaluated");
        assert!(solver.rho < 0.1, "ρ = {}", solver.rho);
    }

    /// A growing set whose model explains `r_k` inside its span takes a step inside the span; the new point replaces
    /// one rather than joining as a new direction.
    #[test]
    fn a_growing_point_inside_the_span_replaces_rather_than_adds() {
        let line = Residuals(3, |x: &[f64]| vec![x[0] - 1.0]);
        let mut sink = NoProgress;
        let mut ctl = quiet_monitor_control(&mut sink);
        let mut monitor = Monitor::new(&mut ctl);
        let xs: [&[f64]; 2] = [&[0.1, 0.0, 0.0], &[0.0, 0.0, 0.0]];
        let mut solver = solver_over(&line, &mut monitor, &xs, 0.1);
        assert!(solver.iterate().unwrap().is_continue());
        assert_eq!(solver.monitor.evaluations, 3);
        assert_eq!(solver.set.len(), 2, "the set did not grow");
        assert!((solver.set.xk() - DVector::from_column_slice(&[0.2, 0.0, 0.0])).norm() < 1e-12);
    }

    /// With one initial direction in ten variables, the third evaluation is the first trust-region step: the point
    /// the filled model's step predicts, not a growing safety point.
    #[test]
    fn the_third_evaluation_of_a_one_direction_start_is_a_model_step() {
        let rosenbrock = Residuals(10, |x: &[f64]| {
            x.chunks(2)
                .flat_map(|p| [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
                .collect()
        });
        let x0: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
        let mut sink = NoProgress;
        let mut ctl = quiet_monitor_control(&mut sink);
        let mut monitor = Monitor::new(&mut ctl);
        let Continue(mut solver) = Solver::start(&rosenbrock, &x0, Some(1), &mut monitor).unwrap() else {
            panic!("stopped during the initial set");
        };
        assert_eq!(solver.monitor.evaluations, 2);
        let Build::Model(model) = solver.set.model() else {
            panic!("two points are poised");
        };
        let j = model.filled(&mut solver.rng.clone());
        let (lower, upper) = solver.local_box();
        let s = trust_region_step(&j, &model.r, solver.delta, &lower, &upper);
        assert!(s.norm() >= GAMMA_S * solver.rho && predicted_decrease(&j, &model.r, &s) > 0.0);
        let want = solver.set.xk() + &s;
        assert!(solver.iterate().unwrap().is_continue());
        assert_eq!(solver.monitor.evaluations, 3);
        assert!(
            solver.set.points.iter().any(|p| (&p.x - &want).norm() < 1e-14),
            "the step's point was evaluated"
        );
    }
}
