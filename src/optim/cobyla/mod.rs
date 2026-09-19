//! COBYLA (Powell's Constrained Optimization BY Linear Approximations), vendored.
//!
//! The solver is the `cobyla` crate 1.0.4 by Rémi Lafage (MIT licence, in `LICENSE.md` beside this file), a
//! translation of NLopt's COBYLA.  It is vendored rather than used as a dependency for one reason: its public
//! `minimize` keeps NLopt's force-stop flag private, and ctrl-freeq has to end a run early when the target fidelity
//! is reached or the user cancels.  This wrapper passes a caller-owned flag in its place; `nlopt_cobyla.rs` only
//! differs from upstream in its clock import and lint allowances.

mod nlopt_cobyla;

use std::cell::Cell;
use std::ffi::{c_int, c_void};

use nlopt_cobyla::{NLoptFunctionCfg, cobyla_minimize, nlopt_function_raw_callback, nlopt_stopping};

/// Why COBYLA stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    /// The trust region shrank to its final radius.
    Converged,
    /// The caller raised the stop flag.
    ForcedStop,
    /// The evaluation budget ran out.
    MaxEvalReached,
    /// Rounding errors stopped further progress; the best point is still valid.
    RoundoffLimited,
    /// The solver failed; the message says how.
    Failed(&'static str),
}

/// Minimise `func` inside the box `lower..=upper` from `x0`.
///
/// `rhobeg` is the initial trust-region radius.  At most `max_eval` evaluations are made.  The run also ends right
/// after the evaluation during which `force_stop` is set to a non-zero value.  Returns the status, the final point
/// and its value, as the solver reports them; callers that need the best point ever evaluated track it inside
/// `func`.
pub(crate) fn minimize<F: Fn(&[f64]) -> f64>(
    func: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    rhobeg: f64,
    max_eval: usize,
    force_stop: &Cell<c_int>,
) -> (Status, Vec<f64>, f64) {
    let n = x0.len();
    assert!(lower.len() == n && upper.len() == n, "bounds must match x0");

    let objective = move |x: &[f64], _: &mut ()| func(x);
    let fn_cfg = NLoptFunctionCfg {
        objective_fn: objective,
        user_data: (),
    };
    let fn_cfg_ptr = &fn_cfg as *const _ as *mut c_void;

    let mut x = x0.to_vec();
    let x_weights = vec![0.0; n];
    let dx = vec![rhobeg; n];
    let mut minf = f64::INFINITY;
    let mut nevals: c_int = 0;
    let mut stop = nlopt_stopping {
        n: n as u32,
        minf_max: f64::NEG_INFINITY,
        ftol_rel: 0.0,
        ftol_abs: 0.0,
        xtol_rel: 0.0,
        xtol_abs: std::ptr::null(),
        x_weights: x_weights.as_ptr(),
        nevals_p: &mut nevals,
        maxeval: c_int::try_from(max_eval).unwrap_or(c_int::MAX),
        maxtime: 0.0,
        start: 0.0,
        force_stop: force_stop.as_ptr(),
        stop_msg: String::new(),
    };

    // SAFETY: every pointer handed over refers to a local that outlives the call: `fn_cfg` (whose type matches the
    // callback's type parameters), the bounds, `x`, `minf`, `stop` and `dx`, each of length `n` where an array is
    // expected.  `force_stop` is a `Cell`, so the solver reading it through a raw pointer while the objective
    // writes it through the `Cell` is sound.  There are no constraints, so the constraint arrays are null with
    // counts of zero.
    let status = unsafe {
        cobyla_minimize::<()>(
            n as u32,
            Some(raw_callback(&fn_cfg)),
            fn_cfg_ptr,
            0,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            lower.as_ptr(),
            upper.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            &mut stop,
            dx.as_ptr(),
        )
    };
    // Keep the closure's storage alive until the solver has returned.
    drop(fn_cfg);

    let status = match status as i32 {
        1..=4 => Status::Converged,
        5 => Status::MaxEvalReached,
        -4 => Status::RoundoffLimited,
        -5 => Status::ForcedStop,
        -2 => Status::Failed("invalid arguments"),
        -3 => Status::Failed("out of memory"),
        _ => Status::Failed("unexpected failure"),
    };
    (status, x, minf)
}

/// The C-style callback matching `cfg`'s closure type.  A function, so the closure's unnameable type can be
/// inferred from a value.
fn raw_callback<G: nlopt_cobyla::Func<()>>(
    _cfg: &NLoptFunctionCfg<G, ()>,
) -> fn(u32, *const f64, *mut f64, *mut c_void) -> f64 {
    nlopt_function_raw_callback::<G, ()>
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum()
    }

    #[test]
    fn converges_on_a_sphere() {
        let stop = Cell::new(0);
        let (status, x, f) = minimize(sphere, &[1.0; 5], &[-10.0; 5], &[10.0; 5], 0.5, 5_000, &stop);
        assert!(f < 1e-8, "{status:?} {f}");
        assert!(x.iter().all(|v| (v - 0.3).abs() < 1e-3));
    }

    /// Raising the flag inside the objective ends the run right after that evaluation.
    #[test]
    fn the_stop_flag_ends_the_run_at_once() {
        let stop = Cell::new(0);
        let evals = Cell::new(0usize);
        let frozen_at = Cell::new(None::<usize>);
        let (status, _, _) = minimize(
            |x| {
                evals.set(evals.get() + 1);
                let f = sphere(x);
                if f < 1e-2 && frozen_at.get().is_none() {
                    frozen_at.set(Some(evals.get()));
                    stop.set(1);
                }
                f
            },
            &[1.0; 10],
            &[-10.0; 10],
            &[10.0; 10],
            1.0,
            10_000,
            &stop,
        );
        assert_eq!(status, Status::ForcedStop);
        assert_eq!(Some(evals.get()), frozen_at.get(), "no evaluation after the stop");
    }

    #[test]
    fn the_evaluation_budget_is_respected() {
        let stop = Cell::new(0);
        let evals = Cell::new(0usize);
        let (status, _, _) = minimize(
            |x| {
                evals.set(evals.get() + 1);
                sphere(x)
            },
            &[1.0; 10],
            &[-10.0; 10],
            &[10.0; 10],
            1.0,
            25,
            &stop,
        );
        assert_eq!(status, Status::MaxEvalReached);
        assert_eq!(evals.get(), 25);
    }
}
