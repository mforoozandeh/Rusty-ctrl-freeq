//! The cost function: waveform → modulation → control amplitudes → evolution of every batch element → fidelity.
//!
//! `cost = penalty − fidelity`, with the fidelity averaged over the batch and the penalty on amplitudes above one.
//! Gradients come from the reverse pass, Hessian-vector products from a reverse pass over dual numbers, and the
//! Hessian from one such product per parameter.

mod graph;

use std::sync::Mutex;

use nalgebra::DMatrix;

use crate::autodiff::{Dual, Scalar, Tape, Value};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat};
use crate::optim::{Eval, Objective};
use crate::setup::{Evolution, Problem};

/// A problem's cost, ready to evaluate.
pub struct CostModel {
    problem: Problem,
    pool: crate::parallel::Pool,
    /// The most recent evaluation, so an optimiser asking for the value, gradient and Hessian at one point in
    /// separate calls pays for one evaluation.
    cache: Mutex<Option<Cached>>,
}

struct Cached {
    x: Vec<f64>,
    eval: Eval,
    grad: Option<Vec<f64>>,
    hess: Option<DMatrix<f64>>,
}

/// Modulated waveforms for plotting and export, `n_pulse × n_qubits` each.
#[derive(Clone, Debug, PartialEq)]
pub struct Waveforms {
    /// In-phase quadrature, after modulation.
    pub cx: RMat<f64>,
    /// Quadrature, after modulation.
    pub cy: RMat<f64>,
    /// `sqrt(cx² + cy²)`.
    pub amp: RMat<f64>,
    /// `atan2(cy, cx)`.
    pub phase: RMat<f64>,
}

impl CostModel {
    /// Wrap a built problem.
    pub fn new(problem: Problem) -> Self {
        CostModel {
            problem,
            pool: crate::parallel::Pool::default(),
            cache: Mutex::new(None),
        }
    }

    /// Evaluate on a pool of `threads` threads instead of the default one.  No effect without the `parallel`
    /// feature or on `wasm32`.
    pub fn with_threads(mut self, threads: usize) -> Result<Self> {
        self.pool = crate::parallel::Pool::with_threads(threads)?;
        Ok(self)
    }

    /// The problem.
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.problem.n_params()
    }

    fn check_len(&self, n: usize) -> Result<()> {
        if n != self.dim() {
            return Err(Error::Dimension(format!(
                "{n} parameters for a problem with {}",
                self.dim()
            )));
        }
        Ok(())
    }

    /// `(cost, fidelity, penalty, gradient of the cost)` over any scalar type; the gradient only when `grad`.
    pub fn eval_generic<T: Scalar>(&self, x: &[T], grad: bool) -> Result<(T, T, T, Option<Vec<T>>)> {
        self.check_len(x.len())?;
        let mut tape: Tape<'_, T> = if grad { Tape::new() } else { Tape::no_grad() };
        let xv = if grad {
            tape.leaf(Value::R(RMat::column(x.to_vec())))
        } else {
            tape.constant(Value::R(RMat::column(x.to_vec())))
        };
        let problem = &self.problem;
        let (cost, fid, pen) = self.pool.install(|| graph::build_cost(problem, &mut tape, xv))?;
        let values = (tape.scalar(cost)?, tape.scalar(fid)?, tape.scalar(pen)?);
        let g = if grad {
            Some(tape.backward(cost)?.wrt_real(xv, x.len()))
        } else {
            None
        };
        Ok((values.0, values.1, values.2, g))
    }

    /// The cost at `x`.
    pub fn value(&self, x: &[f64]) -> Result<Eval> {
        let (cost, fidelity, penalty, _) = self.eval_generic(x, false)?;
        finite(Eval {
            cost,
            fidelity,
            penalty,
        })
    }

    /// The cost and its gradient at `x`.
    pub fn value_and_gradient(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        let (cost, fidelity, penalty, g) = self.eval_generic(x, true)?;
        let g = g.unwrap_or_default();
        if g.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("the gradient is not finite".into()));
        }
        Ok((
            finite(Eval {
                cost,
                fidelity,
                penalty,
            })?,
            g,
        ))
    }

    /// The Hessian of the cost at `x` applied to `v`, exactly: the reverse pass over dual numbers `x + ε·v`.
    pub fn hvp(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>> {
        self.check_len(v.len())?;
        let xd: Vec<Dual> = x.iter().zip(v).map(|(&a, &d)| Dual::new(a, d)).collect();
        let (_, _, _, g) = self.eval_generic(&xd, true)?;
        Ok(g.unwrap_or_default().iter().map(|d| d.eps).collect())
    }

    /// The Hessian of the cost at `x`: one exact Hessian-vector product per column, possibly in parallel,
    /// symmetrised.
    pub fn hessian_matrix(&self, x: &[f64]) -> Result<DMatrix<f64>> {
        let n = self.dim();
        self.check_len(x.len())?;
        let columns = self.pool.install(|| {
            crate::parallel::map(n, |i| {
                let mut e = vec![0.0; n];
                e[i] = 1.0;
                self.hvp(x, &e)
            })
        });
        let mut h = DMatrix::zeros(n, n);
        for (i, col) in columns.into_iter().enumerate() {
            for (r, v) in col?.into_iter().enumerate() {
                h[(r, i)] = v;
            }
        }
        let sym = (&h + h.transpose()) * 0.5;
        if sym.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("the Hessian is not finite".into()));
        }
        Ok(sym)
    }

    /// The cost at `x` and its residual vector, whose squared norm is `cost + 1`: the least-squares form of the
    /// objective, for Hilbert-space problems.
    ///
    /// A batch element evolving `Ψ` (`D × k`, orthonormal columns) against `T` scores
    /// `F = (‖M‖² + |s|²)/(k(k+1))` with `M = T†Ψ` and `s = Tr M`, and exactly
    /// `1 − F = ‖Ψ − T·M‖²/k + ‖M − (s/k)·I‖²/(k+1)`: population leaving the target's span, then the error inside
    /// it.  Each element contributes both blocks scaled by `1/√B`; the last entry is `max(0, max|a| − 1)`, the
    /// square root of the amplitude penalty.  Liouville-space evolution has no such form.
    pub fn residuals(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        self.check_len(x.len())?;
        let p = &self.problem;
        if !matches!(p.evolution, Evolution::Hilbert) {
            return Err(Error::NotSupported(
                "the least-squares form of the objective exists only in Hilbert space, without dissipation".into(),
            ));
        }
        let mut tape: Tape<'_, f64> = Tape::no_grad();
        let xv = tape.constant(Value::R(RMat::column(x.to_vec())));
        let w = graph::build_waveforms(p, &mut tape, xv)?;
        let peak = tape
            .value(w.amp)
            .r("residuals")?
            .data
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let excess = (peak - 1.0).max(0.0);
        let (cx, cy) = (tape.value(w.cx).clone(), tape.value(w.cy).clone());
        let b = p.batch.len();
        let scale = 1.0 / (b as f64).sqrt();
        let elements = self.pool.install(|| {
            crate::parallel::map(b, |i| -> Result<(f64, Vec<f64>)> {
                let mut sub: Tape<'_, f64> = Tape::no_grad();
                let cx = sub.constant(cx.clone());
                let cy = sub.constant(cy.clone());
                let state = graph::element_state(p, &p.batch[i], &mut sub, cx, cy)?;
                element_residual(&p.batch[i].target, sub.value(state).c("residuals")?, scale)
            })
        });
        let mut r = Vec::new();
        let mut fidelity = 0.0;
        for element in elements {
            let (f, part) = element?;
            fidelity += f;
            r.extend(part);
        }
        fidelity /= b as f64;
        r.push(excess);
        if r.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("the residuals are not finite".into()));
        }
        let penalty = excess * excess;
        let eval = finite(Eval {
            cost: penalty - fidelity,
            fidelity,
            penalty,
        })?;
        Ok((eval, r))
    }

    /// The modulated waveforms at `x`.
    pub fn waveforms(&self, x: &[f64]) -> Result<Waveforms> {
        self.check_len(x.len())?;
        let mut tape: Tape<'_, f64> = Tape::no_grad();
        let xv = tape.constant(Value::R(RMat::column(x.to_vec())));
        let w = graph::build_waveforms(&self.problem, &mut tape, xv)?;
        let cx = tape.value(w.cx).r("waveforms")?.clone();
        let cy = tape.value(w.cy).r("waveforms")?.clone();
        let amp = RMat::from_fn(cx.rows, cx.cols, |r, c| cx.get(r, c).hypot(cy.get(r, c)));
        let phase = RMat::from_fn(cx.rows, cx.cols, |r, c| cy.get(r, c).atan2(cx.get(r, c)));
        Ok(Waveforms { cx, cy, amp, phase })
    }
}

impl CostModel {
    fn cached<R>(&self, x: &[f64], f: impl FnOnce(&Cached) -> Option<R>) -> Option<R> {
        let guard = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        guard.as_ref().filter(|c| c.x == x).and_then(f)
    }

    fn store(&self, x: &[f64], eval: Eval, grad: Option<Vec<f64>>, hess: Option<DMatrix<f64>>) {
        let mut guard = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        match guard.as_mut().filter(|c| c.x == x) {
            Some(c) => {
                c.grad = grad.or(c.grad.take());
                c.hess = hess.or(c.hess.take());
            }
            None => {
                *guard = Some(Cached {
                    x: x.to_vec(),
                    eval,
                    grad,
                    hess,
                })
            }
        }
    }
}

impl Objective for CostModel {
    fn dim(&self) -> usize {
        CostModel::dim(self)
    }

    fn value(&self, x: &[f64]) -> Result<Eval> {
        if let Some(e) = self.cached(x, |c| Some(c.eval)) {
            return Ok(e);
        }
        let e = CostModel::value(self, x)?;
        self.store(x, e, None, None);
        Ok(e)
    }

    fn gradient(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        if let Some(hit) = self.cached(x, |c| c.grad.clone().map(|g| (c.eval, g))) {
            return Ok(hit);
        }
        let (e, g) = self.value_and_gradient(x)?;
        self.store(x, e, Some(g.clone()), None);
        Ok((e, g))
    }

    fn hessian(&self, x: &[f64]) -> Result<(Eval, Vec<f64>, DMatrix<f64>)> {
        if let Some(hit) = self.cached(x, |c| match (&c.grad, &c.hess) {
            (Some(g), Some(h)) => Some((c.eval, g.clone(), h.clone())),
            _ => None,
        }) {
            return Ok(hit);
        }
        let (e, g) = Objective::gradient(self, x)?;
        let h = self.hessian_matrix(x)?;
        self.store(x, e, None, Some(h.clone()));
        Ok((e, g, h))
    }

    fn hvp(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>> {
        CostModel::hvp(self, x, v)
    }

    fn residuals(&self, x: &[f64]) -> Result<(Eval, Vec<f64>)> {
        CostModel::residuals(self, x)
    }
}

fn finite(e: Eval) -> Result<Eval> {
    if e.cost.is_finite() && e.fidelity.is_finite() && e.penalty.is_finite() {
        Ok(e)
    } else {
        Err(Error::Numerical("the objective is not finite".into()))
    }
}

/// One Hilbert batch element: its fidelity, and its two residual blocks `(Ψ − T·M)/√k` and `(M − (s/k)·I)/√(k+1)`
/// times `scale`, each as real parts then imaginary parts, row-major.
fn element_residual(target: &CMat<f64>, psi: &CMat<f64>, scale: f64) -> Result<(f64, Vec<f64>)> {
    let k = target.cols as f64;
    let m = target.adjoint().matmul(psi)?;
    let s = m.trace();
    let leak = psi.sub(&target.matmul(&m)?)?;
    let mut inside = m.clone();
    for i in 0..inside.rows {
        let d = inside.get(i, i) - s / k;
        inside.set(i, i, d);
    }
    let fidelity = (m.data.iter().map(|z| z.norm_sqr()).sum::<f64>() + s.norm_sqr()) / (k * (k + 1.0));
    let (a, b) = (scale / k.sqrt(), scale / (k + 1.0).sqrt());
    let mut out = Vec::with_capacity(2 * (leak.data.len() + inside.data.len()));
    out.extend(leak.data.iter().map(|z| a * z.re));
    out.extend(leak.data.iter().map(|z| a * z.im));
    out.extend(inside.data.iter().map(|z| b * z.re));
    out.extend(inside.data.iter().map(|z| b * z.im));
    Ok((fidelity, out))
}

#[cfg(test)]
mod tests;
