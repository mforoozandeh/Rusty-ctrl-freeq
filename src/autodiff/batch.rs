//! The batch mean: the only place the objective runs in parallel.
//!
//! The objective averages a scalar fidelity over many independent batch elements - drift snapshots, Rabi
//! snapshots and initial states.  Each element gets its own sub-tape, built from copies of the shared inputs.  When
//! gradients are wanted the element's reverse pass runs right away, so what comes back is just its value and its
//! gradient with respect to the inputs: small, and all the outer tape has to keep.  The sums are merged by
//! [`parallel::fold`](crate::parallel), whose grouping is fixed by the batch size, so the result does not depend on
//! the number of threads.

use super::Scalar;
use super::tape::{Op, Tape, Value, Var};
use crate::error::{Error, Result};
use crate::linalg::RMat;
use crate::parallel::{PIECES, fold};

/// What one piece of the fold carries: the summed values and the summed input gradients.
type Partial<T> = Result<(T, Vec<Value<T>>)>;

impl<'a, T: Scalar> Tape<'a, T> {
    /// The mean over `n` elements of the real scalar each builds with `f`.
    ///
    /// `f(tape, inputs, b)` builds element `b` on a fresh sub-tape whose first nodes are copies of `inputs` (as
    /// leaves when this tape records gradients, as constants otherwise) and returns a `1 × 1` real node.  Elements
    /// may run on different threads.
    pub fn batch_mean<F>(&mut self, inputs: &[Var], n: usize, f: F) -> Result<Var>
    where
        F: Fn(&mut Tape<'a, T>, &[Var], usize) -> Result<Var> + Sync,
    {
        if n == 0 {
            return Err(Error::Dimension("batch_mean over an empty batch".into()));
        }
        let grad = self.grad_enabled() && inputs.iter().any(|&v| self.requires(v));
        let values: Vec<Value<T>> = inputs.iter().map(|&v| self.value(v).clone()).collect();

        let element = |b: usize| -> Partial<T> {
            let mut sub = if grad { Tape::new() } else { Tape::no_grad() };
            let vars: Vec<Var> = values
                .iter()
                .map(|v| {
                    if grad {
                        sub.leaf(v.clone())
                    } else {
                        sub.constant(v.clone())
                    }
                })
                .collect();
            let out = f(&mut sub, &vars, b)?;
            let value = sub.scalar(out)?;
            if !grad {
                return Ok((value, Vec::new()));
            }
            let grads = sub.backward(out)?;
            let per_input = vars
                .iter()
                .zip(&values)
                .map(|(&v, orig)| grads.wrt(v).cloned().unwrap_or_else(|| zeros_like(orig)))
                .collect();
            Ok((value, per_input))
        };

        let total: Partial<T> = fold(
            n,
            PIECES,
            || Ok((T::zero(), Vec::new())),
            |acc, b| {
                if acc.is_err() {
                    return;
                }
                match element(b) {
                    Ok(part) => merge_into(acc, part),
                    Err(e) => *acc = Err(e),
                }
            },
            |left, right| match right {
                Err(e) => {
                    if left.is_ok() {
                        *left = Err(e);
                    }
                }
                Ok(part) => {
                    if left.is_ok() {
                        merge_into(left, part);
                    }
                }
            },
        );
        let (sum, grads) = total?;
        let inv = T::from_f64(1.0 / n as f64);
        let mean = RMat::from_vec(1, 1, vec![sum * inv])?;
        let stored = grads.iter().map(|g| g.scaled(inv)).collect();
        Ok(self.push(Value::R(mean), Op::BatchMean(inputs.to_vec(), stored), inputs))
    }
}

fn zeros_like<T: Scalar>(v: &Value<T>) -> Value<T> {
    let (r, c) = v.shape();
    match v {
        Value::R(_) => Value::R(RMat::<T>::zeros(r, c)),
        Value::C(_) => Value::C(crate::linalg::CMat::<T>::zeros(r, c)),
    }
}

/// Add `part` into the running `acc`, which must be `Ok`.
fn merge_into<T: Scalar>(acc: &mut Partial<T>, part: (T, Vec<Value<T>>)) {
    let Ok((sum, grads)) = acc else { return };
    *sum += part.0;
    if grads.is_empty() {
        *grads = part.1;
    } else {
        for (g, p) in grads.iter_mut().zip(part.1) {
            add_value(g, p);
        }
    }
}

fn add_value<T: Scalar>(a: &mut Value<T>, b: Value<T>) {
    match (a, b) {
        (Value::R(x), Value::R(y)) => x.data.iter_mut().zip(y.data).for_each(|(p, q)| *p += q),
        (Value::C(x), Value::C(y)) => x.data.iter_mut().zip(y.data).for_each(|(p, q)| *p += q),
        _ => unreachable!("an input's gradient always shares its type"),
    }
}
