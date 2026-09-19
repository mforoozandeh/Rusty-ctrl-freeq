//! Automatic differentiation.
//!
//! * [`Scalar`] abstracts the real number type, so every kernel runs on `f64` and on [`Dual`] numbers.
//! * [`Tape`] records operations on real and complex matrices and differentiates a real scalar result in reverse,
//!   like PyTorch's autograd.
//! * [`Tape::batch_mean`] evaluates independent sub-graphs in parallel and averages them.
//!
//! Exact Hessian-vector products come from running a tape over [`Dual`] numbers: seeding the inputs with a direction
//! `v`, the derivative part of the gradient is `H·v`.

mod batch;
mod dual;
mod ops_complex;
mod ops_real;
mod scalar;
mod tape;

pub use dual::Dual;
pub use ops_complex::LindbladOps;
pub use scalar::{C, Scalar};
pub use tape::{Grads, Tape, Value, Var};

#[cfg(test)]
mod tests;
