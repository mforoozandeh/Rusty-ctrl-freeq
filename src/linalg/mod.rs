//! Small dense linear algebra generic over [`Scalar`](crate::autodiff::Scalar): matrices, a linear solve, the
//! matrix exponential and the adjoint of its derivative.

mod expm;
mod mat;

pub use expm::{expm, expm_frechet_adjoint, expm_mi_dt};
pub use mat::{CMat, Mat, RMat, solve};
