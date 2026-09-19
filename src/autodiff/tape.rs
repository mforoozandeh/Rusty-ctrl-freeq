//! The reverse-mode tape.
//!
//! A [`Tape`] records every operation as a node holding its value and how it was made.  [`Tape::backward`] walks
//! the nodes in reverse and applies each operation's adjoint, giving the gradient of a real scalar with respect to
//! every leaf - the same model as PyTorch's autograd, at the granularity of whole matrices.
//!
//! Gradients with respect to complex values follow PyTorch's conjugate convention: for a real loss `L` and a complex
//! `z = x + iy`, the gradient is `∂L/∂x + i·∂L/∂y`.
//!
//! Constants an operation needs - basis matrices, control operators, targets - are borrowed for the tape's lifetime
//! `'a` rather than copied, because the propagation records several hundred nodes per batch element per evaluation.

use super::ops_complex::{LindbladOps, Step};
use super::{C, Scalar};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat};

/// A node's value: a real or a complex matrix.  Vectors are single columns and scalars are `1 × 1`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value<T> {
    /// A real matrix.
    R(RMat<T>),
    /// A complex matrix.
    C(CMat<T>),
}

impl<T: Scalar> Value<T> {
    /// `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Value::R(m) => m.shape(),
            Value::C(m) => m.shape(),
        }
    }

    /// The real matrix, or an error naming `what`.
    pub fn r(&self, what: &str) -> Result<&RMat<T>> {
        match self {
            Value::R(m) => Ok(m),
            Value::C(_) => Err(Error::Dimension(format!("{what} needs a real operand"))),
        }
    }

    /// The complex matrix, or an error naming `what`.
    pub fn c(&self, what: &str) -> Result<&CMat<T>> {
        match self {
            Value::C(m) => Ok(m),
            Value::R(_) => Err(Error::Dimension(format!("{what} needs a complex operand"))),
        }
    }

    /// `self += o`, elementwise.
    fn accumulate(&mut self, o: Value<T>) {
        match (self, o) {
            (Value::R(a), Value::R(b)) => a.data.iter_mut().zip(b.data).for_each(|(x, y)| *x += y),
            (Value::C(a), Value::C(b)) => a.data.iter_mut().zip(b.data).for_each(|(x, y)| *x += y),
            _ => unreachable!("gradients of one node always share its type"),
        }
    }

    /// Every entry times the real `s`.
    pub(crate) fn scaled(&self, s: T) -> Value<T> {
        match self {
            Value::R(m) => Value::R(m.map(|v| v * s)),
            Value::C(m) => Value::C(m.scale_re(s)),
        }
    }
}

/// A handle to a node on a [`Tape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(pub(crate) usize);

/// How a node was made, with what its adjoint needs.
pub(crate) enum Op<'a, T> {
    Leaf,
    Constant,
    Add(Var, Var),
    Sub(Var, Var),
    Mul(Var, Var),
    Scale(Var, f64),
    Square(Var),
    Sqrt(Var),
    Cos(Var),
    Sin(Var),
    Atan2(Var, Var),
    MatmulConst(&'a RMat<f64>, Var),
    Slice(Var, usize),
    Col(Var, usize),
    HStack(Vec<Var>),
    AmpPenalty(Var, Option<usize>),
    Sum(Var),
    Mean(Var),
    Complex(Var, Var),
    Real(Var),
    Imag(Var),
    MulConstC(Var, &'a CMat<f64>),
    LincombRow(Var, usize, &'a [CMat<f64>]),
    ExpmMiDt(Var, f64),
    Matmul(Var, Var),
    Sandwich(Var, Var),
    Lindblad(Var, &'a LindbladOps, Step),
    Fidelity(&'a CMat<f64>, Var),
    ReTraceProduct(&'a CMat<f64>, Var),
    BatchMean(Vec<Var>, Vec<Value<T>>),
}

pub(crate) struct Node<'a, T> {
    pub(crate) value: Value<T>,
    pub(crate) op: Op<'a, T>,
    pub(crate) requires_grad: bool,
}

/// A record of operations that can be differentiated in reverse.
pub struct Tape<'a, T: Scalar> {
    pub(crate) nodes: Vec<Node<'a, T>>,
    grad_enabled: bool,
}

impl<T: Scalar> Default for Tape<'_, T> {
    fn default() -> Self {
        Tape::new()
    }
}

/// Gradients from one [`Tape::backward`] call.
pub struct Grads<T> {
    grads: Vec<Option<Value<T>>>,
}

impl<T: Scalar> Grads<T> {
    /// The gradient with respect to `v`, or `None` if the output does not depend on it.
    pub fn wrt(&self, v: Var) -> Option<&Value<T>> {
        self.grads.get(v.0).and_then(|g| g.as_ref())
    }

    /// The gradient with respect to a real leaf, as a flat vector (zeros if the output does not depend on it).
    pub fn wrt_real(&self, v: Var, len: usize) -> Vec<T> {
        match self.wrt(v) {
            Some(Value::R(m)) => m.data.clone(),
            _ => vec![T::zero(); len],
        }
    }
}

impl<'a, T: Scalar> Tape<'a, T> {
    /// A tape that records what it needs for [`backward`](Self::backward).
    pub fn new() -> Self {
        Tape {
            nodes: Vec::new(),
            grad_enabled: true,
        }
    }

    /// A tape for evaluation only: values, no gradients.
    pub fn no_grad() -> Self {
        Tape {
            nodes: Vec::new(),
            grad_enabled: false,
        }
    }

    /// Whether this tape records gradients.
    pub fn grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// A differentiable input.
    pub fn leaf(&mut self, value: Value<T>) -> Var {
        let requires = self.grad_enabled;
        self.nodes.push(Node {
            value,
            op: Op::Leaf,
            requires_grad: requires,
        });
        Var(self.nodes.len() - 1)
    }

    /// A constant input.
    pub fn constant(&mut self, value: Value<T>) -> Var {
        self.nodes.push(Node {
            value,
            op: Op::Constant,
            requires_grad: false,
        });
        Var(self.nodes.len() - 1)
    }

    /// The value of `v`.
    pub fn value(&self, v: Var) -> &Value<T> {
        &self.nodes[v.0].value
    }

    /// The value of a `1 × 1` real node.
    pub fn scalar(&self, v: Var) -> Result<T> {
        let m = self.value(v).r("scalar")?;
        if m.shape() != (1, 1) {
            return Err(Error::Dimension(format!(
                "expected a scalar, got {}x{}",
                m.rows, m.cols
            )));
        }
        Ok(m.data[0])
    }

    /// The number of nodes recorded.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn requires(&self, v: Var) -> bool {
        self.nodes[v.0].requires_grad
    }

    /// Record a node made from `parents`.
    pub(crate) fn push(&mut self, value: Value<T>, op: Op<'a, T>, parents: &[Var]) -> Var {
        let requires = self.grad_enabled && parents.iter().any(|&p| self.requires(p));
        let op = if requires { op } else { Op::Constant };
        self.nodes.push(Node {
            value,
            op,
            requires_grad: requires,
        });
        Var(self.nodes.len() - 1)
    }

    /// Gradients of the real `1 × 1` node `root` with respect to every node it depends on.
    pub fn backward(&self, root: Var) -> Result<Grads<T>> {
        if !self.grad_enabled {
            return Err(Error::NotSupported("backward on a tape made with Tape::no_grad".into()));
        }
        self.scalar(root)?;
        let mut grads: Vec<Option<Value<T>>> = Vec::with_capacity(root.0 + 1);
        grads.resize_with(root.0 + 1, || None);
        grads[root.0] = Some(Value::R(RMat::from_vec(1, 1, vec![T::one()])?));

        for i in (0..=root.0).rev() {
            let Some(g) = grads[i].take() else { continue };
            let node = &self.nodes[i];
            if node.requires_grad {
                for (parent, pg) in self.adjoint(node, &g)? {
                    if !self.requires(parent) {
                        continue;
                    }
                    match &mut grads[parent.0] {
                        Some(acc) => acc.accumulate(pg),
                        slot => *slot = Some(pg),
                    }
                }
            }
            grads[i] = Some(g);
        }
        Ok(Grads { grads })
    }

    /// The adjoint of one node: the gradient contribution to each parent given the node's own gradient `g`.
    fn adjoint(&self, node: &Node<'a, T>, g: &Value<T>) -> Result<Vec<(Var, Value<T>)>> {
        use super::ops_complex as oc;
        use super::ops_real as or;
        let v = |x: Var| &self.nodes[x.0].value;
        Ok(match &node.op {
            Op::Leaf | Op::Constant => Vec::new(),
            Op::Add(a, b) => vec![(*a, g.clone()), (*b, g.clone())],
            Op::Sub(a, b) => vec![(*a, g.clone()), (*b, g.scaled(-T::one()))],
            Op::Mul(a, b) => or::mul_adjoint(*a, *b, v(*a), v(*b), g)?,
            Op::Scale(a, s) => vec![(*a, g.scaled(T::from_f64(*s)))],
            Op::Square(a) => vec![(*a, or::square_adjoint(v(*a), g)?)],
            Op::Sqrt(a) => vec![(*a, or::sqrt_adjoint(&node.value, g)?)],
            Op::Cos(a) => vec![(*a, or::cos_adjoint(v(*a), g)?)],
            Op::Sin(a) => vec![(*a, or::sin_adjoint(v(*a), g)?)],
            Op::Atan2(y, x) => or::atan2_adjoint(*y, *x, v(*y), v(*x), g)?,
            Op::MatmulConst(m, a) => vec![(*a, or::matmul_const_adjoint(m, g)?)],
            Op::Slice(a, start) => vec![(*a, or::slice_adjoint(v(*a), *start, g)?)],
            Op::Col(a, c) => vec![(*a, or::col_adjoint(v(*a), *c, g)?)],
            Op::HStack(parts) => or::hstack_adjoint(parts, |p| v(p).shape().1, g)?,
            Op::AmpPenalty(a, arg) => vec![(*a, or::amp_penalty_adjoint(v(*a), *arg, g)?)],
            Op::Sum(a) => vec![(*a, or::sum_adjoint(v(*a), g, T::one())?)],
            Op::Mean(a) => {
                let (r, c) = v(*a).shape();
                vec![(*a, or::sum_adjoint(v(*a), g, T::from_f64(1.0 / (r * c) as f64))?)]
            }
            Op::Complex(re, im) => oc::complex_adjoint(*re, *im, g)?,
            Op::Real(z) => vec![(*z, oc::real_adjoint(g, false)?)],
            Op::Imag(z) => vec![(*z, oc::real_adjoint(g, true)?)],
            Op::MulConstC(z, c) => vec![(*z, oc::mul_const_c_adjoint(c, g)?)],
            Op::LincombRow(u, row, ops) => vec![(*u, oc::lincomb_row_adjoint(v(*u), *row, ops, g)?)],
            Op::ExpmMiDt(h, dt) => vec![(*h, oc::expm_mi_dt_adjoint(v(*h), *dt, g)?)],
            Op::Matmul(a, b) => oc::matmul_adjoint(*a, *b, v(*a), v(*b), g)?,
            Op::Sandwich(u, rho) => oc::sandwich_adjoint(*u, *rho, v(*u), v(*rho), g)?,
            Op::Lindblad(rho, ops, step) => vec![(*rho, oc::lindblad_adjoint(ops, *step, g)?)],
            Op::Fidelity(t, psi) => vec![(*psi, oc::fidelity_adjoint(t, v(*psi), g)?)],
            Op::ReTraceProduct(sigma, rho) => vec![(*rho, oc::re_trace_product_adjoint(sigma, g)?)],
            Op::BatchMean(inputs, stored) => {
                let s = g.r("batch mean gradient")?.data[0];
                inputs.iter().zip(stored).map(|(&x, gx)| (x, gx.scaled(s))).collect()
            }
        })
    }
}

/// Multiply an `f64` complex constant by a complex value of any scalar type.
#[inline]
pub(crate) fn mulc<T: Scalar>(a: C<f64>, b: C<T>) -> C<T> {
    let (ar, ai) = (T::from_f64(a.re), T::from_f64(a.im));
    C::new(ar * b.re - ai * b.im, ar * b.im + ai * b.re)
}

/// Lift an `f64` complex constant.
#[inline]
pub(crate) fn liftc<T: Scalar>(a: C<f64>) -> C<T> {
    C::new(T::from_f64(a.re), T::from_f64(a.im))
}
