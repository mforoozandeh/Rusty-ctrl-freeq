//! Operations on complex matrices - the propagation's building blocks - and their adjoints.
//!
//! Adjoints use the conjugate convention of the tape: for `W = A·B`, `∂L/∂A = G·Bᴴ` and `∂L/∂B = Aᴴ·G`.

use super::tape::{Op, Tape, Value, Var, liftc, mulc};
use super::{C, Scalar};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat, expm, expm_frechet_adjoint, expm_mi_dt};

/// The dissipative half of a Lindblad time step: the channel `exp(dt·D)`, with
/// `D[ρ] = Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})` for the collapse operators `L_k`.
///
/// The channel is exact, so every step is completely positive and trace preserving: states stay physical and
/// fidelities at most 1 however long the step is compared with T1 and T2.  Alternating it with the unitary step is
/// first-order (Lie-Trotter) splitting in `dt`, the order of the explicit Euler step it replaces.
#[derive(Clone, Debug, PartialEq)]
pub struct LindbladOps {
    dim: usize,
    /// The non-zero entries `(to, from, value)` of `exp(dt·D)` acting on the row-major `vec(ρ)`.
    channel: Vec<(usize, usize, C<f64>)>,
}

impl LindbladOps {
    /// The channel for `collapse`, square operators of one dimension, over a step `dt`.
    ///
    /// Time runs forwards: a negative step is not a channel and would take populations below zero.
    pub fn new(collapse: &[CMat<f64>], dt: f64) -> Result<Self> {
        if !(dt >= 0.0 && dt.is_finite()) {
            return Err(Error::Config(format!(
                "a Lindblad step of {dt} seconds; it must be zero or more"
            )));
        }
        let dim = collapse
            .first()
            .ok_or_else(|| Error::Config("Lindblad evolution needs at least one collapse operator".into()))?
            .rows;
        if collapse.iter().any(|l| l.shape() != (dim, dim)) {
            return Err(Error::Dimension(
                "collapse operators must be square and of one size".into(),
            ));
        }
        // The generator column by column: column i·dim + j is vec(D[|i⟩⟨j|]).
        let n = dim * dim;
        let mut generator = CMat::<f64>::zeros(n, n);
        for from in 0..n {
            let mut e = CMat::<f64>::zeros(dim, dim);
            e.data[from] = C::new(dt, 0.0);
            for (to, v) in dissipator(collapse, &e)?.data.into_iter().enumerate() {
                generator.set(to, from, v);
            }
        }
        let map = expm(&generator)?;
        let channel = (0..n)
            .flat_map(|to| (0..n).map(move |from| (to, from)))
            .map(|(to, from)| (to, from, map.get(to, from)))
            .filter(|&(_, _, v)| v != C::new(0.0, 0.0))
            .collect();
        Ok(LindbladOps { dim, channel })
    }

    /// The channel applied to `rho`, or its adjoint (the Heisenberg-picture map) when `adjoint` is set.
    pub fn apply<T: Scalar>(&self, rho: &CMat<T>, adjoint: bool) -> Result<CMat<T>> {
        if rho.shape() != (self.dim, self.dim) {
            return Err(Error::Dimension(format!(
                "a {}x{} density matrix for {}-level collapse operators",
                rho.rows, rho.cols, self.dim
            )));
        }
        let mut out = CMat::<T>::zeros(self.dim, self.dim);
        for &(to, from, v) in &self.channel {
            if adjoint {
                out.data[from] += mulc(v.conj(), rho.data[to]);
            } else {
                out.data[to] += mulc(v, rho.data[from]);
            }
        }
        Ok(out)
    }
}

/// `D[ρ] = Σ_k (L ρ L† − ½{L†L, ρ})`.
fn dissipator(collapse: &[CMat<f64>], rho: &CMat<f64>) -> Result<CMat<f64>> {
    let mut out = CMat::<f64>::zeros(rho.rows, rho.cols);
    for l in collapse {
        let l_dag = l.adjoint();
        let l_dag_l = l_dag.matmul(l)?;
        out = out.add(&l.matmul(rho)?.matmul(&l_dag)?)?;
        out.axpy_re(-0.5, &l_dag_l.matmul(rho)?.add(&rho.matmul(&l_dag_l)?)?);
    }
    Ok(out)
}

impl<'a, T: Scalar> Tape<'a, T> {
    /// `re + i·im`.
    pub fn complex(&mut self, re: Var, im: Var) -> Result<Var> {
        let (a, b) = (self.value(re).r("complex")?, self.value(im).r("complex")?);
        if a.shape() != b.shape() {
            return Err(Error::Dimension(
                "complex: real and imaginary parts differ in shape".into(),
            ));
        }
        let out = CMat {
            rows: a.rows,
            cols: a.cols,
            data: a.data.iter().zip(&b.data).map(|(&x, &y)| C::new(x, y)).collect(),
        };
        Ok(self.push(Value::C(out), Op::Complex(re, im), &[re, im]))
    }

    /// The real part.
    pub fn real(&mut self, z: Var) -> Result<Var> {
        let out = self.value(z).c("real")?.map(|v| v.re);
        Ok(self.push(Value::R(out), Op::Real(z), &[z]))
    }

    /// The imaginary part.
    pub fn imag(&mut self, z: Var) -> Result<Var> {
        let out = self.value(z).c("imag")?.map(|v| v.im);
        Ok(self.push(Value::R(out), Op::Imag(z), &[z]))
    }

    /// Elementwise `z · c` for a constant `c` of the same shape.
    pub fn mul_const_c(&mut self, z: Var, c: &'a CMat<f64>) -> Result<Var> {
        let x = self.value(z).c("mul_const_c")?;
        if x.shape() != c.shape() {
            return Err(Error::Dimension("mul_const_c: shapes differ".into()));
        }
        let out = CMat {
            rows: x.rows,
            cols: x.cols,
            data: x.data.iter().zip(&c.data).map(|(&v, &k)| mulc(k, v)).collect(),
        };
        Ok(self.push(Value::C(out), Op::MulConstC(z, c), &[z]))
    }

    /// `H = H₀ + Σ_k u[row, k]·ops[k]` for a real `u` and constant `H₀`, `ops`.
    pub fn lincomb_row(&mut self, h0: &CMat<f64>, u: Var, row: usize, ops: &'a [CMat<f64>]) -> Result<Var> {
        let um = self.value(u).r("lincomb_row")?;
        if um.cols != ops.len() || row >= um.rows {
            return Err(Error::Dimension(format!(
                "lincomb_row: row {row} of a {}x{} control matrix with {} operators",
                um.rows,
                um.cols,
                ops.len()
            )));
        }
        let mut h = CMat::<T>::lift(h0);
        for (k, op) in ops.iter().enumerate() {
            if op.shape() != h.shape() {
                return Err(Error::Dimension("lincomb_row: operator shape differs from H0".into()));
            }
            let coeff = um.get(row, k);
            for (d, &o) in h.data.iter_mut().zip(&op.data) {
                d.re += coeff * T::from_f64(o.re);
                d.im += coeff * T::from_f64(o.im);
            }
        }
        Ok(self.push(Value::C(h), Op::LincombRow(u, row, ops), &[u]))
    }

    /// `U = exp(−i·dt·H)`.
    pub fn expm_mi_dt(&mut self, h: Var, dt: f64) -> Result<Var> {
        let out = expm_mi_dt(self.value(h).c("expm_mi_dt")?, T::from_f64(dt))?;
        Ok(self.push(Value::C(out), Op::ExpmMiDt(h, dt), &[h]))
    }

    /// Complex matrix product `a·b` (a matrix-vector product when `b` is a column).
    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var> {
        let out = self.value(a).c("matmul")?.matmul(self.value(b).c("matmul")?)?;
        Ok(self.push(Value::C(out), Op::Matmul(a, b), &[a, b]))
    }

    /// `U·ρ·U†`.
    pub fn sandwich(&mut self, u: Var, rho: Var) -> Result<Var> {
        let um = self.value(u).c("sandwich")?;
        let out = um.matmul(self.value(rho).c("sandwich")?)?.matmul(&um.adjoint())?;
        Ok(self.push(Value::C(out), Op::Sandwich(u, rho), &[u, rho]))
    }

    /// One dissipation step, `exp(dt·D)[ρ]`.
    pub fn lindblad_step(&mut self, rho: Var, ops: &'a LindbladOps) -> Result<Var> {
        let out = ops.apply(self.value(rho).c("lindblad_step")?, false)?;
        Ok(self.push(Value::C(out), Op::Lindblad(rho, ops), &[rho]))
    }

    /// The fidelity of the evolved columns `Ψ` to the target columns `T`, as a `1 × 1` real.
    ///
    /// With `k` columns - the images of `k` orthonormal states - it is the average fidelity over their span,
    /// `(‖M‖² + |Tr M|²)/(k·(k + 1))` with `M = T†·Ψ`, which counts population leaving the span as lost.  One column
    /// gives `|⟨t|ψ⟩|²`.
    pub fn fidelity(&mut self, target: &'a CMat<f64>, psi: Var) -> Result<Var> {
        let p = self.value(psi).c("fidelity")?;
        if p.shape() != target.shape() {
            return Err(Error::Dimension("fidelity: state and target differ in shape".into()));
        }
        let m = adjoint_times(target, p);
        let k = T::from_f64(target.cols as f64);
        let norm: T = m.data.iter().fold(T::zero(), |acc, z| acc + z.re * z.re + z.im * z.im);
        let s = m.trace();
        let out = RMat::from_vec(1, 1, vec![(norm + s.re * s.re + s.im * s.im) / (k * (k + T::one()))])?;
        Ok(self.push(Value::R(out), Op::Fidelity(target, psi), &[psi]))
    }

    /// `Re Tr(ρ·σ)`, as a `1 × 1` real.
    pub fn re_trace_product(&mut self, sigma: &'a CMat<f64>, rho: Var) -> Result<Var> {
        let r = self.value(rho).c("re_trace_product")?;
        if r.shape() != sigma.shape() || !r.is_square() {
            return Err(Error::Dimension("re_trace_product: shapes differ".into()));
        }
        let n = r.rows;
        let mut acc = T::zero();
        for i in 0..n {
            for j in 0..n {
                acc += mulc(sigma.get(j, i), r.get(i, j)).re;
            }
        }
        let out = RMat::from_vec(1, 1, vec![acc])?;
        Ok(self.push(Value::R(out), Op::ReTraceProduct(sigma, rho), &[rho]))
    }
}

/// `t†·p` for a constant `t`.
fn adjoint_times<T: Scalar>(t: &CMat<f64>, p: &CMat<T>) -> CMat<T> {
    let mut out = CMat::<T>::zeros(t.cols, p.cols);
    for r in 0..t.rows {
        for i in 0..t.cols {
            let a = t.get(r, i).conj();
            for j in 0..p.cols {
                out.data[i * p.cols + j] += mulc(a, p.get(r, j));
            }
        }
    }
    out
}

// ------------------------------------------------------------------ adjoints ----

pub(crate) fn complex_adjoint<T: Scalar>(re: Var, im: Var, g: &Value<T>) -> Result<Vec<(Var, Value<T>)>> {
    let g = g.c("complex")?;
    Ok(vec![(re, Value::R(g.map(|z| z.re))), (im, Value::R(g.map(|z| z.im)))])
}

/// Adjoint of taking the real (`imag == false`) or imaginary part.
pub(crate) fn real_adjoint<T: Scalar>(g: &Value<T>, imag: bool) -> Result<Value<T>> {
    let g = g.r("real/imag")?;
    Ok(Value::C(g.map(|v| {
        if imag {
            C::new(T::zero(), v)
        } else {
            C::new(v, T::zero())
        }
    })))
}

pub(crate) fn mul_const_c_adjoint<T: Scalar>(c: &CMat<f64>, g: &Value<T>) -> Result<Value<T>> {
    let g = g.c("mul_const_c")?;
    Ok(Value::C(CMat {
        rows: g.rows,
        cols: g.cols,
        data: g.data.iter().zip(&c.data).map(|(&v, &k)| mulc(k.conj(), v)).collect(),
    }))
}

pub(crate) fn lincomb_row_adjoint<T: Scalar>(
    u: &Value<T>,
    row: usize,
    ops: &[CMat<f64>],
    g: &Value<T>,
) -> Result<Value<T>> {
    let (rows, cols) = u.shape();
    let g = g.c("lincomb_row")?;
    let mut out = RMat::<T>::zeros(rows, cols);
    for (k, op) in ops.iter().enumerate() {
        // Re⟨op, G⟩ = Σ Re(conj(op)·G)
        let mut acc = T::zero();
        for (&o, &gv) in op.data.iter().zip(&g.data) {
            acc += T::from_f64(o.re) * gv.re + T::from_f64(o.im) * gv.im;
        }
        out.set(row, k, acc);
    }
    Ok(Value::R(out))
}

pub(crate) fn expm_mi_dt_adjoint<T: Scalar>(h: &Value<T>, dt: f64, g: &Value<T>) -> Result<Value<T>> {
    let h = h.c("expm_mi_dt")?;
    let a = h.scale(C::new(T::zero(), T::from_f64(-dt)));
    let ga = expm_frechet_adjoint(&a, g.c("expm_mi_dt")?)?;
    // A = −i·dt·H, so ∂L/∂H = conj(−i·dt)·∂L/∂A = i·dt·∂L/∂A.
    Ok(Value::C(ga.scale(C::new(T::zero(), T::from_f64(dt)))))
}

pub(crate) fn matmul_adjoint<T: Scalar>(
    a: Var,
    b: Var,
    va: &Value<T>,
    vb: &Value<T>,
    g: &Value<T>,
) -> Result<Vec<(Var, Value<T>)>> {
    let (x, y, g) = (va.c("matmul")?, vb.c("matmul")?, g.c("matmul")?);
    Ok(vec![
        (a, Value::C(g.matmul(&y.adjoint())?)),
        (b, Value::C(x.adjoint().matmul(g)?)),
    ])
}

pub(crate) fn sandwich_adjoint<T: Scalar>(
    u: Var,
    rho: Var,
    vu: &Value<T>,
    vr: &Value<T>,
    g: &Value<T>,
) -> Result<Vec<(Var, Value<T>)>> {
    let (um, r, g) = (vu.c("sandwich")?, vr.c("sandwich")?, g.c("sandwich")?);
    // W = U ρ U†:  ∂L/∂ρ = U† G U,   ∂L/∂U = G U ρᴴ + Gᴴ U ρ.
    let grad_rho = um.adjoint().matmul(g)?.matmul(um)?;
    let grad_u = g
        .matmul(um)?
        .matmul(&r.adjoint())?
        .add(&g.adjoint().matmul(um)?.matmul(r)?)?;
    Ok(vec![(u, Value::C(grad_u)), (rho, Value::C(grad_rho))])
}

pub(crate) fn lindblad_adjoint<T: Scalar>(ops: &LindbladOps, g: &Value<T>) -> Result<Value<T>> {
    Ok(Value::C(ops.apply(g.c("lindblad_step")?, true)?))
}

pub(crate) fn fidelity_adjoint<T: Scalar>(t: &CMat<f64>, psi: &Value<T>, g: &Value<T>) -> Result<Value<T>> {
    let m = adjoint_times(t, psi.c("fidelity")?);
    let s = m.trace();
    let k = t.cols as f64;
    let scale = g.r("fidelity")?.data[0] * T::from_f64(2.0 / (k * (k + 1.0)));
    // In the conjugate convention ∂‖T†Ψ‖²/∂Ψ = 2·T·M and ∂|Tr T†Ψ|²/∂Ψ = 2·s·T.
    let mut out = CMat::<T>::zeros(t.rows, t.cols);
    for r in 0..t.rows {
        for c in 0..t.cols {
            let mut acc = s * liftc(t.get(r, c));
            for i in 0..t.cols {
                acc += mulc(t.get(r, i), m.get(i, c));
            }
            out.data[r * t.cols + c] = C::new(acc.re * scale, acc.im * scale);
        }
    }
    Ok(Value::C(out))
}

pub(crate) fn re_trace_product_adjoint<T: Scalar>(sigma: &CMat<f64>, g: &Value<T>) -> Result<Value<T>> {
    let s = g.r("re_trace_product")?.data[0];
    // ∂ Re Tr(ρσ) / ∂ρ = σᴴ.
    Ok(Value::C(CMat::<T>::lift(&sigma.adjoint()).scale_re(s)))
}
