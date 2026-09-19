//! Operations on complex matrices - the propagation's building blocks - and their adjoints.
//!
//! Adjoints use the conjugate convention of the tape: for `W = A·B`, `∂L/∂A = G·Bᴴ` and `∂L/∂B = Aᴴ·G`.

use super::tape::{Op, Tape, Value, Var, liftc, mulc};
use super::{C, Scalar};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat, expm_frechet_adjoint, expm_mi_dt};

/// Lindblad collapse operators with the products the dissipator needs, computed once.
#[derive(Clone, Debug, PartialEq)]
pub struct LindbladOps {
    /// The collapse operators `L_k`.
    pub l: Vec<CMat<f64>>,
    /// `L_k†`.
    pub l_dag: Vec<CMat<f64>>,
    /// `L_k†·L_k`.
    pub l_dag_l: Vec<CMat<f64>>,
}

impl LindbladOps {
    /// Precompute the products for `collapse`.
    pub fn new(collapse: &[CMat<f64>]) -> Result<Self> {
        let l = collapse.to_vec();
        let l_dag: Vec<CMat<f64>> = l.iter().map(|m| m.adjoint()).collect();
        let l_dag_l = l_dag.iter().zip(&l).map(|(a, b)| a.matmul(b)).collect::<Result<_>>()?;
        Ok(LindbladOps { l, l_dag, l_dag_l })
    }

    /// `D[ρ] = Σ_k (L ρ L† − ½{L†L, ρ})`, or its adjoint `Σ_k (L† ρ L − ½{L†L, ρ})` when `adjoint` is set.
    pub(crate) fn dissipator<T: Scalar>(&self, rho: &CMat<T>, adjoint: bool) -> Result<CMat<T>> {
        let n = rho.rows;
        let mut out = CMat::<T>::zeros(n, n);
        let half = T::from_f64(0.5);
        for k in 0..self.l.len() {
            let (left, right) = if adjoint {
                (&self.l_dag[k], &self.l[k])
            } else {
                (&self.l[k], &self.l_dag[k])
            };
            let jump = mat_cf(&mat_fc(left, rho)?, right)?;
            let anti = mat_fc(&self.l_dag_l[k], rho)?.add(&mat_cf(rho, &self.l_dag_l[k])?)?;
            out = out.add(&jump)?;
            out.axpy_re(-half, &anti);
        }
        Ok(out)
    }
}

/// `a·b` with a constant `f64` left factor.
pub(crate) fn mat_fc<T: Scalar>(a: &CMat<f64>, b: &CMat<T>) -> Result<CMat<T>> {
    if a.cols != b.rows {
        return Err(Error::Dimension(format!(
            "cannot multiply {}x{} by {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let mut out = CMat::<T>::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let x = a.get(i, k);
            if x.re == 0.0 && x.im == 0.0 {
                continue;
            }
            for j in 0..b.cols {
                let idx = i * b.cols + j;
                out.data[idx] += mulc(x, b.get(k, j));
            }
        }
    }
    Ok(out)
}

/// `a·b` with a constant `f64` right factor.
pub(crate) fn mat_cf<T: Scalar>(a: &CMat<T>, b: &CMat<f64>) -> Result<CMat<T>> {
    if a.cols != b.rows {
        return Err(Error::Dimension(format!(
            "cannot multiply {}x{} by {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let mut out = CMat::<T>::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let x = a.get(i, k);
            for j in 0..b.cols {
                let y = b.get(k, j);
                if y.re == 0.0 && y.im == 0.0 {
                    continue;
                }
                out.data[i * b.cols + j] += mulc(y, x);
            }
        }
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

    /// One explicit Euler dissipation step, `ρ + dt·D[ρ]`.
    pub fn lindblad_step(&mut self, rho: Var, ops: &'a LindbladOps, dt: f64) -> Result<Var> {
        let r = self.value(rho).c("lindblad_step")?;
        let mut out = r.clone();
        out.axpy_re(T::from_f64(dt), &ops.dissipator(r, false)?);
        Ok(self.push(Value::C(out), Op::Lindblad(rho, ops, dt), &[rho]))
    }

    /// `|⟨target|ψ⟩|²` for column vectors, as a `1 × 1` real.
    pub fn overlap_sq(&mut self, target: &'a CMat<f64>, psi: Var) -> Result<Var> {
        let p = self.value(psi).c("overlap_sq")?;
        if p.shape() != target.shape() {
            return Err(Error::Dimension("overlap_sq: state and target differ in shape".into()));
        }
        let s = overlap(target, p);
        let out = RMat::from_vec(1, 1, vec![s.re * s.re + s.im * s.im])?;
        Ok(self.push(Value::R(out), Op::OverlapSq(target, psi), &[psi]))
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

/// `⟨t|p⟩ = Σ conj(t_i)·p_i`.
fn overlap<T: Scalar>(t: &CMat<f64>, p: &CMat<T>) -> C<T> {
    t.data
        .iter()
        .zip(&p.data)
        .fold(C::new(T::zero(), T::zero()), |acc, (&a, &b)| acc + mulc(a.conj(), b))
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

pub(crate) fn lindblad_adjoint<T: Scalar>(ops: &LindbladOps, dt: f64, g: &Value<T>) -> Result<Value<T>> {
    let g = g.c("lindblad_step")?;
    let mut out = g.clone();
    out.axpy_re(T::from_f64(dt), &ops.dissipator(g, true)?);
    Ok(Value::C(out))
}

pub(crate) fn overlap_sq_adjoint<T: Scalar>(t: &CMat<f64>, psi: &Value<T>, g: &Value<T>) -> Result<Value<T>> {
    let p = psi.c("overlap_sq")?;
    let s = overlap(t, p);
    let scale = g.r("overlap_sq")?.data[0] * T::from_f64(2.0);
    // ∂|s|²/∂ψ = 2·s·t in the conjugate convention.
    let s2 = C::new(s.re * scale, s.im * scale);
    Ok(Value::C(t.map(|v| s2 * liftc(v))))
}

pub(crate) fn re_trace_product_adjoint<T: Scalar>(sigma: &CMat<f64>, g: &Value<T>) -> Result<Value<T>> {
    let s = g.r("re_trace_product")?.data[0];
    // ∂ Re Tr(ρσ) / ∂ρ = σᴴ.
    Ok(Value::C(CMat::<T>::lift(&sigma.adjoint()).scale_re(s)))
}
