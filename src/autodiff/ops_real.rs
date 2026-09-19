//! Operations on real matrices, and their adjoints.

use super::Scalar;
use super::tape::{Op, Tape, Value, Var};
use crate::error::{Error, Result};
use crate::linalg::RMat;

fn same_shape<T: Scalar>(a: &RMat<T>, b: &RMat<T>, what: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::Dimension(format!(
            "{what}: shapes {}x{} and {}x{} differ",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    Ok(())
}

fn zip<T: Scalar>(a: &RMat<T>, b: &RMat<T>, f: impl Fn(T, T) -> T) -> RMat<T> {
    RMat {
        rows: a.rows,
        cols: a.cols,
        data: a.data.iter().zip(&b.data).map(|(&x, &y)| f(x, y)).collect(),
    }
}

impl<'a, T: Scalar> Tape<'a, T> {
    fn unary(&mut self, a: Var, what: &str, f: impl Fn(T) -> T, op: Op<'a, T>) -> Result<Var> {
        let out = self.value(a).r(what)?.map(f);
        Ok(self.push(Value::R(out), op, &[a]))
    }

    fn binary(&mut self, a: Var, b: Var, what: &str, f: impl Fn(T, T) -> T, op: Op<'a, T>) -> Result<Var> {
        let (x, y) = (self.value(a).r(what)?, self.value(b).r(what)?);
        same_shape(x, y, what)?;
        let out = zip(x, y, f);
        Ok(self.push(Value::R(out), op, &[a, b]))
    }

    /// Elementwise `a + b`.
    pub fn add(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(a, b, "add", |x, y| x + y, Op::Add(a, b))
    }

    /// Elementwise `a - b`.
    pub fn sub(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(a, b, "sub", |x, y| x - y, Op::Sub(a, b))
    }

    /// Elementwise `a · b`.
    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(a, b, "mul", |x, y| x * y, Op::Mul(a, b))
    }

    /// `s · a`.
    pub fn scale(&mut self, a: Var, s: f64) -> Result<Var> {
        let st = T::from_f64(s);
        self.unary(a, "scale", |x| x * st, Op::Scale(a, s))
    }

    /// Elementwise `a²`.
    pub fn square(&mut self, a: Var) -> Result<Var> {
        self.unary(a, "square", |x| x * x, Op::Square(a))
    }

    /// Elementwise square root; its derivative at zero is taken as zero.
    pub fn sqrt(&mut self, a: Var) -> Result<Var> {
        self.unary(a, "sqrt", |x| x.sqrt(), Op::Sqrt(a))
    }

    /// Elementwise cosine.
    pub fn cos(&mut self, a: Var) -> Result<Var> {
        self.unary(a, "cos", |x| x.cos(), Op::Cos(a))
    }

    /// Elementwise sine.
    pub fn sin(&mut self, a: Var) -> Result<Var> {
        self.unary(a, "sin", |x| x.sin(), Op::Sin(a))
    }

    /// Elementwise four-quadrant `atan2(y, x)`.
    pub fn atan2(&mut self, y: Var, x: Var) -> Result<Var> {
        self.binary(y, x, "atan2", |a, b| a.atan2(b), Op::Atan2(y, x))
    }

    /// `m · a` for a constant `m`.
    pub fn matmul_const(&mut self, m: &'a RMat<f64>, a: Var) -> Result<Var> {
        let x = self.value(a).r("matmul_const")?;
        let out = RMat::<T>::lift(m).matmul(x)?;
        Ok(self.push(Value::R(out), Op::MatmulConst(m, a), &[a]))
    }

    /// Entries `start..start + len` of a column vector.
    pub fn slice(&mut self, a: Var, start: usize, len: usize) -> Result<Var> {
        let x = self.value(a).r("slice")?;
        if x.cols != 1 || start + len > x.rows {
            return Err(Error::Dimension(format!(
                "slice {start}..{} of a {}x{} matrix",
                start + len,
                x.rows,
                x.cols
            )));
        }
        let out = RMat::column(x.data[start..start + len].to_vec());
        Ok(self.push(Value::R(out), Op::Slice(a, start), &[a]))
    }

    /// Column `c`.
    pub fn col(&mut self, a: Var, c: usize) -> Result<Var> {
        let x = self.value(a).r("col")?;
        if c >= x.cols {
            return Err(Error::Dimension(format!(
                "column {c} of a {}x{} matrix",
                x.rows, x.cols
            )));
        }
        let out = RMat::column(x.col(c));
        Ok(self.push(Value::R(out), Op::Col(a, c), &[a]))
    }

    /// Place matrices with equal row counts side by side.
    pub fn hstack(&mut self, parts: &[Var]) -> Result<Var> {
        let values: Vec<&RMat<T>> = parts
            .iter()
            .map(|&p| self.value(p).r("hstack"))
            .collect::<Result<_>>()?;
        let rows = values.first().map_or(0, |m| m.rows);
        if values.iter().any(|m| m.rows != rows) {
            return Err(Error::Dimension("hstack of matrices with different row counts".into()));
        }
        let cols: usize = values.iter().map(|m| m.cols).sum();
        let mut out = RMat::<T>::zeros(rows, cols);
        let mut offset = 0;
        for m in &values {
            for r in 0..rows {
                for c in 0..m.cols {
                    out.set(r, offset + c, m.get(r, c));
                }
            }
            offset += m.cols;
        }
        Ok(self.push(Value::R(out), Op::HStack(parts.to_vec()), parts))
    }

    /// `(max|a| − 1)²` when `max|a| > 1`, otherwise `0` with no gradient.  The Python package's amplitude penalty.
    pub fn amp_penalty(&mut self, a: Var) -> Result<Var> {
        let x = self.value(a).r("amp_penalty")?;
        let mut best: Option<(usize, T)> = None;
        for (i, &v) in x.data.iter().enumerate() {
            let m = v.abs();
            if best.is_none_or(|(_, b)| m.re() > b.re()) {
                best = Some((i, m));
            }
        }
        let (value, arg) = match best {
            Some((i, m)) if m.re() > 1.0 => {
                let d = m - T::one();
                (d * d, Some(i))
            }
            _ => (T::zero(), None),
        };
        let out = RMat::from_vec(1, 1, vec![value])?;
        Ok(self.push(Value::R(out), Op::AmpPenalty(a, arg), &[a]))
    }

    /// Sum of all entries, as a `1 × 1` matrix.
    pub fn sum(&mut self, a: Var) -> Result<Var> {
        let x = self.value(a).r("sum")?;
        let s = x.data.iter().fold(T::zero(), |acc, &v| acc + v);
        Ok(self.push(Value::R(RMat::from_vec(1, 1, vec![s])?), Op::Sum(a), &[a]))
    }

    /// Mean of all entries, as a `1 × 1` matrix.
    pub fn mean(&mut self, a: Var) -> Result<Var> {
        let x = self.value(a).r("mean")?;
        let n = T::from_f64(x.data.len() as f64);
        let s = x.data.iter().fold(T::zero(), |acc, &v| acc + v) / n;
        Ok(self.push(Value::R(RMat::from_vec(1, 1, vec![s])?), Op::Mean(a), &[a]))
    }
}

// ------------------------------------------------------------------ adjoints ----

pub(crate) fn mul_adjoint<T: Scalar>(
    a: Var,
    b: Var,
    va: &Value<T>,
    vb: &Value<T>,
    g: &Value<T>,
) -> Result<Vec<(Var, Value<T>)>> {
    let (x, y, g) = (va.r("mul")?, vb.r("mul")?, g.r("mul")?);
    Ok(vec![
        (a, Value::R(zip(g, y, |g, y| g * y))),
        (b, Value::R(zip(g, x, |g, x| g * x))),
    ])
}

pub(crate) fn square_adjoint<T: Scalar>(va: &Value<T>, g: &Value<T>) -> Result<Value<T>> {
    let two = T::from_f64(2.0);
    Ok(Value::R(zip(g.r("square")?, va.r("square")?, |g, x| two * x * g)))
}

pub(crate) fn sqrt_adjoint<T: Scalar>(out: &Value<T>, g: &Value<T>) -> Result<Value<T>> {
    let half = T::from_f64(0.5);
    Ok(Value::R(zip(g.r("sqrt")?, out.r("sqrt")?, |g, s| {
        if s.re() == 0.0 { T::zero() } else { half * g / s }
    })))
}

pub(crate) fn cos_adjoint<T: Scalar>(va: &Value<T>, g: &Value<T>) -> Result<Value<T>> {
    Ok(Value::R(zip(g.r("cos")?, va.r("cos")?, |g, x| -(x.sin()) * g)))
}

pub(crate) fn sin_adjoint<T: Scalar>(va: &Value<T>, g: &Value<T>) -> Result<Value<T>> {
    Ok(Value::R(zip(g.r("sin")?, va.r("sin")?, |g, x| x.cos() * g)))
}

pub(crate) fn atan2_adjoint<T: Scalar>(
    y: Var,
    x: Var,
    vy: &Value<T>,
    vx: &Value<T>,
    g: &Value<T>,
) -> Result<Vec<(Var, Value<T>)>> {
    let (ym, xm, g) = (vy.r("atan2")?, vx.r("atan2")?, g.r("atan2")?);
    let n = ym.data.len();
    let mut gy = RMat::<T>::zeros(ym.rows, ym.cols);
    let mut gx = RMat::<T>::zeros(ym.rows, ym.cols);
    for i in 0..n {
        let (a, b) = (ym.data[i], xm.data[i]);
        let r2 = a * a + b * b;
        if r2.re() != 0.0 {
            gy.data[i] = g.data[i] * b / r2;
            gx.data[i] = -(g.data[i] * a / r2);
        }
    }
    Ok(vec![(y, Value::R(gy)), (x, Value::R(gx))])
}

pub(crate) fn matmul_const_adjoint<T: Scalar>(m: &RMat<f64>, g: &Value<T>) -> Result<Value<T>> {
    Ok(Value::R(RMat::<T>::lift(&m.transpose()).matmul(g.r("matmul_const")?)?))
}

pub(crate) fn slice_adjoint<T: Scalar>(va: &Value<T>, start: usize, g: &Value<T>) -> Result<Value<T>> {
    let (rows, cols) = va.shape();
    let mut out = RMat::<T>::zeros(rows, cols);
    let g = g.r("slice")?;
    out.data[start..start + g.data.len()].copy_from_slice(&g.data);
    Ok(Value::R(out))
}

pub(crate) fn col_adjoint<T: Scalar>(va: &Value<T>, c: usize, g: &Value<T>) -> Result<Value<T>> {
    let (rows, cols) = va.shape();
    let mut out = RMat::<T>::zeros(rows, cols);
    for (r, &v) in g.r("col")?.data.iter().enumerate() {
        out.set(r, c, v);
    }
    Ok(Value::R(out))
}

pub(crate) fn hstack_adjoint<T: Scalar>(
    parts: &[Var],
    cols_of: impl Fn(Var) -> usize,
    g: &Value<T>,
) -> Result<Vec<(Var, Value<T>)>> {
    let g = g.r("hstack")?;
    let mut offset = 0;
    let mut out = Vec::with_capacity(parts.len());
    for &p in parts {
        let cols = cols_of(p);
        out.push((p, Value::R(RMat::from_fn(g.rows, cols, |r, c| g.get(r, offset + c)))));
        offset += cols;
    }
    Ok(out)
}

pub(crate) fn amp_penalty_adjoint<T: Scalar>(va: &Value<T>, arg: Option<usize>, g: &Value<T>) -> Result<Value<T>> {
    let x = va.r("amp_penalty")?;
    let mut out = RMat::<T>::zeros(x.rows, x.cols);
    if let Some(i) = arg {
        let v = x.data[i];
        let m = v.abs();
        let sign = if v.re() >= 0.0 { T::one() } else { -T::one() };
        out.data[i] = T::from_f64(2.0) * (m - T::one()) * sign * g.r("amp_penalty")?.data[0];
    }
    Ok(Value::R(out))
}

pub(crate) fn sum_adjoint<T: Scalar>(va: &Value<T>, g: &Value<T>, factor: T) -> Result<Value<T>> {
    let (rows, cols) = va.shape();
    let s = g.r("sum")?.data[0] * factor;
    Ok(Value::R(RMat {
        rows,
        cols,
        data: vec![s; rows * cols],
    }))
}
