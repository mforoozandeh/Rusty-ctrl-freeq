//! Gradient checks for every tape operation.

use super::*;
use crate::error::{Error, Result};
use crate::linalg::{CMat, Mat, RMat};

/// Central-difference check of the tape gradient of `f` at `x`.
fn check_gradient<'a, F>(name: &str, f: F, x: &[f64], tol: f64)
where
    F: Fn(&mut Tape<'a, f64>, Var) -> Result<Var>,
{
    let eval = |xs: &[f64]| {
        let mut t = Tape::no_grad();
        let v = t.constant(Value::R(RMat::column(xs.to_vec())));
        let out = f(&mut t, v).unwrap();
        t.scalar(out).unwrap()
    };
    let mut t = Tape::new();
    let v = t.leaf(Value::R(RMat::column(x.to_vec())));
    let out = f(&mut t, v).unwrap();
    let grads = t.backward(out).unwrap();
    let g = grads.wrt_real(v, x.len());
    let h = 1e-6;
    for i in 0..x.len() {
        let mut p = x.to_vec();
        let mut m = x.to_vec();
        p[i] += h;
        m[i] -= h;
        let fd = (eval(&p) - eval(&m)) / (2.0 * h);
        let scale = fd.abs().max(g[i].abs()).max(1.0);
        assert!(
            (fd - g[i]).abs() <= tol * scale,
            "{name}: d/dx{i} tape {} vs fd {fd}",
            g[i]
        );
    }
}

/// A few pseudo-random numbers in (-1, 1).
fn numbers(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn hermitian(n: usize, seed: u64, scale: f64) -> CMat<f64> {
    let v = numbers(2 * n * n, seed);
    let m = Mat::from_fn(n, n, |r, c| C::new(v[2 * (r * n + c)], v[2 * (r * n + c) + 1]));
    m.add(&m.adjoint()).unwrap().scale_re(0.5 * scale)
}

fn unit_vector(n: usize, seed: u64) -> CMat<f64> {
    let v = numbers(2 * n, seed);
    let m = Mat::from_fn(n, 1, |r, _| C::new(v[2 * r], v[2 * r + 1]));
    let norm = m.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    m.scale_re(1.0 / norm)
}

fn pure_density(n: usize, seed: u64) -> CMat<f64> {
    let psi = unit_vector(n, seed);
    psi.matmul(&psi.adjoint()).unwrap()
}

#[test]
fn real_elementwise_ops() {
    let x = [0.3, -0.7, 1.2, 0.05];
    check_gradient(
        "square-sin-cos",
        |t, v| {
            let a = t.square(v)?;
            let b = t.sin(a)?;
            let c = t.cos(v)?;
            let d = t.mul(b, c)?;
            let e = t.add(d, v)?;
            let f = t.sub(e, a)?;
            let g = t.scale(f, 1.7)?;
            t.sum(g)
        },
        &x,
        1e-6,
    );
    check_gradient(
        "sqrt-atan2",
        |t, v| {
            let a = t.square(v)?;
            let s = t.sqrt(a)?;
            let y = t.sin(v)?;
            let z = t.cos(v)?;
            let at = t.atan2(y, z)?;
            let m = t.mul(s, at)?;
            t.mean(m)
        },
        &x,
        1e-6,
    );
}

#[test]
fn structural_ops() {
    let x = numbers(8, 3);
    let basis = RMat::from_vec(5, 4, numbers(20, 4)).unwrap();
    check_gradient(
        "slice-matmul-hstack-col",
        |t, v| {
            let a = t.slice(v, 0, 4)?;
            let b = t.slice(v, 4, 4)?;
            let ma = t.matmul_const(&basis, a)?;
            let mb = t.matmul_const(&basis, b)?;
            let sq = t.square(mb)?;
            let h = t.hstack(&[ma, sq])?;
            let c1 = t.col(h, 1)?;
            let c0 = t.col(h, 0)?;
            let p = t.mul(c0, c1)?;
            t.sum(p)
        },
        &x,
        1e-6,
    );
}

#[test]
fn amplitude_penalty_above_and_below_one() {
    check_gradient("penalty-active", |t, v| t.amp_penalty(v), &[0.2, -1.6, 0.9], 1e-6);
    check_gradient("penalty-inactive", |t, v| t.amp_penalty(v), &[0.2, -0.6, 0.9], 1e-6);
    let mut t: Tape<'_, f64> = Tape::new();
    let v = t.leaf(Value::R(RMat::column(vec![0.5, -2.0])));
    let p = t.amp_penalty(v).unwrap();
    assert_eq!(t.scalar(p).unwrap(), 1.0);
}

#[test]
fn complex_elementwise_ops() {
    let x = numbers(6, 5);
    let c = Mat::from_fn(3, 1, |r, _| C::new(0.3 * r as f64 + 0.1, -0.2 * r as f64 + 0.5));
    check_gradient(
        "complex-mulconst-real-imag",
        |t, v| {
            let a = t.slice(v, 0, 3)?;
            let b = t.slice(v, 3, 3)?;
            let z = t.complex(a, b)?;
            let w = t.mul_const_c(z, &c)?;
            let re = t.real(w)?;
            let im = t.imag(w)?;
            let r2 = t.square(re)?;
            let p = t.mul(r2, im)?;
            t.sum(p)
        },
        &x,
        1e-6,
    );
}

/// A few time steps of Hilbert-space propagation, the objective's inner loop.
fn hilbert_propagation(dim: usize) {
    let n_ctrl = 2;
    let steps = 3;
    let h0 = hermitian(dim, 11, 2.0);
    let ops: Vec<CMat<f64>> = (0..n_ctrl).map(|k| hermitian(dim, 20 + k as u64, 1.0)).collect();
    let psi0 = unit_vector(dim, 30);
    let target = unit_vector(dim, 31);
    let x = numbers(steps * n_ctrl, 7);
    // `x` holds the controls column by column: control k's values for every step, then control k + 1's.
    check_gradient(
        &format!("hilbert D={dim}"),
        |t, v| {
            let cols: Vec<Var> = (0..n_ctrl)
                .map(|k| t.slice(v, k * steps, steps))
                .collect::<Result<_>>()?;
            let umat = t.hstack(&cols)?;
            let mut psi = t.constant(Value::C(psi0.clone()));
            for s in 0..steps {
                let h = t.lincomb_row(&h0, umat, s, &ops)?;
                let uu = t.expm_mi_dt(h, 0.4)?;
                psi = t.matmul(uu, psi)?;
            }
            t.overlap_sq(&target, psi)
        },
        &x,
        1e-5,
    );
}

#[test]
fn hilbert_propagation_gradients() {
    hilbert_propagation(2);
    hilbert_propagation(4);
    hilbert_propagation(3);
}

#[test]
fn liouville_and_lindblad_gradients() {
    let dim = 2;
    let h0 = hermitian(dim, 41, 2.0);
    let ops: Vec<CMat<f64>> = (0..2).map(|k| hermitian(dim, 50 + k as u64, 1.0)).collect();
    let rho0 = pure_density(dim, 60);
    let sigma = pure_density(dim, 61);
    let collapse = vec![hermitian(dim, 70, 0.3), {
        let mut m = CMat::<f64>::zeros(2, 2);
        m.set(0, 1, C::new(0.5, 0.0));
        m
    }];
    let lops = LindbladOps::new(&collapse).unwrap();
    let x = numbers(4, 9);
    for dissipative in [false, true] {
        check_gradient(
            &format!("liouville dissipative={dissipative}"),
            |t, v| {
                let a = t.slice(v, 0, 2)?;
                let b = t.slice(v, 2, 2)?;
                let u = t.hstack(&[a, b])?;
                let mut rho = t.constant(Value::C(rho0.clone()));
                for s in 0..2 {
                    let h = t.lincomb_row(&h0, u, s, &ops)?;
                    let uu = t.expm_mi_dt(h, 0.3)?;
                    rho = t.sandwich(uu, rho)?;
                    if dissipative {
                        rho = t.lindblad_step(rho, &lops, 0.3)?;
                    }
                }
                t.re_trace_product(&sigma, rho)
            },
            &x,
            1e-5,
        );
    }
}

#[test]
fn batch_mean_matches_the_sequential_sum() {
    let h0s: Vec<CMat<f64>> = (0..7).map(|b| hermitian(2, 100 + b, 1.5)).collect();
    let op = vec![hermitian(2, 200, 1.0)];
    let psi0 = unit_vector(2, 201);
    let target = unit_vector(2, 202);
    let x = numbers(3, 13);

    fn element<'a>(
        t: &mut Tape<'a, f64>,
        u: Var,
        b: usize,
        h0s: &[CMat<f64>],
        op: &'a [CMat<f64>],
        psi0: &CMat<f64>,
        target: &'a CMat<f64>,
    ) -> Result<Var> {
        let mut psi = t.constant(Value::C(psi0.clone()));
        for s in 0..3 {
            let h = t.lincomb_row(&h0s[b], u, s, op)?;
            let uu = t.expm_mi_dt(h, 0.5)?;
            psi = t.matmul(uu, psi)?;
        }
        t.overlap_sq(target, psi)
    }

    // Batched.
    let mut t = Tape::new();
    let v = t.leaf(Value::R(RMat::column(x.clone())));
    let m = t
        .batch_mean(&[v], h0s.len(), |sub, inputs, b| {
            element(sub, inputs[0], b, &h0s, &op, &psi0, &target)
        })
        .unwrap();
    let batched = t.scalar(m).unwrap();
    let g_batched = t.backward(m).unwrap().wrt_real(v, 3);

    // Sequential, on one tape.
    let mut t2 = Tape::new();
    let v2 = t2.leaf(Value::R(RMat::column(x.clone())));
    let mut acc: Option<Var> = None;
    for b in 0..h0s.len() {
        let f = element(&mut t2, v2, b, &h0s, &op, &psi0, &target).unwrap();
        acc = Some(match acc {
            None => f,
            Some(a) => t2.add(a, f).unwrap(),
        });
    }
    let mean = t2.scale(acc.unwrap(), 1.0 / h0s.len() as f64).unwrap();
    let seq = t2.scalar(mean).unwrap();
    let g_seq = t2.backward(mean).unwrap().wrt_real(v2, 3);

    assert!((batched - seq).abs() < 1e-14);
    for i in 0..3 {
        assert!((g_batched[i] - g_seq[i]).abs() < 1e-13);
    }

    // No-grad batch gives the same value.
    let mut t3: Tape<'_, f64> = Tape::no_grad();
    let v3 = t3.constant(Value::R(RMat::column(x.clone())));
    let m3 = t3
        .batch_mean(&[v3], h0s.len(), |sub, inputs, b| {
            element(sub, inputs[0], b, &h0s, &op, &psi0, &target)
        })
        .unwrap();
    assert_eq!(t3.scalar(m3).unwrap(), batched);
}

/// Forward-over-reverse: running the reverse pass over duals seeded along `v` gives the Hessian-vector product.
#[test]
fn dual_reverse_pass_gives_hessian_vector_products() {
    let h0 = hermitian(2, 300, 2.0);
    let ops = vec![hermitian(2, 301, 1.0), hermitian(2, 302, 1.0)];
    let psi0 = unit_vector(2, 303);
    let target = unit_vector(2, 304);
    let x = numbers(4, 17);
    let dir = numbers(4, 18);

    fn build<'t, T: Scalar>(
        t: &mut Tape<'t, T>,
        v: Var,
        h0: &CMat<f64>,
        ops: &'t [CMat<f64>],
        psi0: &CMat<f64>,
        target: &'t CMat<f64>,
    ) -> Result<Var> {
        let a = t.slice(v, 0, 2)?;
        let b = t.slice(v, 2, 2)?;
        let u = t.hstack(&[a, b])?;
        let mut psi = t.constant(Value::C(CMat::<T>::lift(psi0)));
        for s in 0..2 {
            let h = t.lincomb_row(h0, u, s, ops)?;
            let uu = t.expm_mi_dt(h, 0.6)?;
            psi = t.matmul(uu, psi)?;
        }
        let f = t.overlap_sq(target, psi)?;
        let sq = t.square(v)?;
        let reg = t.sum(sq)?;
        let r = t.scale(reg, 0.1)?;
        t.add(f, r)
    }

    let grad_at = |xs: &[f64]| {
        let mut t = Tape::new();
        let v = t.leaf(Value::R(RMat::column(xs.to_vec())));
        let out = build(&mut t, v, &h0, &ops, &psi0, &target).unwrap();
        t.backward(out).unwrap().wrt_real(v, 4)
    };

    let mut td: Tape<'_, Dual> = Tape::new();
    let seeded: Vec<Dual> = x.iter().zip(&dir).map(|(&a, &d)| Dual::new(a, d)).collect();
    let vd = td.leaf(Value::R(RMat::column(seeded)));
    let out = build(&mut td, vd, &h0, &ops, &psi0, &target).unwrap();
    let gd = td.backward(out).unwrap().wrt_real(vd, 4);

    let g0 = grad_at(&x);
    let h = 1e-6;
    let plus: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a + h * d).collect();
    let minus: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a - h * d).collect();
    let (gp, gm) = (grad_at(&plus), grad_at(&minus));
    for i in 0..4 {
        assert!((gd[i].re - g0[i]).abs() < 1e-14, "gradient value part");
        let fd = (gp[i] - gm[i]) / (2.0 * h);
        assert!(
            (gd[i].eps - fd).abs() < 1e-6 * fd.abs().max(1.0),
            "Hv[{i}]: dual {} vs fd {fd}",
            gd[i].eps
        );
    }
}

#[test]
fn misuse_is_reported() {
    let mut t: Tape<'_, f64> = Tape::no_grad();
    let v = t.constant(Value::R(RMat::column(vec![1.0])));
    let s = t.sum(v).unwrap();
    assert!(matches!(t.backward(s), Err(Error::NotSupported(_))));

    let mut t: Tape<'_, f64> = Tape::new();
    let a = t.leaf(Value::R(RMat::column(vec![1.0, 2.0])));
    let b = t.leaf(Value::R(RMat::column(vec![1.0, 2.0, 3.0])));
    assert!(matches!(t.add(a, b), Err(Error::Dimension(_))));
    assert!(matches!(t.backward(a), Err(Error::Dimension(_))));
    assert!(matches!(t.real(a), Err(Error::Dimension(_))));
}
