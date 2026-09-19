//! Three-level transmons (Duffing oscillators), which show leakage out of the computational subspace.

use super::{ControlChannel, HamiltonianModel, Source, embed_local};
use crate::autodiff::C;
use crate::error::{Error, Result};
use crate::linalg::{CMat, Mat, RMat};

/// Three-level transmons in the per-qubit rotating frame.
///
/// `H0 = Σ [δ_q·n_q + (α_q/2)·(n_q² − n_q)] + Σ_{i<j} g_ij·(a_i†·a_j + a_i·a_j†)`; controls
/// `Ω_q·[cx_q·(a_q + a_q†)/2 + cy_q·(−i·a_q + i·a_q†)/2]`.  The space has dimension `3ⁿ`; computational states and
/// gates are embedded by mapping each qubit's `|0⟩, |1⟩` to the transmon's.
pub struct DuffingTransmon {
    n: usize,
    alpha: Vec<f64>,
    a: Vec<CMat<f64>>,
    adag: Vec<CMat<f64>>,
    number: Vec<CMat<f64>>,
    x: Vec<CMat<f64>>,
    y: Vec<CMat<f64>>,
    /// The `3ⁿ × 2ⁿ` embedding of the computational basis.
    p: CMat<f64>,
}

impl DuffingTransmon {
    /// `n` transmons with anharmonicities `alpha` in rad/s.
    pub fn new(n: usize, alpha: Vec<f64>) -> Result<Self> {
        if alpha.len() != n {
            return Err(Error::Config(format!(
                "anharmonicities has {} entries for {n} qubits",
                alpha.len()
            )));
        }
        let (o, s2) = (C::new(0.0, 0.0), C::new(2f64.sqrt(), 0.0));
        let l = C::new(1.0, 0.0);
        let a_local = Mat::from_vec(3, 3, vec![o, l, o, o, o, s2, o, o, o])?;
        let adag_local = a_local.adjoint();
        let n_local = adag_local.matmul(&a_local)?;
        let x_local = a_local.add(&adag_local)?.scale_re(0.5);
        let i = C::new(0.0, 1.0);
        let y_local = a_local.scale(-i).add(&adag_local.scale(i))?.scale_re(0.5);
        let site = |m: &CMat<f64>| (0..n).map(|q| embed_local(m, q, n)).collect::<Vec<_>>();

        let d_comp = 1usize << n;
        let d_full = 3usize.pow(n as u32);
        let mut p = CMat::<f64>::zeros(d_full, d_comp);
        for c in 0..d_comp {
            let full: usize = (0..n).filter(|b| c & (1 << b) != 0).map(|b| 3usize.pow(b as u32)).sum();
            p.set(full, c, l);
        }
        Ok(DuffingTransmon {
            n,
            alpha,
            a: site(&a_local),
            adag: site(&adag_local),
            number: site(&n_local),
            x: site(&x_local),
            y: site(&y_local),
            p,
        })
    }

    /// Population outside the computational subspace for a state vector in the model's space.
    pub fn leakage(&self, psi: &CMat<f64>) -> f64 {
        let comp: f64 = (0..self.p.cols)
            .map(|c| {
                let r = (0..self.p.rows).find(|&r| self.p.get(r, c).re != 0.0).unwrap_or(0);
                psi.get(r, 0).norm_sqr()
            })
            .sum();
        1.0 - comp
    }
}

impl HamiltonianModel for DuffingTransmon {
    fn name(&self) -> &'static str {
        "duffing_transmon"
    }

    fn dim(&self) -> usize {
        3usize.pow(self.n as u32)
    }

    fn drift(&self, offsets: &[f64], coupling: Option<&RMat<f64>>) -> Result<CMat<f64>> {
        let d = self.dim();
        let mut h = CMat::<f64>::zeros(d, d);
        for (q, &delta) in offsets.iter().enumerate().take(self.n) {
            h.axpy_re(delta, &self.number[q]);
            let nn = self.number[q].matmul(&self.number[q])?.sub(&self.number[q])?;
            h.axpy_re(self.alpha[q] / 2.0, &nn);
        }
        if let (Some(g), true) = (coupling, self.n > 1) {
            for i in 0..self.n {
                for j in i + 1..self.n {
                    let gij = g.get(i, j);
                    if gij != 0.0 {
                        let hop = self.adag[i]
                            .matmul(&self.a[j])?
                            .add(&self.a[i].matmul(&self.adag[j])?)?;
                        h.axpy_re(gij, &hop);
                    }
                }
            }
        }
        Ok(h)
    }

    fn control_ops(&self) -> Vec<CMat<f64>> {
        (0..self.n)
            .flat_map(|q| [self.x[q].clone(), self.y[q].clone()])
            .collect()
    }

    fn control_channels(&self) -> Vec<ControlChannel> {
        (0..self.n)
            .flat_map(|q| {
                [
                    ControlChannel::drive(q, Source::Cx),
                    ControlChannel::drive(q, Source::Cy),
                ]
            })
            .collect()
    }

    fn embed_state(&self, psi: &CMat<f64>) -> Result<CMat<f64>> {
        self.p.matmul(psi)
    }

    fn embed_density(&self, rho: &CMat<f64>) -> Result<CMat<f64>> {
        self.p.matmul(rho)?.matmul(&self.p.adjoint())
    }

    fn embed_gate(&self, g: &CMat<f64>) -> Result<CMat<f64>> {
        let pp = self.p.matmul(&self.p.adjoint())?;
        let rest = CMat::<f64>::identity(self.dim()).sub(&pp)?;
        self.p.matmul(g)?.matmul(&self.p.adjoint())?.add(&rest)
    }
}
