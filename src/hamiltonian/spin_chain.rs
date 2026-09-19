//! Spin chains (NMR, spin qubits).

use super::{ControlChannel, HamiltonianModel, Source};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat};
use crate::setup::spin_ops;

/// Spin-spin coupling form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coupling {
    /// `J·Z_i·Z_j`, weak coupling.
    Z,
    /// `J·(X_i·X_j + Y_i·Y_j)`.
    Xy,
    /// `J·(X_i·X_j + Y_i·Y_j + Z_i·Z_j)`, strong (isotropic) coupling.
    Xyz,
}

impl Coupling {
    /// Parse `Z`, `XY` or `XYZ`, ignoring case as the Python package does.
    pub fn parse(s: &str) -> Result<Coupling> {
        match s.to_uppercase().as_str() {
            "Z" => Ok(Coupling::Z),
            "XY" => Ok(Coupling::Xy),
            "XYZ" => Ok(Coupling::Xyz),
            _ => Err(Error::Config(format!("coupling_type \"{s}\" is not one of Z, XY, XYZ"))),
        }
    }
}

/// `H0 = Σ Δ_q·Z_q + Σ_{i<j} J_ij·C_ij`, controls `Ω_q·(cx_q·X_q + cy_q·Y_q)`, with `X, Y, Z` the spin-½
/// operators.
pub struct SpinChain {
    n: usize,
    coupling: Coupling,
    ops: Vec<[CMat<f64>; 3]>,
}

impl SpinChain {
    /// A chain of `n` spins with the given coupling form.
    pub fn new(n: usize, coupling: Coupling) -> Self {
        SpinChain {
            n,
            coupling,
            ops: spin_ops(n),
        }
    }
}

impl HamiltonianModel for SpinChain {
    fn name(&self) -> &'static str {
        "spin_chain"
    }

    fn dim(&self) -> usize {
        1 << self.n
    }

    fn drift(&self, offsets: &[f64], coupling: Option<&RMat<f64>>) -> Result<CMat<f64>> {
        let d = self.dim();
        let mut h = CMat::<f64>::zeros(d, d);
        for (q, &delta) in offsets.iter().enumerate().take(self.n) {
            h.axpy_re(delta, &self.ops[q][2]);
        }
        if let (Some(j), true) = (coupling, self.n > 1) {
            for a in 0..self.n {
                for b in a + 1..self.n {
                    let jab = j.get(a, b);
                    if jab == 0.0 {
                        continue;
                    }
                    let axes: &[usize] = match self.coupling {
                        Coupling::Z => &[2],
                        Coupling::Xy => &[0, 1],
                        Coupling::Xyz => &[0, 1, 2],
                    };
                    for &k in axes {
                        h.axpy_re(jab, &self.ops[a][k].matmul(&self.ops[b][k])?);
                    }
                }
            }
        }
        Ok(h)
    }

    fn control_ops(&self) -> Vec<CMat<f64>> {
        self.ops.iter().flat_map(|o| [o[0].clone(), o[1].clone()]).collect()
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
}
