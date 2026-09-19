//! Fixed-frequency transmons truncated to two levels, in the per-qubit rotating frame.

use super::{ControlChannel, HamiltonianModel, Source};
use crate::error::Result;
use crate::linalg::{CMat, RMat};
use crate::setup::spin_ops;

/// Two-level transmons.
///
/// `H0 = Σ δ_q·Z_q + Σ_{i<j} g_ij·(X_i·X_j + Y_i·Y_j) + Σ_{i<j} ζ_ij·Z_i·Z_j`, with the exchange term present when
/// the coupling type contains `XY` and the static ZZ term when it contains `ZZ`.  `ζ` is the calibrated
/// `zz_crosstalk` matrix if given, otherwise `2·g²·(1/α_i + 1/α_j)` from the anharmonicities, otherwise zero.
///
/// Controls are I/Q drives `Ω_q·(cx_q·X_q + cy_q·Y_q)`.  With Stark coefficients `s_q` each qubit gets a third
/// channel `s_q·Ω_q²·(cx_q² + cy_q²)·Z_q`, the drive-induced frequency shift.
pub struct Superconducting {
    n: usize,
    coupling_type: String,
    anharmonicities: Option<Vec<f64>>,
    zz_crosstalk: Option<RMat<f64>>,
    stark: Option<Vec<f64>>,
    ops: Vec<[CMat<f64>; 3]>,
}

impl Superconducting {
    /// `n` transmons.  Frequencies are in rad/s; `stark` is dimensionless.
    pub fn new(
        n: usize,
        coupling_type: String,
        anharmonicities: Option<Vec<f64>>,
        zz_crosstalk: Option<RMat<f64>>,
        stark: Option<Vec<f64>>,
    ) -> Self {
        Superconducting {
            n,
            coupling_type,
            anharmonicities,
            zz_crosstalk,
            stark,
            ops: spin_ops(n),
        }
    }
}

impl HamiltonianModel for Superconducting {
    fn name(&self) -> &'static str {
        "superconducting"
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
        if self.n < 2 {
            return Ok(h);
        }
        let g = coupling.filter(|_| self.n > 1);
        if let (Some(g), true) = (g, self.coupling_type.contains("XY")) {
            for a in 0..self.n {
                for b in a + 1..self.n {
                    let gab = g.get(a, b);
                    if gab != 0.0 {
                        let xx = self.ops[a][0].matmul(&self.ops[b][0])?;
                        let yy = self.ops[a][1].matmul(&self.ops[b][1])?;
                        h.axpy_re(gab, &xx.add(&yy)?);
                    }
                }
            }
        }
        if self.coupling_type.contains("ZZ") {
            for a in 0..self.n {
                for b in a + 1..self.n {
                    let zeta = match (&self.zz_crosstalk, &self.anharmonicities, g) {
                        (Some(zz), _, _) => zz.get(a, b),
                        (None, Some(alpha), Some(g)) if alpha[a] != 0.0 && alpha[b] != 0.0 => {
                            2.0 * g.get(a, b).powi(2) * (1.0 / alpha[a] + 1.0 / alpha[b])
                        }
                        _ => 0.0,
                    };
                    if zeta != 0.0 {
                        h.axpy_re(zeta, &self.ops[a][2].matmul(&self.ops[b][2])?);
                    }
                }
            }
        }
        Ok(h)
    }

    fn control_ops(&self) -> Vec<CMat<f64>> {
        self.ops
            .iter()
            .flat_map(|o| {
                let mut v = vec![o[0].clone(), o[1].clone()];
                if self.stark.is_some() {
                    v.push(o[2].clone());
                }
                v
            })
            .collect()
    }

    fn control_channels(&self) -> Vec<ControlChannel> {
        (0..self.n)
            .flat_map(|q| {
                let mut v = vec![
                    ControlChannel::drive(q, Source::Cx),
                    ControlChannel::drive(q, Source::Cy),
                ];
                if let Some(s) = &self.stark {
                    v.push(ControlChannel {
                        qubit: q,
                        source: Source::Power,
                        rabi_power: 2,
                        coeff: s[q],
                    });
                }
                v
            })
            .collect()
    }
}
