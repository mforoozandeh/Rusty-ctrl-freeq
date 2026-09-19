//! Fixed-frequency transmons truncated to two levels, in the per-qubit rotating frame.

use super::{ControlChannel, HamiltonianModel, Source};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat};
use crate::setup::spin_ops;

/// Two-level transmons: the [`DuffingTransmon`](super::DuffingTransmon) model restricted to `|0⟩, |1⟩`, so one
/// configuration describes the same device in either model.
///
/// `H0 = −Σ δ_q·Z_q + Σ_{i<j} 2·g_ij·(X_i·X_j + Y_i·Y_j) + Σ_{i<j} ζ_ij·Z_i·Z_j` with `X, Y, Z = σ/2`.  Qubit `q`'s
/// frequency is `δ_q` above the frame (`−δ·Z = δ·n` up to a constant) and `g` is the hopping between `|01⟩` and
/// `|10⟩`, `g·(σ⁺σ⁻ + σ⁻σ⁺)`.  The exchange term is present when the coupling type contains `XY` and the static ZZ
/// term when it contains `ZZ`.  `ζ` is the calibrated `zz_crosstalk` matrix if given, otherwise the second-order
/// estimate `2g²·(α_i + α_j)/((Δ + α_i)·(Δ − α_j))` with `Δ = δ_i − δ_j` from the anharmonicities, otherwise zero.
/// The estimate is refused unless `|11⟩` mixes weakly with `|20⟩` and `|02⟩`.
///
/// Controls are I/Q drives `Ω_q·(cx_q·X_q + cy_q·Y_q)`.  With Stark coefficients `s_q` each qubit gets a third
/// channel `−s_q·Ω_q²·(cx_q² + cy_q²)·Z_q`, the drive-induced shift of its frequency by `s_q·Ω_q²·(cx_q² + cy_q²)`.
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
            h.axpy_re(-delta, &self.ops[q][2]);
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
                        h.axpy_re(2.0 * gab, &xx.add(&yy)?);
                    }
                }
            }
        }
        if self.coupling_type.contains("ZZ") {
            for a in 0..self.n {
                for b in a + 1..self.n {
                    let zeta = match (&self.zz_crosstalk, &self.anharmonicities, g) {
                        (Some(zz), _, _) => zz.get(a, b),
                        (None, Some(alpha), Some(g)) => {
                            zz_estimate(g.get(a, b), offsets[a] - offsets[b], alpha[a], alpha[b])?
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
                        coeff: -s[q],
                    });
                }
                v
            })
            .collect()
    }
}

/// The largest mixing ratio `√2·g/Δ` of `|11⟩` with `|20⟩` or `|02⟩` that [`zz_estimate`] accepts.
///
/// A small-mixing heuristic, not an accuracy bound.  The estimate's relative error grows roughly as the square of
/// that ratio: well under a percent for the couplings transmons usually have, but close to 2% at the cutoff
/// itself, where `g/2π = 21 MHz` with `α/2π = −300 MHz` at equal frequencies passes at 0.099 and estimates
/// 5.88 MHz against the three-level spectrum's 5.77 MHz.  Where the ZZ matters to better than a few percent, give
/// a calibrated `zz_crosstalk` or use the three-level model.
const MAX_MIXING: f64 = 0.1;

/// The static ZZ, `E₁₁ − E₁₀ − E₀₁ + E₀₀`, of two transmons with hopping `g`, detuning `Δ = ω_a − ω_b` and
/// anharmonicities `α_a, α_b`, all in rad/s: `ζ = 2g²·(α_a + α_b)/((Δ + α_a)·(Δ − α_b))`, the second-order repulsion
/// of `|11⟩` by `|20⟩` and `|02⟩`.  Positive near resonance for transmons' negative anharmonicities.
///
/// Refused unless the mixing ratio `√2·g/Δ` with the nearer of the two stays below [`MAX_MIXING`].
fn zz_estimate(g: f64, detuning: f64, alpha_a: f64, alpha_b: f64) -> Result<f64> {
    if g == 0.0 {
        return Ok(0.0);
    }
    let (to_20, to_02) = (detuning + alpha_a, detuning - alpha_b);
    let mixing = 2f64.sqrt() * g.abs() / to_20.abs().min(to_02.abs());
    if mixing > MAX_MIXING || mixing.is_nan() {
        return Err(Error::Config(format!(
            "the ZZ estimate from the anharmonicities is second order and needs |11> far from |20> and |02>, but \
             their mixing ratio is {mixing:.2}, above the {MAX_MIXING} this accepts; give zz_crosstalk instead"
        )));
    }
    Ok(2.0 * g * g * (alpha_a + alpha_b) / (to_20 * to_02))
}
