//! Hamiltonian models: the physical platforms.
//!
//! Every platform uses the bilinear control form `H(t) = H_drift + Σ_k u_k(t)·H_k`.  A model supplies the drift for
//! given offsets and couplings, the fixed control operators `H_k`, and - as data - how each control amplitude
//! `u_k(t)` is made from the waveform: see [`ControlChannel`].  Models never see the autodiff tape, so a new
//! platform needs no gradient code.
//!
//! A new model is one type implementing [`HamiltonianModel`] plus entries in [`model_from_config`],
//! [`model_names`](crate::config::HAMILTONIAN_TYPES) and [`default_config`].

mod duffing;
mod spin_chain;
mod superconducting;

pub use duffing::DuffingTransmon;
pub use spin_chain::{Coupling, SpinChain};
pub use superconducting::Superconducting;

use crate::config::{Config, Coverage, Optimization, Parameters, Space, Targets, WaveformMode};
use crate::error::{Error, Result};
use crate::linalg::{CMat, RMat};

/// Which part of a qubit's waveform drives a control channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The in-phase (x) quadrature.
    Cx,
    /// The quadrature (y) component.
    Cy,
    /// The instantaneous power `cx² + cy²`.
    Power,
}

/// How one control amplitude is made: `u(t) = coeff · rabi[qubit]^rabi_power · source(t)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlChannel {
    /// The qubit whose waveform and Rabi frequency drive the channel.
    pub qubit: usize,
    /// Which part of the waveform.
    pub source: Source,
    /// Power of the Rabi frequency.
    pub rabi_power: i32,
    /// Constant factor.
    pub coeff: f64,
}

impl ControlChannel {
    /// `coeff · rabi · cx` or `· cy`, the usual I/Q drive.
    pub fn drive(qubit: usize, source: Source) -> Self {
        ControlChannel {
            qubit,
            source,
            rabi_power: 1,
            coeff: 1.0,
        }
    }
}

/// A physical platform.
pub trait HamiltonianModel: Send + Sync {
    /// The JSON name.
    fn name(&self) -> &'static str;

    /// Hilbert-space dimension.
    fn dim(&self) -> usize;

    /// The drift Hamiltonian for per-qubit `offsets` and an upper-triangular `coupling` matrix, both in rad/s.
    fn drift(&self, offsets: &[f64], coupling: Option<&RMat<f64>>) -> Result<CMat<f64>>;

    /// The fixed control operators `H_k`, one per channel.
    fn control_ops(&self) -> Vec<CMat<f64>>;

    /// How each control amplitude is made, one per operator.
    fn control_channels(&self) -> Vec<ControlChannel>;

    /// A computational-basis state vector in the model's space.  The identity for two-level models.
    fn embed_state(&self, psi: &CMat<f64>) -> Result<CMat<f64>> {
        Ok(psi.clone())
    }

    /// A computational-basis density matrix, or any operator, in the model's space and zero outside the
    /// computational subspace: `P·ρ·P†`.  Also embeds Pauli operators and plot observables.  The identity for
    /// two-level models.
    fn embed_density(&self, rho: &CMat<f64>) -> Result<CMat<f64>> {
        Ok(rho.clone())
    }

    /// A computational-basis operator in the model's space: the operator on the computational subspace and the
    /// identity elsewhere.  The identity map for two-level models.
    fn embed_gate(&self, g: &CMat<f64>) -> Result<CMat<f64>> {
        Ok(g.clone())
    }
}

/// The model a configuration describes.  No `hamiltonian_type` means the Python package's original spin-chain
/// path, whose default coupling is `Z` rather than the `spin_chain` model's `XY`.
pub fn model_from_config(cfg: &Config) -> Result<Box<dyn HamiltonianModel>> {
    let n = cfg.n_qubits();
    let p = &cfg.parameters;
    let two_pi = 2.0 * std::f64::consts::PI;
    let scaled = |v: &Option<Vec<f64>>| v.as_ref().map(|v| v.iter().map(|x| two_pi * x).collect::<Vec<_>>());
    Ok(match cfg.hamiltonian_type.as_deref() {
        None => Box::new(SpinChain::new(
            n,
            Coupling::parse(p.coupling_type.as_deref().unwrap_or("Z"))?,
        )),
        Some("spin_chain") => Box::new(SpinChain::new(
            n,
            Coupling::parse(p.coupling_type.as_deref().unwrap_or("XY"))?,
        )),
        Some("superconducting") => {
            let zz = match &p.zz_crosstalk {
                Some(m) => Some(upper_coupling(m, n).map_err(Error::Config)?.map(|v| two_pi * v)),
                None => None,
            };
            Box::new(Superconducting::new(
                n,
                p.coupling_type.clone().unwrap_or_else(|| "XY".into()),
                scaled(&p.anharmonicities),
                zz,
                p.stark_shift_coeffs.clone(),
            ))
        }
        Some("duffing_transmon") => {
            let alpha = scaled(&p.anharmonicities)
                .ok_or_else(|| Error::Config("the duffing_transmon model needs anharmonicities".into()))?;
            Box::new(DuffingTransmon::new(n, alpha)?)
        }
        Some(other) => {
            return Err(Error::Config(format!(
                "unknown hamiltonian_type \"{other}\"; choose one of {}",
                model_names().join(", ")
            )));
        }
    })
}

/// Every model name, in the order the interface lists them.
pub fn model_names() -> &'static [&'static str] {
    &crate::config::HAMILTONIAN_TYPES
}

/// The couplings of `j` as an upper-triangular matrix.
///
/// Accepts an upper-triangular, lower-triangular or symmetric matrix, as the Python package does for spin chains.
/// A pair given in both triangles with different values is an error.
#[allow(clippy::needless_range_loop)] // `j[a][b]` and `j[b][a]` are read together
pub fn upper_coupling(j: &[Vec<f64>], n: usize) -> std::result::Result<RMat<f64>, String> {
    if j.len() != n || j.iter().any(|r| r.len() != n) {
        return Err(format!("J must be a {n}x{n} matrix"));
    }
    let mut out = RMat::<f64>::zeros(n, n);
    for a in 0..n {
        for b in a + 1..n {
            let (u, l) = (j[a][b], j[b][a]);
            let close = (u - l).abs() <= 1e-8 * u.abs().max(l.abs());
            let v = match (u != 0.0, l != 0.0) {
                (true, true) if !close => {
                    return Err(format!(
                        "J is asymmetric: J[{a},{b}] = {u} but J[{b},{a}] = {l}; give a symmetric matrix or one triangle"
                    ));
                }
                (true, _) => u,
                (false, _) => l,
            };
            out.set(a, b, v);
        }
    }
    Ok(out)
}

/// A complete, runnable configuration for `n_qubits` qubits of the model `name` - the Python `default_config`.
pub fn default_config(name: &str, n_qubits: usize) -> Result<Config> {
    if n_qubits == 0 {
        return Err(Error::Config("at least one qubit is needed".into()));
    }
    let n = n_qubits;
    let (offset_step, coupling, target_gate, first_states) = match name {
        "spin_chain" => (10e6, 16.67e6, "CNOT", vec!["Z", "-Z"]),
        "superconducting" | "duffing_transmon" => (10e6, 1.047e7, "iSWAP", vec!["-Z", "Z"]),
        _ => {
            return Err(Error::Config(format!(
                "unknown hamiltonian_type \"{name}\"; choose one of {}",
                model_names().join(", ")
            )));
        }
    };
    let mut j = vec![vec![0.0; n]; n];
    for (i, row) in j.iter_mut().enumerate().take(n - 1) {
        row[i + 1] = coupling;
    }
    let (initial_states, target_states) = if n == 1 {
        (vec![vec!["Z".to_string()]], Targets::Axis(vec![vec!["-Z".to_string()]]))
    } else {
        let mut init: Vec<String> = first_states.iter().map(|s| s.to_string()).collect();
        init.extend(std::iter::repeat_n("-Z".to_string(), n - 2));
        (vec![init], Targets::Gate(vec![target_gate.to_string()]))
    };
    let per = |v: f64| vec![v; n];
    Ok(Config {
        hamiltonian_type: Some(name.to_string()),
        qubits: (1..=n).map(|i| format!("q{i}")).collect(),
        compute_resource: Some("cpu".into()),
        cpu_cores: None,
        seed: None,
        parameters: Parameters {
            delta: Some((1..=n).map(|i| i as f64 * offset_step).collect()),
            sigma_delta: Some(per(0.0)),
            omega_r_max: per(40e6),
            sigma_omega_r_max: per(0.0),
            pulse_duration: per(200e-9),
            point_in_pulse: vec![100; n],
            wf_type: vec!["cheb".into(); n],
            wf_mode: vec![WaveformMode::Cart; n],
            amplitude_envelope: vec!["gn".into(); n],
            amplitude_order: vec![1; n],
            coverage: vec![Coverage::Broadband; n],
            sw: per(5e6),
            pulse_offset: per(0.0),
            pulse_bandwidth: per(5e5),
            ratio_factor: per(0.5),
            profile_order: vec![2; n],
            n_para: vec![16; n],
            j: Some(j),
            coupling_type: Some("XY".into()),
            sigma_j: Some(0.0),
            t1: None,
            t2: None,
            anharmonicities: (name == "duffing_transmon").then(|| per(-330e6)),
            zz_crosstalk: None,
            stark_shift_coeffs: None,
        },
        initial_states,
        target_states,
        optimization: Optimization {
            space: Space::Hilbert,
            dissipation_mode: None,
            h0_snapshots: 1,
            omega_r_snapshots: 1,
            algorithm: "l-bfgs".into(),
            max_iter: 300,
            targ_fid: 0.999,
        },
    })
}

/// `op` acting on site `site` of `n` sites of local dimension `op.rows`, identity elsewhere.
pub(crate) fn embed_local(op: &CMat<f64>, site: usize, n: usize) -> CMat<f64> {
    crate::setup::embed(op, site, n)
}

#[cfg(test)]
mod tests;
