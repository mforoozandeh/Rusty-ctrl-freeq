//! From a [`Config`] to the numbers the optimiser works with - the Python `Initialise`.
//!
//! [`Problem::build`] validates the configuration, draws every random quantity from one seeded generator, and
//! assembles the batch: one element per (initial state, drift snapshot, Rabi snapshot), each with its drift
//! Hamiltonian, Rabi frequencies, initial state and target.  One gate for every initial state is scored as a gate
//! instead; [`BatchElement`] says how.

mod dissipation;
mod sampling;
mod states;

pub use dissipation::collapse_operators;
pub use states::{
    canonical_gate, embed, gate, gate_names, pauli_ops, paulis, product_density, product_state, rotation, spin_ops,
};

use rand::SeedableRng;

use crate::autodiff::{C, LindbladOps};
use crate::basis::{QubitBasis, basis, envelope, initial_params, linspace, qubit_basis, raw_coefficients};
use crate::config::{Config, Coverage, Space, Targets};
use crate::error::{Error, Result};
use crate::hamiltonian::{ControlChannel, HamiltonianModel, model_from_config, upper_coupling};
use crate::linalg::{CMat, RMat};
use sampling::{QubitOffsets, coupling_instances, excitation_profile, rabi_instances, sample_offsets};

/// The random number generator every random choice in a run is drawn from.  Seeded explicitly, so runs are
/// reproducible and nothing needs an operating-system entropy source.
pub type Rng = rand_chacha::ChaCha8Rng;

/// The seed a run of `cfg` uses: the configured one, or one drawn from the clock.
///
/// The clock reading is mixed with SplitMix64, so seeds drawn a moment apart - or from a browser clock with
/// millisecond resolution - still differ in every bit.
pub fn resolve_seed(cfg: &Config) -> u64 {
    cfg.seed.unwrap_or_else(|| {
        let now = crate::time::SystemTime::now()
            .duration_since(crate::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut z = (now as u64 ^ (now >> 64) as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

/// How states evolve over one time step.
#[derive(Clone, Debug, PartialEq)]
pub enum Evolution {
    /// `ψ ← U·ψ`.
    Hilbert,
    /// `ρ ← U·ρ·U†`.
    Liouville,
    /// `ρ ← U·ρ·U†`, then the Lindblad dissipator's exact channel over the step.
    Lindblad(LindbladOps),
}

/// One batch element.
///
/// For state targets it evolves one initial state and scores it against its target.  For one gate
/// ([`Targets::single_gate`]) it scores the average gate fidelity instead: in Hilbert space one element evolves the
/// whole computational basis as the columns of `initial`; in Liouville space the elements evolve the Pauli operators,
/// each scored against the gate's image of it, weighted so the batch mean is Nielsen's average gate fidelity.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchElement {
    /// Drift Hamiltonian in rad/s.
    pub h0: CMat<f64>,
    /// Rabi frequency per qubit in rad/s.
    pub rabi: Vec<f64>,
    /// What is evolved: state vectors as columns (Hilbert) or a density matrix or Pauli operator (Liouville).
    pub initial: CMat<f64>,
    /// What `initial` is scored against after the pulse, of the same shape.
    pub target: CMat<f64>,
    /// Which initial state, or which Pauli operator for a gate in Liouville space (0 in Hilbert space).
    pub initial_index: usize,
    /// Which drift snapshot.
    pub snapshot: usize,
    /// Which Rabi snapshot.
    pub rabi_index: usize,
}

/// Everything an evaluation of the objective needs, derived from a [`Config`].
pub struct Problem {
    /// Number of qubits.
    pub n_qubits: usize,
    /// Hilbert-space dimension of the model.
    pub dim: usize,
    /// Number of time steps.
    pub n_pulse: usize,
    /// Time step in seconds: duration / steps.
    pub dt: f64,
    /// Pulse duration in seconds.
    pub duration: f64,
    /// Waveform sample times in seconds: the middle of each step, `(t + ½)·dt`.
    pub times: Vec<f64>,
    /// Each qubit's basis matrices.
    pub qubits: Vec<QubitBasis>,
    /// `exp(i·offset_q·(t − T/2))`, `n_pulse × n_qubits`.
    pub modulation: CMat<f64>,
    /// Control operators `H_k`.
    pub control_ops: Vec<CMat<f64>>,
    /// How each control amplitude is made.
    pub channels: Vec<ControlChannel>,
    /// The batch, ordered by initial state (or Pauli operator), then drift snapshot, then Rabi snapshot.
    pub batch: Vec<BatchElement>,
    /// State evolution per step.
    pub evolution: Evolution,
    /// Effective state space.
    pub space: Space,
    /// Initial optimiser parameters.
    pub x0: Vec<f64>,
    /// The seed every random quantity was drawn from.
    pub seed: u64,
    /// Target fidelity.
    pub target_fidelity: f64,
    /// Iteration limit.
    pub max_iter: usize,
    /// Optimiser name.
    pub algorithm: String,
    /// Initial states in the model's space, one per configured initial state; the plots start from these.
    pub initial_states: Vec<CMat<f64>>,
    /// Per qubit, the Pauli observables `[σx, σy, σz]` in the model's space, zero outside the computational subspace.
    pub observables: Vec<[CMat<f64>; 3]>,
    /// The Hamiltonian model.
    pub model: Box<dyn HamiltonianModel>,
    /// Maximum Rabi frequency per qubit in rad/s.
    pub rabi_max: Vec<f64>,
    /// Offset per qubit in rad/s.
    pub delta: Vec<f64>,
    /// Sweep width per qubit in rad/s.
    pub sw: Vec<f64>,
    /// Configured coupling (upper-triangular, rad/s, without noise); `None` for one qubit.
    pub coupling: Option<RMat<f64>>,
    /// Number of drift snapshots.
    pub n_snapshots: usize,
    /// Number of Rabi snapshots.
    pub n_rabi: usize,
}

impl Problem {
    /// Number of optimiser parameters.
    pub fn n_params(&self) -> usize {
        self.qubits.iter().map(|q| q.n_params).sum()
    }

    /// Build the problem `cfg` describes, with its own seed or one from the clock.
    pub fn build(cfg: &Config) -> Result<Problem> {
        Problem::build_with_seed(cfg, resolve_seed(cfg))
    }

    /// Build the problem `cfg` describes, drawing every random quantity from `seed`.
    pub fn build_with_seed(cfg: &Config, seed: u64) -> Result<Problem> {
        cfg.check()?;
        let mut rng = Rng::seed_from_u64(seed);
        let n = cfg.n_qubits();
        let p = &cfg.parameters;
        let two_pi = 2.0 * std::f64::consts::PI;
        let per_qubit = |v: &Option<Vec<f64>>| -> Vec<f64> {
            v.as_ref()
                .map_or(vec![0.0; n], |v| v.iter().map(|x| two_pi * x).collect())
        };
        let scaled = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| two_pi * x).collect() };

        // Time grid.
        let duration = p.pulse_duration[0];
        let n_pulse = p.point_in_pulse[0];
        let dt = duration / n_pulse as f64;
        let times: Vec<f64> = (0..n_pulse).map(|t| (t as f64 + 0.5) * dt).collect();

        // Initial coefficients (all qubits first, as Python draws them), then the bases.
        let bases = p.wf_type.iter().map(|name| basis(name)).collect::<Result<Vec<_>>>()?;
        let raw: Vec<Vec<f64>> = p.n_para.iter().map(|&k| raw_coefficients(k, &mut rng)).collect();
        let x_grid = linspace(-1.0, 1.0, n_pulse);
        let mut qubits = Vec::with_capacity(n);
        let mut x0 = Vec::new();
        for q in 0..n {
            let env = envelope(&p.amplitude_envelope[q], &x_grid, p.amplitude_order[q])?;
            let qb = qubit_basis(bases[q].as_ref(), &env, p.n_para[q], p.wf_mode[q], n_pulse, &mut rng)?;
            x0.extend(initial_params(bases[q].as_ref(), &raw[q], p.wf_mode[q]));
            qubits.push(qb);
        }

        // Rabi frequencies.
        let rabi_max = scaled(&p.omega_r_max);
        let rabi = rabi_instances(
            &rabi_max,
            &scaled(&p.sigma_omega_r_max),
            cfg.optimization.omega_r_snapshots,
            &mut rng,
        );

        // Offsets and excitation profiles.
        let delta = per_qubit(&p.delta);
        let sigma_delta = per_qubit(&p.sigma_delta);
        let sw = scaled(&p.sw);
        let specs: Vec<QubitOffsets> = (0..n)
            .map(|q| QubitOffsets {
                coverage: p.coverage[q],
                delta: delta[q],
                sigma: sigma_delta[q],
                sw: sw[q],
                bandwidth: two_pi * p.pulse_bandwidth[q],
                ratio: p.ratio_factor[q],
                profile_order: p.profile_order[q],
            })
            .collect();
        let coupling = match (&p.j, n > 1) {
            (Some(j), true) => Some(upper_coupling(j, n).map_err(Error::Config)?.map(|v| two_pi * v)),
            _ => None,
        };
        let sigma_j = two_pi * p.sigma_j.unwrap_or(0.0);
        // One snapshot suffices only when nothing in the drift is uncertain.
        let fixed = p.coverage.iter().all(|&c| c == Coverage::Single)
            && sigma_delta.iter().all(|&s| s == 0.0)
            && (coupling.is_none() || sigma_j == 0.0);
        let per_qubit_offsets: Vec<Vec<f64>> = if fixed {
            delta.iter().map(|&d| vec![d]).collect()
        } else {
            specs
                .iter()
                .map(|s| sample_offsets(s, cfg.optimization.h0_snapshots, &mut rng))
                .collect()
        };
        let m = per_qubit_offsets[0].len();
        let profiles: Vec<Vec<f64>> = specs
            .iter()
            .zip(&per_qubit_offsets)
            .map(|(s, o)| excitation_profile(s, o))
            .collect();

        // Couplings.
        let couplings = coupling.as_ref().map(|j| coupling_instances(j, sigma_j, m, &mut rng));

        // The model and the drifts.
        let model = model_from_config(cfg)?;
        let drifts = (0..m)
            .map(|k| {
                let offsets: Vec<f64> = per_qubit_offsets.iter().map(|o| o[k]).collect();
                model.drift(&offsets, couplings.as_ref().map(|c| &c[k]))
            })
            .collect::<Result<Vec<_>>>()?;

        // States and targets, computed in the qubits' space and then embedded.
        let space = cfg.space();
        let raw_state = |axes: &[String]| -> Result<CMat<f64>> {
            match space {
                Space::Hilbert => product_state(axes),
                Space::Liouville => product_density(axes),
            }
        };
        let embed = |s: &CMat<f64>| -> Result<CMat<f64>> {
            match space {
                Space::Hilbert => model.embed_state(s),
                Space::Liouville => model.embed_density(s),
            }
        };
        let apply = |u: &CMat<f64>, s: &CMat<f64>| -> Result<CMat<f64>> {
            match space {
                Space::Hilbert => u.matmul(s),
                Space::Liouville => u.matmul(s)?.matmul(&u.adjoint()),
            }
        };
        let raw_initials: Vec<CMat<f64>> = cfg.initial_states.iter().map(|a| raw_state(a)).collect::<Result<_>>()?;
        let initial_states: Vec<CMat<f64>> = raw_initials.iter().map(&embed).collect::<Result<_>>()?;
        // Where a target applies: a qubit is in its band where its excitation profile is 1.
        let in_band = |q: usize, k: usize| profiles[q][k] == 1.0;
        let everyone_in_band = |k: usize| (0..n).all(|q| in_band(q, k));

        // Per initial state, what it is scored against at each snapshot: its target state, or for one gate the
        // gate's action on the whole computational basis (see `gate_rows`).
        let rows: Vec<Row> = match cfg.target_states.single_gate() {
            Some(name) => {
                let v = gate(name, n)?;
                let id = CMat::<f64>::identity(1 << n);
                let gates: Vec<&CMat<f64>> = (0..m).map(|k| if everyone_in_band(k) { &v } else { &id }).collect();
                gate_rows(&gates, space, model.as_ref())?
            }
            None => {
                let deg = std::f64::consts::PI / 180.0;
                let target = |pi: usize, init: &CMat<f64>, k: usize| -> Result<CMat<f64>> {
                    let t = match &cfg.target_states {
                        // A product state: each qubit's target axis in its band, its initial axis elsewhere.
                        Targets::Axis(axes) => {
                            let per_qubit: Vec<String> = (0..n)
                                .map(|q| {
                                    if in_band(q, k) {
                                        &axes[pi][q]
                                    } else {
                                        &cfg.initial_states[pi][q]
                                    }
                                })
                                .cloned()
                                .collect();
                            raw_state(&per_qubit)?
                        }
                        Targets::Gate(names) if everyone_in_band(k) => apply(&gate(&names[pi], n)?, init)?,
                        Targets::Gate(_) => init.clone(),
                        Targets::PhiBeta { phi, beta } => {
                            let angles: Vec<f64> = (0..n).map(|q| beta[pi][q] * deg * profiles[q][k]).collect();
                            apply(&rotation(&phi[pi], &angles, n)?, init)?
                        }
                    };
                    let t = embed(&t)?;
                    if space == Space::Liouville && (t.matmul(&t)?.trace().re - 1.0).abs() > 1e-9 {
                        return Err(Error::NotSupported(
                            "Liouville-space fidelity needs pure targets; this target is mixed".into(),
                        ));
                    }
                    Ok(t)
                };
                raw_initials
                    .iter()
                    .enumerate()
                    .map(|(pi, init)| {
                        Ok(Row {
                            evolved: embed(init)?,
                            targets: (0..m).map(|k| target(pi, init, k)).collect::<Result<_>>()?,
                        })
                    })
                    .collect::<Result<_>>()?
            }
        };

        // The batch: initial state (or Pauli operator), then drift snapshot, then Rabi snapshot.
        let mut batch = Vec::with_capacity(rows.len() * m * rabi.len());
        for (pi, row) in rows.into_iter().enumerate() {
            for (k, h0) in drifts.iter().enumerate() {
                for (r, rb) in rabi.iter().enumerate() {
                    batch.push(BatchElement {
                        h0: h0.clone(),
                        rabi: rb.clone(),
                        initial: row.evolved.clone(),
                        target: row.targets[k].clone(),
                        initial_index: pi,
                        snapshot: k,
                        rabi_index: r,
                    });
                }
            }
        }

        let evolution = if cfg.is_dissipative() {
            let (t1, t2) = (p.t1.as_deref().unwrap_or(&[]), p.t2.as_deref().unwrap_or(&[]));
            Evolution::Lindblad(LindbladOps::new(&collapse_operators(t1, t2)?, dt)?)
        } else if space == Space::Liouville {
            Evolution::Liouville
        } else {
            Evolution::Hilbert
        };

        let offsets = scaled(&p.pulse_offset);
        let modulation = CMat::from_fn(n_pulse, n, |t, q| {
            C::from_polar(1.0, offsets[q] * (times[t] - duration / 2.0))
        });
        // Projected, zero outside the computational subspace, so leaked population reads as zero on every axis.
        let observables = pauli_ops(n)
            .iter()
            .map(|o| -> Result<[CMat<f64>; 3]> {
                Ok([
                    model.embed_density(&o[0])?,
                    model.embed_density(&o[1])?,
                    model.embed_density(&o[2])?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Problem {
            n_qubits: n,
            dim: model.dim(),
            n_pulse,
            dt,
            duration,
            times,
            qubits,
            modulation,
            control_ops: model.control_ops(),
            channels: model.control_channels(),
            batch,
            evolution,
            space,
            x0,
            seed,
            target_fidelity: cfg.optimization.targ_fid,
            max_iter: cfg.optimization.max_iter,
            algorithm: cfg.optimization.algorithm.clone(),
            initial_states,
            observables,
            model,
            rabi_max,
            delta,
            sw,
            coupling,
            n_snapshots: m,
            n_rabi: rabi.len(),
        })
    }
}

/// One row of the batch before the drift and Rabi snapshots multiply it.
struct Row {
    /// What is evolved.
    evolved: CMat<f64>,
    /// Per drift snapshot, what it is scored against.
    targets: Vec<CMat<f64>>,
}

/// The rows that score the average gate fidelity to `gates[k]` at drift snapshot `k`.
///
/// Hilbert space: one row evolving the embedded computational basis `P`, scored against `P·W` by
/// `(‖M‖² + |Tr M|²)/(d(d+1))` with `M = (P·W)†·U·P`, which counts leakage (Pedersen et al. 2007).
///
/// Liouville space: a row per Pauli string `P_j` (the identity first), scored by `Re Tr(σ_j·Λ(P_j))` against the
/// scaled image `σ_j = c_j·W·P_j·W†`.  Nielsen's formula, generalised to maps that lose population,
/// `F = (d·x₀ + Σ_j x_j)/(d²·(d + 1))` with `x_j = Re Tr(W·P_j·W†·Λ(P_j))`, is then the mean over the `d²` rows with
/// `c₀ = 1` and `c_j = 1/(d + 1)` otherwise.
fn gate_rows(gates: &[&CMat<f64>], space: Space, model: &dyn HamiltonianModel) -> Result<Vec<Row>> {
    let d = gates.first().map_or(1, |g| g.rows);
    match space {
        Space::Hilbert => Ok(vec![Row {
            evolved: model.embed_state(&CMat::identity(d))?,
            targets: gates.iter().map(|w| model.embed_state(w)).collect::<Result<_>>()?,
        }]),
        Space::Liouville => pauli_strings(d.trailing_zeros() as usize)
            .iter()
            .enumerate()
            .map(|(j, u)| {
                let weight = if j == 0 { 1.0 } else { 1.0 / (d + 1) as f64 };
                let targets = gates
                    .iter()
                    .map(|w| {
                        Ok(model
                            .embed_density(&w.matmul(u)?.matmul(&w.adjoint())?)?
                            .scale_re(weight))
                    })
                    .collect::<Result<_>>()?;
                Ok(Row {
                    evolved: model.embed_density(u)?,
                    targets,
                })
            })
            .collect(),
    }
}

/// The `4ⁿ` Pauli strings on `n` qubits, the identity first: an orthogonal basis, `Tr(P_i·P_j) = 2ⁿ·δ_ij`.
fn pauli_strings(n: usize) -> Vec<CMat<f64>> {
    let [x, y, z] = paulis();
    let single = [CMat::identity(2), x, y, z];
    (0..1usize << (2 * n))
        .map(|i| {
            (0..n)
                .rev()
                .fold(CMat::identity(1), |acc, q| acc.kron(&single[(i >> (2 * q)) & 3]))
        })
        .collect()
}

#[cfg(test)]
mod tests;
