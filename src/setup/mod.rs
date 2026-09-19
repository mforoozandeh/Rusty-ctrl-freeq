//! From a [`Config`] to the numbers the optimiser works with - the Python `Initialise`.
//!
//! [`Problem::build`] validates the configuration, draws every random quantity from one seeded generator, and
//! assembles the batch: one element per (initial state, drift snapshot, Rabi snapshot), each with its drift
//! Hamiltonian, Rabi frequencies, initial state and target.

mod dissipation;
mod sampling;
mod states;

pub use dissipation::collapse_operators;
pub use states::{embed, gate, gate_names, pauli_ops, paulis, product_density, product_state, rotation, spin_ops};

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
pub fn resolve_seed(cfg: &Config) -> u64 {
    cfg.seed.unwrap_or_else(|| {
        let now = crate::time::SystemTime::now()
            .duration_since(crate::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        (now as u64) ^ ((now >> 64) as u64)
    })
}

/// How states evolve over one time step.
#[derive(Clone, Debug, PartialEq)]
pub enum Evolution {
    /// `ψ ← U·ψ`.
    Hilbert,
    /// `ρ ← U·ρ·U†`.
    Liouville,
    /// `ρ ← U·ρ·U†`, then an explicit Euler step of the Lindblad dissipator.
    Lindblad(LindbladOps),
}

/// One batch element.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchElement {
    /// Drift Hamiltonian in rad/s.
    pub h0: CMat<f64>,
    /// Rabi frequency per qubit in rad/s.
    pub rabi: Vec<f64>,
    /// Initial state: a column vector (Hilbert) or a density matrix (Liouville).
    pub initial: CMat<f64>,
    /// Target, of the same kind as `initial`.
    pub target: CMat<f64>,
    /// Which initial state.
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
    /// Time grid in seconds, `linspace(ε, T, n_pulse)` as in Python.
    pub times: Vec<f64>,
    /// Each qubit's basis matrices.
    pub qubits: Vec<QubitBasis>,
    /// `exp(i·offset_q·(t − T/2))`, `n_pulse × n_qubits`.
    pub modulation: CMat<f64>,
    /// Control operators `H_k`.
    pub control_ops: Vec<CMat<f64>>,
    /// How each control amplitude is made.
    pub channels: Vec<ControlChannel>,
    /// The batch, ordered by initial state, then drift snapshot, then Rabi snapshot.
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
    /// Initial states in the model's space, one per configured initial state.
    pub initial_states: Vec<CMat<f64>>,
    /// Per qubit, the Pauli observables `[σx, σy, σz]` in the model's space.
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
        let times = linspace(f64::EPSILON, duration, n_pulse);

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
        let fixed = p.coverage.iter().all(|&c| c == Coverage::Single) && sigma_delta.iter().all(|&s| s == 0.0);
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
        let coupling = match (&p.j, n > 1) {
            (Some(j), true) => Some(upper_coupling(j, n).map_err(Error::Config)?.map(|v| two_pi * v)),
            _ => None,
        };
        let sigma_j = two_pi * p.sigma_j.unwrap_or(0.0);
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
        let deg = std::f64::consts::PI / 180.0;
        // targets[p][k]: what initial state p should become at snapshot k.
        let mut targets: Vec<Vec<CMat<f64>>> = Vec::with_capacity(raw_initials.len());
        for (pi, init) in raw_initials.iter().enumerate() {
            let mut row = Vec::with_capacity(m);
            for (k, &first) in profiles[0].iter().enumerate() {
                let applies = first == 1.0;
                let t = match &cfg.target_states {
                    Targets::Axis(axes) if applies => raw_state(&axes[pi])?,
                    Targets::Gate(names) if applies => apply(&gate(&names[pi], n)?, init)?,
                    Targets::PhiBeta { phi, beta } => {
                        let angles: Vec<f64> = (0..n).map(|q| beta[pi][q] * deg * profiles[q][k]).collect();
                        apply(&rotation(&phi[pi], &angles, n)?, init)?
                    }
                    _ => init.clone(),
                };
                row.push(embed(&t)?);
            }
            targets.push(row);
        }
        if space == Space::Liouville {
            for t in targets.iter().flatten() {
                let purity = t.matmul(t)?.trace().re;
                if (purity - 1.0).abs() > 1e-9 {
                    return Err(Error::NotSupported(
                        "Liouville-space fidelity needs pure targets; this target is mixed".into(),
                    ));
                }
            }
        }

        // The batch: initial state, then drift snapshot, then Rabi snapshot.
        let mut batch = Vec::with_capacity(initial_states.len() * m * rabi.len());
        for (pi, init) in initial_states.iter().enumerate() {
            for (k, h0) in drifts.iter().enumerate() {
                for (r, rb) in rabi.iter().enumerate() {
                    batch.push(BatchElement {
                        h0: h0.clone(),
                        rabi: rb.clone(),
                        initial: init.clone(),
                        target: targets[pi][k].clone(),
                        initial_index: pi,
                        snapshot: k,
                        rabi_index: r,
                    });
                }
            }
        }

        let evolution = if cfg.is_dissipative() {
            let (t1, t2) = (p.t1.as_deref().unwrap_or(&[]), p.t2.as_deref().unwrap_or(&[]));
            Evolution::Lindblad(LindbladOps::new(&collapse_operators(t1, t2)?)?)
        } else if space == Space::Liouville {
            Evolution::Liouville
        } else {
            Evolution::Hilbert
        };

        let offsets = scaled(&p.pulse_offset);
        let modulation = CMat::from_fn(n_pulse, n, |t, q| {
            C::from_polar(1.0, offsets[q] * (times[t] - duration / 2.0))
        });
        let observables = pauli_ops(n)
            .iter()
            .map(|o| -> Result<[CMat<f64>; 3]> {
                Ok([
                    model.embed_gate(&o[0])?,
                    model.embed_gate(&o[1])?,
                    model.embed_gate(&o[2])?,
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

#[cfg(test)]
mod tests;
