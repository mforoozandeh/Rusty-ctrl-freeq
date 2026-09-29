//! What the plots need after a run: the pulses, the state dynamics, the excitation profile, and - with a Rabi
//! spread - the map of the profile over Rabi frequency too.
//!
//! Everything here is plain `f64` propagation of the optimised pulse, with the same Hamiltonians, time step and
//! evolution as the objective.

use serde::{Deserialize, Serialize};

use crate::autodiff::{C, Step};
use crate::config::{Config, Space};
use crate::error::Result;
use crate::hamiltonian::Source;
use crate::linalg::{CMat, expm_mi_dt};
use crate::objective::{CostModel, Waveforms};
use crate::run::RunResult;
use crate::setup::{Evolution, Problem};

/// Batch snapshots whose trajectories are kept for plotting.
const MAX_SNAPSHOTS: usize = 20;
/// Largest dimension whose full density matrix is kept; larger ones keep the populations only.
const MAX_FULL_DENSITY: usize = 4;
/// Offsets in the excitation profile.
#[cfg(not(target_arch = "wasm32"))]
const PROFILE_POINTS: usize = 1000;
#[cfg(target_arch = "wasm32")]
const PROFILE_POINTS: usize = 400;
/// Offsets, and Rabi frequencies, in the Rabi map.
#[cfg(not(target_arch = "wasm32"))]
const MAP_POINTS: usize = 101;
#[cfg(target_arch = "wasm32")]
const MAP_POINTS: usize = 51;

/// Plot data for a finished run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Analysis {
    /// Pulse sample times in nanoseconds: the middle of each time step.
    pub times_ns: Vec<f64>,
    /// State times in nanoseconds: the step boundaries from 0 to the pulse duration, one more than the samples.
    pub state_times_ns: Vec<f64>,
    /// One trace per qubit.
    pub pulses: Vec<PulseTrace>,
    /// One per initial state.
    pub dynamics: Vec<Dynamics>,
    /// One per initial state.
    pub profiles: Vec<ExcitationProfile>,
    /// One per initial state when the run drew more than one set of Rabi frequencies; otherwise none.
    pub rabi_maps: Vec<RabiMap>,
}

/// A qubit's modulated waveform, as a fraction of its maximum Rabi frequency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PulseTrace {
    /// Qubit index.
    pub qubit: usize,
    /// In-phase quadrature.
    pub cx: Vec<f64>,
    /// Quadrature.
    pub cy: Vec<f64>,
    /// Amplitude.
    pub amp: Vec<f64>,
    /// Phase in radians.
    pub phase: Vec<f64>,
}

/// The evolution of one initial state under the pulse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dynamics {
    /// Which initial state.
    pub initial_state: usize,
    /// A label per component: `ψ1…` for state vectors, `ρ11…` for density matrices.
    pub labels: Vec<String>,
    /// `[component][t]` as `(re, im)` at the state times, under the mean drift (configured offsets and couplings) and maximum Rabi
    /// frequencies.
    pub mean: Vec<Vec<[f64; 2]>>,
    /// `[snapshot][component][t]` for up to twenty batch snapshots.
    pub snapshots: Vec<Vec<Vec<[f64; 2]>>>,
    /// Per qubit, `⟨σx⟩, ⟨σy⟩, ⟨σz⟩` over time under the mean drift.
    pub observables: Vec<[Vec<f64>; 3]>,
    /// Per qubit, the smallest value of each observable over the batch snapshots, over time.
    pub observables_min: Vec<[Vec<f64>; 3]>,
    /// Per qubit, the largest value of each observable over the batch snapshots, over time.
    pub observables_max: Vec<[Vec<f64>; 3]>,
    /// Population outside the computational subspace over time, under the mean drift: `1 − Tr(P·ρ)`.  Zero
    /// throughout for models without room to leak.  Total, not a sum over qubits, which would count leakage from
    /// two qubits at once twice.
    pub leakage: Vec<f64>,
    /// The smallest leakage over the batch snapshots, over time.
    pub leakage_min: Vec<f64>,
    /// The largest leakage over the batch snapshots, over time.
    pub leakage_max: Vec<f64>,
}

/// Where the pulse acts: the final state's observables as the offsets are swept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcitationProfile {
    /// Which initial state.
    pub initial_state: usize,
    /// Per qubit, the offsets swept, in Hz, across 1.5 sweep widths about the configured offset.
    pub offsets_hz: Vec<Vec<f64>>,
    /// Per qubit, `⟨σx⟩, ⟨σy⟩, ⟨σz⟩` at the end of the pulse at each offset.
    pub xyz: Vec<[Vec<f64>; 3]>,
}

/// The excitation profile swept through the Rabi frequencies as well as the offsets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RabiMap {
    /// Which initial state.
    pub initial_state: usize,
    /// Per qubit, the offsets swept, in Hz, across 1.5 sweep widths about the configured offset.
    pub offsets_hz: Vec<Vec<f64>>,
    /// The Rabi frequencies swept, as fractions of the maximum.  Every qubit with a Rabi spread is scaled
    /// together; qubits without one keep their maximum, as they did throughout the optimisation.  From half to one
    /// and a half, or four times the widest relative spread either side of one if that reaches further, never
    /// below zero.
    pub scales: Vec<f64>,
    /// Per qubit, `⟨σx⟩, ⟨σy⟩, ⟨σz⟩` at the end of the pulse, row by row: at `scales[r]` and offset `c`, entry
    /// `r · offsets + c`.
    pub xyz: Vec<[Vec<f64>; 3]>,
    /// Per qubit, the smallest and largest Rabi frequency drawn for the optimisation, as fractions of the maximum.
    pub drawn: Vec<[f64; 2]>,
}

/// Plot data for `result`, a run of `cfg`.
pub fn analyse(cfg: &Config, result: &RunResult) -> Result<Analysis> {
    let problem = Problem::build_with_seed(cfg, result.seed)?;
    let model = CostModel::new(problem);
    let w = model.waveforms(&result.solution)?;
    let p = model.problem();

    let pulses = (0..p.n_qubits)
        .map(|q| PulseTrace {
            qubit: q,
            cx: w.cx.col(q),
            cy: w.cy.col(q),
            amp: w.amp.col(q),
            phase: w.phase.col(q),
        })
        .collect();

    let mean_drift = p.model.drift(&p.delta, p.coupling.as_ref())?;
    let dynamics = (0..p.initial_states.len())
        .map(|i| dynamics(p, &w, &mean_drift, i))
        .collect::<Result<Vec<_>>>()?;
    let profiles = (0..p.initial_states.len())
        .map(|i| profile(p, &w, i))
        .collect::<Result<Vec<_>>>()?;
    let rabi_maps = if p.n_rabi > 1 {
        rabi_maps(cfg, p, &w)?
    } else {
        Vec::new()
    };

    Ok(Analysis {
        times_ns: p.times.iter().map(|t| t * 1e9).collect(),
        state_times_ns: (0..=p.n_pulse).map(|t| t as f64 * p.dt * 1e9).collect(),
        pulses,
        dynamics,
        profiles,
        rabi_maps,
    })
}

/// The control amplitudes at every step for Rabi frequencies `rabi`: `u[t][k]`.
fn controls(p: &Problem, w: &Waveforms, rabi: &[f64]) -> Vec<Vec<f64>> {
    (0..p.n_pulse)
        .map(|t| {
            p.channels
                .iter()
                .map(|ch| {
                    let (x, y) = (w.cx.get(t, ch.qubit), w.cy.get(t, ch.qubit));
                    let s = match ch.source {
                        Source::Cx => x,
                        Source::Cy => y,
                        Source::Power => x * x + y * y,
                    };
                    ch.coeff * rabi[ch.qubit].powi(ch.rabi_power) * s
                })
                .collect()
        })
        .collect()
}

/// `state` after one step under drift `h0` and controls `ut`.
fn step(p: &Problem, h0: &CMat<f64>, ut: &[f64], state: &CMat<f64>) -> Result<CMat<f64>> {
    let mut h = h0.clone();
    for (k, op) in p.control_ops.iter().enumerate() {
        h.axpy_re(ut[k], op);
    }
    let u = expm_mi_dt(&h, p.dt)?;
    Ok(match &p.evolution {
        Evolution::Hilbert => u.matmul(state)?,
        Evolution::Liouville => u.matmul(state)?.matmul(&u.adjoint())?,
        // Strang splitting, as the objective propagates: half a dissipation step at each end of the step, which is
        // the same sequence once the halves between steps meet.
        Evolution::Lindblad(ops) => {
            let entering = ops.apply(state, Step::Half, false)?;
            let turned = u.matmul(&entering)?.matmul(&u.adjoint())?;
            ops.apply(&turned, Step::Half, false)?
        }
    })
}

/// `initial` and the state after every step under drift `h0` and controls `u`.
fn trajectory(p: &Problem, h0: &CMat<f64>, u: &[Vec<f64>], initial: &CMat<f64>) -> Result<Vec<CMat<f64>>> {
    let mut out = Vec::with_capacity(p.n_pulse + 1);
    out.push(initial.clone());
    for ut in u {
        let next = step(p, h0, ut, out.last().expect("starts with the initial state"))?;
        out.push(next);
    }
    Ok(out)
}

/// The state at the end of the pulse under drift `h0` and controls `u`.
fn final_state(p: &Problem, h0: &CMat<f64>, u: &[Vec<f64>], initial: &CMat<f64>) -> Result<CMat<f64>> {
    u.iter().try_fold(initial.clone(), |state, ut| step(p, h0, ut, &state))
}

/// `Re⟨ψ|O|ψ⟩` or `Re Tr(O·ρ)`.
fn expectation(o: &CMat<f64>, state: &CMat<f64>) -> f64 {
    if state.cols == 1 {
        let o_psi = o.matmul(state).expect("shapes match");
        state.inner(&o_psi).re
    } else {
        o.matmul(state).expect("shapes match").trace().re
    }
}

/// Which components of a state are plotted, and their labels.
fn components(p: &Problem) -> (Vec<(usize, usize)>, Vec<String>) {
    let d = p.dim;
    match p.space {
        Space::Hilbert => (
            (0..d).map(|i| (i, 0)).collect(),
            (1..=d).map(|i| format!("ψ{i}")).collect(),
        ),
        Space::Liouville if d <= MAX_FULL_DENSITY => {
            let idx: Vec<(usize, usize)> = (0..d).flat_map(|r| (0..d).map(move |c| (r, c))).collect();
            let labels = idx.iter().map(|(r, c)| format!("ρ{}{}", r + 1, c + 1)).collect();
            (idx, labels)
        }
        Space::Liouville => (
            (0..d).map(|i| (i, i)).collect(),
            (1..=d).map(|i| format!("ρ{i}{i}")).collect(),
        ),
    }
}

fn pick(traj: &[CMat<f64>], idx: &[(usize, usize)]) -> Vec<Vec<[f64; 2]>> {
    idx.iter()
        .map(|&(r, c)| {
            traj.iter()
                .map(|s| {
                    let z: C<f64> = s.get(r, c);
                    [z.re, z.im]
                })
                .collect()
        })
        .collect()
}

/// Population outside the computational subspace at each step: `1 − Tr(P·ρ)`, or `1 − ‖P·ψ‖²`.
fn leakage_over(projector: &CMat<f64>, traj: &[CMat<f64>]) -> Vec<f64> {
    traj.iter().map(|s| 1.0 - expectation(projector, s)).collect()
}

/// Per qubit, `⟨σx⟩, ⟨σy⟩, ⟨σz⟩` in `state`.
fn bloch(p: &Problem, state: &CMat<f64>) -> Vec<[f64; 3]> {
    p.observables
        .iter()
        .map(|ops| [0, 1, 2].map(|k| expectation(&ops[k], state)))
        .collect()
}

fn observables_over(p: &Problem, traj: &[CMat<f64>]) -> Vec<[Vec<f64>; 3]> {
    p.observables
        .iter()
        .map(|ops| [0, 1, 2].map(|k| traj.iter().map(|s| expectation(&ops[k], s)).collect()))
        .collect()
}

fn dynamics(p: &Problem, w: &Waveforms, mean_drift: &CMat<f64>, i: usize) -> Result<Dynamics> {
    let (idx, labels) = components(p);
    let initial = &p.initial_states[i];
    let mean_traj = trajectory(p, mean_drift, &controls(p, w, &p.rabi_max), initial)?;

    // One trajectory from this initial state per drift and Rabi snapshot.  Batch elements do not always start from
    // it: a gate target evolves the whole computational basis.
    let mut seen = std::collections::HashSet::new();
    let snapshots: Vec<_> = p
        .batch
        .iter()
        .filter(|e| seen.insert((e.snapshot, e.rabi_index)))
        .collect();
    let trajectories = crate::parallel::map(snapshots.len(), |j| {
        let e = snapshots[j];
        trajectory(p, &e.h0, &controls(p, w, &e.rabi), initial)
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    let per_snapshot: Vec<Vec<[Vec<f64>; 3]>> = trajectories.iter().map(|t| observables_over(p, t)).collect();
    // The computational subspace's projector, `P·P†`, which is the identity where there is nowhere to leak.
    let projector = p.model.embed_density(&CMat::identity(1 << p.n_qubits))?;
    let leaks: Vec<Vec<f64>> = trajectories.iter().map(|t| leakage_over(&projector, t)).collect();
    let leak_fold = |pick_min: bool| -> Vec<f64> {
        (0..=p.n_pulse)
            .map(|t| {
                let vals = leaks.iter().map(|l| l[t]);
                if pick_min {
                    vals.fold(f64::INFINITY, f64::min)
                } else {
                    vals.fold(f64::NEG_INFINITY, f64::max)
                }
            })
            .collect()
    };
    let fold = |pick_min: bool| -> Vec<[Vec<f64>; 3]> {
        (0..p.n_qubits)
            .map(|q| {
                [0, 1, 2].map(|k| {
                    (0..=p.n_pulse)
                        .map(|t| {
                            let vals = per_snapshot.iter().map(|s| s[q][k][t]);
                            if pick_min {
                                vals.fold(f64::INFINITY, f64::min)
                            } else {
                                vals.fold(f64::NEG_INFINITY, f64::max)
                            }
                        })
                        .collect()
                })
            })
            .collect()
    };

    Ok(Dynamics {
        initial_state: i,
        labels,
        mean: pick(&mean_traj, &idx),
        snapshots: trajectories.iter().take(MAX_SNAPSHOTS).map(|t| pick(t, &idx)).collect(),
        observables: observables_over(p, &mean_traj),
        observables_min: fold(true),
        observables_max: fold(false),
        leakage: leakage_over(&projector, &mean_traj),
        leakage_min: leak_fold(true),
        leakage_max: leak_fold(false),
    })
}

/// Per qubit, `n` offsets in rad/s across 1.5 sweep widths about the configured offset.
fn offset_sweep(p: &Problem, n: usize) -> Vec<Vec<f64>> {
    (0..p.n_qubits)
        .map(|q| crate::basis::linspace(p.delta[q] - 0.75 * p.sw[q], p.delta[q] + 0.75 * p.sw[q], n))
        .collect()
}

/// Offsets in rad/s as Hz.
fn in_hz(offsets: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let two_pi = 2.0 * std::f64::consts::PI;
    offsets.iter().map(|o| o.iter().map(|v| v / two_pi).collect()).collect()
}

/// The drift with every qubit at its `j`-th swept offset.
fn drift_at(p: &Problem, offsets: &[Vec<f64>], j: usize) -> Result<CMat<f64>> {
    let at: Vec<f64> = offsets.iter().map(|o| o[j]).collect();
    p.model.drift(&at, p.coupling.as_ref())
}

/// `[qubit][axis][point]` from `[point][qubit][axis]`.
fn per_qubit(p: &Problem, finals: &[Vec<[f64; 3]>]) -> Vec<[Vec<f64>; 3]> {
    (0..p.n_qubits)
        .map(|q| [0, 1, 2].map(|k| finals.iter().map(|f| f[q][k]).collect()))
        .collect()
}

fn profile(p: &Problem, w: &Waveforms, i: usize) -> Result<ExcitationProfile> {
    let offsets = offset_sweep(p, PROFILE_POINTS);
    let u = controls(p, w, &p.rabi_max);
    let initial = &p.initial_states[i];
    let finals = crate::parallel::map(PROFILE_POINTS, |j| -> Result<Vec<[f64; 3]>> {
        Ok(bloch(p, &final_state(p, &drift_at(p, &offsets, j)?, &u, initial)?))
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    Ok(ExcitationProfile {
        initial_state: i,
        offsets_hz: in_hz(&offsets),
        xyz: per_qubit(p, &finals),
    })
}

/// One [`RabiMap`] per initial state.
fn rabi_maps(cfg: &Config, p: &Problem, w: &Waveforms) -> Result<Vec<RabiMap>> {
    let par = &cfg.parameters;
    // Relative spread per qubit; none where there is no maximum to be a fraction of.
    let spread: Vec<f64> = (0..p.n_qubits)
        .map(|q| {
            if par.omega_r_max[q] > 0.0 {
                par.sigma_omega_r_max[q] / par.omega_r_max[q]
            } else {
                0.0
            }
        })
        .collect();
    let reach = spread.iter().fold(0.5f64, |r, &s| r.max(4.0 * s));
    let scales = crate::basis::linspace((1.0 - reach).max(0.0), 1.0 + reach, MAP_POINTS);
    let drawn: Vec<[f64; 2]> = (0..p.n_qubits)
        .map(|q| {
            if p.rabi_max[q] == 0.0 {
                return [1.0, 1.0];
            }
            let fractions = p.batch.iter().map(|e| e.rabi[q] / p.rabi_max[q]);
            [
                fractions.clone().fold(f64::INFINITY, f64::min),
                fractions.fold(f64::NEG_INFINITY, f64::max),
            ]
        })
        .collect();

    let offsets = offset_sweep(p, MAP_POINTS);
    let drifts = (0..MAP_POINTS)
        .map(|j| drift_at(p, &offsets, j))
        .collect::<Result<Vec<_>>>()?;
    let rows: Vec<Vec<Vec<f64>>> = scales
        .iter()
        .map(|&s| {
            let rabi: Vec<f64> = (0..p.n_qubits)
                .map(|q| {
                    if spread[q] > 0.0 {
                        s * p.rabi_max[q]
                    } else {
                        p.rabi_max[q]
                    }
                })
                .collect();
            controls(p, w, &rabi)
        })
        .collect();

    (0..p.initial_states.len())
        .map(|i| {
            let initial = &p.initial_states[i];
            let finals = crate::parallel::map(MAP_POINTS * MAP_POINTS, |k| -> Result<Vec<[f64; 3]>> {
                let (r, c) = (k / MAP_POINTS, k % MAP_POINTS);
                Ok(bloch(p, &final_state(p, &drifts[c], &rows[r], initial)?))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
            Ok(RabiMap {
                initial_state: i,
                offsets_hz: in_hz(&offsets),
                scales: scales.clone(),
                xyz: per_qubit(p, &finals),
                drawn: drawn.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::examples;
    use crate::optim::Exit;

    fn result_for(solution: Vec<f64>) -> RunResult {
        RunResult {
            seed: 3,
            algorithm: "l-bfgs".into(),
            solution,
            fidelity: 0.0,
            penalty: 0.0,
            fidelity_history: vec![],
            iterations: 0,
            evaluations: 0,
            exit: Exit::Converged,
            elapsed_s: 0.0,
            notices: vec![],
            peak_amplitude: vec![],
        }
    }

    fn single() -> Config {
        let mut c = Config::from_json(examples()[0].1).unwrap();
        c.optimization.h0_snapshots = 5;
        c
    }

    #[test]
    fn a_zero_pulse_leaves_populations_alone() {
        let cfg = single();
        let n = Problem::build_with_seed(&cfg, 3).unwrap().n_params();
        let a = analyse(&cfg, &result_for(vec![0.0; n])).unwrap();
        let d = &a.dynamics[0];
        assert_eq!(d.labels, vec!["ψ1", "ψ2"]);
        for t in 0..a.times_ns.len() {
            let pop0 = d.mean[0][t][0].powi(2) + d.mean[0][t][1].powi(2);
            assert!((pop0 - 1.0).abs() < 1e-12);
            assert!((d.observables[0][2][t] - 1.0).abs() < 1e-12);
        }
        assert!(a.pulses[0].amp.iter().all(|&v| v == 0.0));
    }

    /// The mean trajectory's last point is the pulse applied to the initial state, as the objective sees it.
    #[test]
    fn the_last_state_matches_an_independent_propagation() {
        let mut cfg = single();
        cfg.parameters.coverage = vec![crate::config::Coverage::Single];
        let problem = Problem::build_with_seed(&cfg, 3).unwrap();
        let x: Vec<f64> = problem.x0.clone();
        let target = problem.batch[0].target.clone();
        let model = CostModel::new(problem);
        let fid = model.value(&x).unwrap().fidelity;
        let a = analyse(&cfg, &result_for(x)).unwrap();
        let last: Vec<C<f64>> = a.dynamics[0]
            .mean
            .iter()
            .map(|c| C::new(c.last().unwrap()[0], c.last().unwrap()[1]))
            .collect();
        let psi = CMat::column(last);
        let overlap = target.inner(&psi).norm_sqr();
        assert!((overlap - fid).abs() < 1e-12, "{overlap} vs {fid}");
    }

    /// States are plotted at the step boundaries, from the initial state at 0 to the final one at T.
    #[test]
    fn states_are_timed_at_the_step_boundaries() {
        let cfg = single();
        let p = Problem::build_with_seed(&cfg, 3).unwrap();
        let (n, duration) = (p.n_pulse, p.duration);
        let a = analyse(&cfg, &result_for(p.x0.clone())).unwrap();
        assert_eq!(a.state_times_ns.len(), n + 1);
        assert_eq!(a.state_times_ns[0], 0.0);
        assert!((a.state_times_ns[n] - duration * 1e9).abs() < 1e-9);
        let d = &a.dynamics[0];
        assert!(
            d.mean
                .iter()
                .chain(d.snapshots.iter().flatten())
                .all(|c| c.len() == n + 1)
        );
        assert!(
            d.observables
                .iter()
                .chain(&d.observables_min)
                .all(|o| o.iter().all(|v| v.len() == n + 1))
        );
        // |0>: ψ1 = 1 before the pulse.
        assert_eq!(d.mean[0][0], [1.0, 0.0]);
    }

    /// Leakage is the population outside the computational subspace: for one three-level transmon, `|ψ₃|²`.
    #[test]
    fn leakage_is_the_population_outside_the_computational_subspace() {
        let cfg = crate::hamiltonian::default_config("duffing_transmon", 1).unwrap();
        let p = Problem::build_with_seed(&cfg, 3).unwrap();
        let a = analyse(&cfg, &result_for(p.x0.clone())).unwrap();
        let d = &a.dynamics[0];
        assert_eq!(d.leakage.len(), a.state_times_ns.len());
        assert_eq!(d.leakage[0], 0.0, "the initial state is in the subspace");
        for (t, &l) in d.leakage.iter().enumerate() {
            let third = d.mean[2][t][0].powi(2) + d.mean[2][t][1].powi(2);
            assert!((l - third).abs() < 1e-12, "t {t}: {l} vs {third}");
            assert!((0.0..=1.0).contains(&l));
            assert!(d.leakage_min[t] <= d.leakage_max[t] && d.leakage_min[t] >= 0.0);
        }
        assert!(d.leakage.iter().any(|&l| l > 1e-9), "a transmon drive leaks something");
    }

    /// Two-level models have nowhere to leak to.
    #[test]
    fn two_level_models_report_no_leakage() {
        let cfg = single();
        let n = Problem::build_with_seed(&cfg, 3).unwrap().n_params();
        let a = analyse(&cfg, &result_for(vec![0.3; n])).unwrap();
        let worst = a.dynamics[0].leakage.iter().fold(0.0f64, |m, l| m.max(l.abs()));
        assert!(worst < 1e-12, "{worst}");
    }

    /// A gate run still plots the configured initial state, under every snapshot.
    #[test]
    fn gate_runs_plot_the_configured_initial_states() {
        let mut cfg = Config::from_json(examples().iter().find(|e| e.0 == "two_qubit_parameters").unwrap().1).unwrap();
        cfg.optimization.h0_snapshots = 3;
        cfg.parameters.coverage = vec![crate::config::Coverage::Broadband; 2];
        let n = Problem::build_with_seed(&cfg, 3).unwrap().n_params();
        let a = analyse(&cfg, &result_for(vec![0.0; n])).unwrap();
        let d = &a.dynamics[0];
        assert_eq!((d.mean.len(), d.snapshots.len()), (4, 3));
        // |Z, −Z> = |01>: ψ2 = 1 before the pulse, in every snapshot.
        assert!(d.snapshots.iter().all(|s| s[1][0] == [1.0, 0.0]));
    }

    /// The dissipative dynamics plot ends where the objective measures its fidelity.
    #[test]
    fn the_last_dissipative_state_matches_the_objective() {
        let mut cfg =
            Config::from_json(examples().iter().find(|e| e.0 == "single_qubit_dissipative").unwrap().1).unwrap();
        cfg.parameters.coverage = vec![crate::config::Coverage::Single];
        let problem = Problem::build_with_seed(&cfg, 3).unwrap();
        let x = problem.x0.clone();
        let target = problem.batch[0].target.clone();
        let model = CostModel::new(problem);
        let fid = model.value(&x).unwrap().fidelity;
        let a = analyse(&cfg, &result_for(x)).unwrap();
        let d = &a.dynamics[0];
        // Two levels, so the plot keeps the whole density matrix: ρ11, ρ12, ρ21, ρ22 in order.
        let rho = CMat::from_fn(2, 2, |r, c| {
            let last = d.mean[r * 2 + c].last().unwrap();
            C::new(last[0], last[1])
        });
        let got = target.matmul(&rho).unwrap().trace().re;
        assert!((got - fid).abs() < 1e-12, "{got} vs {fid}");
    }

    #[test]
    fn the_profile_spans_one_and_a_half_sweep_widths() {
        let cfg = single();
        let n = Problem::build_with_seed(&cfg, 3).unwrap().n_params();
        let a = analyse(&cfg, &result_for(vec![0.0; n])).unwrap();
        let o = &a.profiles[0].offsets_hz[0];
        assert_eq!(o.len(), PROFILE_POINTS);
        assert!((o[0] - (10e6 - 3.75e6)).abs() < 1e-3 && (o[o.len() - 1] - (10e6 + 3.75e6)).abs() < 1e-3);
        // No pulse: every offset leaves Z at +1.
        assert!(a.profiles[0].xyz[0][2].iter().all(|&z| (z - 1.0).abs() < 1e-12));
    }

    /// `single()` with a Rabi spread of `sigma` Hz on its 40 MHz qubit and `count` Rabi snapshots.
    fn spread(sigma: f64, count: usize) -> Config {
        let mut c = single();
        c.parameters.sigma_omega_r_max = vec![sigma];
        c.optimization.omega_r_snapshots = count;
        c
    }

    fn maps_for(cfg: &Config) -> Vec<RabiMap> {
        let p = Problem::build_with_seed(cfg, 3).unwrap();
        analyse(cfg, &result_for(p.x0.clone())).unwrap().rabi_maps
    }

    /// The map needs Rabi frequencies to sweep: a spread, drawn more than once.
    #[test]
    fn rabi_maps_need_a_spread_and_several_snapshots() {
        assert!(maps_for(&spread(0.0, 1)).is_empty());
        assert!(maps_for(&spread(0.0, 5)).is_empty());
        assert!(maps_for(&spread(4e6, 1)).is_empty());
        let maps = maps_for(&spread(4e6, 5));
        assert_eq!(maps.len(), 1, "one per initial state");
        let m = &maps[0];
        assert_eq!((m.offsets_hz[0].len(), m.scales.len()), (MAP_POINTS, MAP_POINTS));
        assert!(m.xyz[0].iter().all(|v| v.len() == MAP_POINTS * MAP_POINTS));
    }

    /// The middle of the map is the configured offset at the maximum Rabi frequency: where the mean dynamics end.
    #[test]
    fn the_rabi_map_centre_matches_the_mean_dynamics() {
        let cfg = spread(4e6, 5);
        let p = Problem::build_with_seed(&cfg, 3).unwrap();
        let a = analyse(&cfg, &result_for(p.x0.clone())).unwrap();
        let m = &a.rabi_maps[0];
        let c = MAP_POINTS / 2;
        assert!((m.offsets_hz[0][c] - 10e6).abs() < 1e-6 && (m.scales[c] - 1.0).abs() < 1e-12);
        for k in 0..3 {
            let got = m.xyz[0][k][c * MAP_POINTS + c];
            let want = *a.dynamics[0].observables[0][k].last().unwrap();
            assert!((got - want).abs() < 1e-9, "axis {k}: {got} vs {want}");
        }
    }

    /// Half to one and a half times the maximum, like the B1 maps, unless the draws reach further.
    #[test]
    fn the_rabi_map_spans_the_drawn_rabi_frequencies() {
        let m = &maps_for(&spread(4e6, 20))[0];
        assert!((m.scales[0] - 0.5).abs() < 1e-12 && (m.scales[MAP_POINTS - 1] - 1.5).abs() < 1e-12);
        let p = Problem::build_with_seed(&spread(4e6, 20), 3).unwrap();
        let drawn: Vec<f64> = p.batch.iter().map(|e| e.rabi[0] / p.rabi_max[0]).collect();
        let lo = drawn.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = drawn.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(m.drawn[0], [lo, hi]);
        assert!(lo < 1.0 && 1.0 < hi);

        // A 25 % spread reaches four standard deviations out: zero to twice the maximum.
        let wide = &maps_for(&spread(10e6, 20))[0];
        assert!(wide.scales[0].abs() < 1e-12 && (wide.scales[MAP_POINTS - 1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_zero_pulse_leaves_z_alone_across_the_rabi_map() {
        let cfg = spread(4e6, 5);
        let n = Problem::build_with_seed(&cfg, 3).unwrap().n_params();
        let a = analyse(&cfg, &result_for(vec![0.0; n])).unwrap();
        assert!(a.rabi_maps[0].xyz[0][2].iter().all(|&z| (z - 1.0).abs() < 1e-12));
    }

    /// A qubit without a spread keeps its Rabi frequency across the map: uncoupled, its own map does not change
    /// from row to row while its neighbour's does.
    #[test]
    fn qubits_without_a_spread_keep_their_rabi_frequency() {
        let mut cfg = Config::from_json(examples().iter().find(|e| e.0 == "two_qubit_parameters").unwrap().1).unwrap();
        cfg.parameters.j = None;
        cfg.parameters.point_in_pulse = vec![20; 2];
        cfg.parameters.sigma_omega_r_max = vec![4e6, 0.0];
        cfg.optimization.h0_snapshots = 3;
        cfg.optimization.omega_r_snapshots = 5;
        let m = &maps_for(&cfg)[0];
        assert_eq!(m.drawn[1], [1.0, 1.0]);
        let change = |q: usize| {
            let (first, last) = (
                &m.xyz[q][2][..MAP_POINTS],
                &m.xyz[q][2][(MAP_POINTS - 1) * MAP_POINTS..],
            );
            first.iter().zip(last).fold(0.0f64, |w, (a, b)| w.max((a - b).abs()))
        };
        assert!(change(1) < 1e-9, "{}", change(1));
        assert!(change(0) > 1e-3, "{}", change(0));
    }
}
