use super::*;
use crate::config::{Config, Dissipation, examples};
use crate::hamiltonian::default_config;

fn example(name: &str) -> Config {
    Config::from_json(examples().iter().find(|e| e.0 == name).unwrap().1).unwrap()
}

#[test]
fn every_example_builds() {
    for (name, json) in examples() {
        let cfg = Config::from_json(json).unwrap();
        let p = Problem::build_with_seed(&cfg, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.x0.len(), p.n_params(), "{name}");
        assert!(!p.batch.is_empty());
        for b in &p.batch {
            assert!(b.h0.is_hermitian(1e-6 * b.h0.norm1().max(1.0)), "{name}");
        }
    }
}

#[test]
fn a_seed_makes_the_build_reproducible() {
    let cfg = example("single_qubit_parameters");
    let a = Problem::build_with_seed(&cfg, 42).unwrap();
    let b = Problem::build_with_seed(&cfg, 42).unwrap();
    let c = Problem::build_with_seed(&cfg, 43).unwrap();
    assert_eq!(a.x0, b.x0);
    assert_eq!(a.batch, b.batch);
    assert_ne!(a.x0, c.x0);
}

/// Broadband single qubit: 100 offset snapshots, one Rabi snapshot, one initial state.
#[test]
fn broadband_samples_the_sweep_width() {
    let cfg = example("single_qubit_parameters");
    let p = Problem::build_with_seed(&cfg, 7).unwrap();
    assert_eq!((p.n_snapshots, p.n_rabi, p.batch.len()), (100, 1, 100));
    let two_pi = 2.0 * std::f64::consts::PI;
    for b in &p.batch {
        // H0 = Δ·Z with Z = σz/2, so H0[0,0] = Δ/2 and Δ is within 10 MHz ± 2.5 MHz.
        let delta = 2.0 * b.h0.get(0, 0).re / two_pi;
        assert!((7.5e6..=12.5e6).contains(&delta), "{delta}");
    }
}

/// Fixed offsets collapse to one snapshot; multiple initial states multiply the batch in order.
#[test]
fn batch_order_is_initial_state_then_snapshot_then_rabi() {
    let mut cfg = example("single_qubit_parameters_multiple_initial_targ");
    cfg.parameters.sigma_omega_r_max = vec![1e5];
    cfg.optimization.omega_r_snapshots = 3;
    let p = Problem::build_with_seed(&cfg, 3).unwrap();
    assert_eq!((p.n_snapshots, p.n_rabi), (1, 3));
    let order: Vec<(usize, usize, usize)> = p
        .batch
        .iter()
        .map(|b| (b.initial_index, b.snapshot, b.rabi_index))
        .collect();
    assert_eq!(order[..4], [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0)]);
    assert_eq!(p.batch.len(), 9);
}

/// Different gates for different initial states can only mean state transfer: each state to its gate's image.
#[test]
fn different_gates_per_state_are_state_targets() {
    let mut cfg = example("two_qubit_parameters_multiple_initial_targ");
    cfg.initial_states[0] = vec!["-Z".into(), "Z".into()];
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    // CNOT |10> = |11>
    let want = product_state(&["-Z".to_string(), "-Z".to_string()]).unwrap();
    assert!(p.batch[0].target.max_abs_diff(&want) < 1e-15);
}

/// One gate is a gate: every computational basis state goes in at once, whatever the initial states.
#[test]
fn one_gate_propagates_the_whole_computational_basis() {
    let mut cfg = example("two_qubit_parameters");
    cfg.initial_states.push(vec!["X".into(), "Y".into()]);
    cfg.target_states = Targets::Gate(vec!["CNOT".into(), "CNOT".into()]);
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    assert_eq!(p.batch.len(), p.n_snapshots * p.n_rabi);
    assert!(p.batch[0].initial.max_abs_diff(&CMat::identity(4)) < 1e-15);
    assert!(p.batch[0].target.max_abs_diff(&gate("CNOT", 2).unwrap()) < 1e-15);
    assert_eq!(p.initial_states.len(), 2, "the configured states remain for the plots");
}

/// With selective coverage a gate applies only where every qubit is in its band - here where qubit 2 is - and the
/// identity elsewhere.
#[test]
fn a_selective_gate_needs_every_qubit_in_band() {
    let mut cfg = default_config("spin_chain", 2).unwrap();
    cfg.parameters.coverage = vec![Coverage::Broadband, Coverage::Selective];
    cfg.optimization.h0_snapshots = 40;
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    let cnot = gate("CNOT", 2).unwrap();
    let (mut gates, mut idles) = (0, 0);
    for b in &p.batch {
        // H0 = δ1·Z1 + δ2·Z2 + coupling, so δ2 = H0[0,0] − H0[1,1]; in band, δ2 is exactly the configured offset.
        let in_band = (b.h0.get(0, 0).re - b.h0.get(1, 1).re - p.delta[1]).abs() < 1e-6 * p.delta[1];
        let want = if in_band { cnot.clone() } else { CMat::identity(4) };
        assert!(b.target.max_abs_diff(&want) < 1e-15);
        if in_band { gates += 1 } else { idles += 1 }
    }
    assert!(gates > 0 && idles > 0);
}

/// An axis target is a product state, so each selective qubit takes its target axis in its band and keeps its
/// initial axis outside it, independently of the other qubits.
#[test]
fn selective_axis_targets_are_chosen_per_qubit() {
    let mut cfg = default_config("spin_chain", 2).unwrap();
    cfg.parameters.coverage = vec![Coverage::Selective, Coverage::Broadband];
    cfg.optimization.h0_snapshots = 40;
    cfg.initial_states = vec![vec!["Z".into(), "Z".into()]];
    cfg.target_states = Targets::Axis(vec![vec!["-Z".into(), "-Z".into()]]);
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    let state = |a: &str, b: &str| product_state(&[a.to_string(), b.to_string()]).unwrap();
    let (both, second) = (state("-Z", "-Z"), state("Z", "-Z"));
    let count = |w: &CMat<f64>| p.batch.iter().filter(|b| b.target.max_abs_diff(w) < 1e-15).count();
    assert_eq!(count(&both) + count(&second), p.batch.len());
    assert!(count(&both) > 0 && count(&second) > 0);
}

#[test]
fn dissipative_runs_use_lindblad_in_liouville_space() {
    let cfg = example("single_qubit_dissipative");
    assert_eq!(cfg.optimization.dissipation_mode, Some(Dissipation::Dissipative));
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    assert_eq!(p.space, Space::Liouville);
    assert!(matches!(p.evolution, Evolution::Lindblad(_)));
    assert_eq!(p.batch[0].initial.shape(), (2, 2));
}

#[test]
fn duffing_states_are_embedded() {
    let cfg = default_config("duffing_transmon", 2).unwrap();
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    assert_eq!(p.dim, 9);
    assert_eq!(p.initial_states[0].shape(), (9, 1));
    // An iSWAP target: the four computational states evolve together.
    assert_eq!(
        (p.batch[0].initial.shape(), p.batch[0].target.shape()),
        ((9, 4), (9, 4))
    );
    let mut liou = cfg.clone();
    liou.optimization.space = Space::Liouville;
    let p = Problem::build_with_seed(&liou, 1).unwrap();
    assert_eq!(p.initial_states[0].shape(), (9, 9));
    assert!((p.initial_states[0].trace().re - 1.0).abs() < 1e-15);
    // Sixteen Pauli strings, the first the identity on the computational subspace.
    assert_eq!(p.batch.len(), 16);
    assert!((p.batch[0].initial.trace().re - 4.0).abs() < 1e-15);
}

#[test]
fn pulse_offset_modulates_the_waveform() {
    let mut cfg = example("single_qubit_parameters");
    cfg.parameters.pulse_offset = vec![1e6];
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    let t = p.times[10];
    let want = C::from_polar(1.0, 2.0 * std::f64::consts::PI * 1e6 * (t - p.duration / 2.0));
    assert!((p.modulation.get(10, 0) - want).norm() < 1e-12);
    assert!((p.dt - 2e-9).abs() < 1e-24);
}

#[test]
fn invalid_configs_are_refused() {
    let mut cfg = example("single_qubit_parameters");
    cfg.parameters.n_para = vec![3];
    assert!(matches!(Problem::build_with_seed(&cfg, 1), Err(Error::Config(_))));
}

/// Fixed offsets with an uncertain coupling still need one snapshot per coupling draw.
#[test]
fn a_coupling_spread_alone_gets_every_snapshot() {
    let mut cfg = default_config("spin_chain", 2).unwrap();
    cfg.parameters.coverage = vec![Coverage::Single; 2];
    cfg.parameters.sigma_j = Some(1e5);
    cfg.optimization.h0_snapshots = 10;
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    assert_eq!(p.n_snapshots, 10);
    // H0[1,2] = J/2 for XY coupling: one value per draw.
    let draws: std::collections::HashSet<u64> = p.batch.iter().map(|b| b.h0.get(1, 2).re.to_bits()).collect();
    assert_eq!(draws.len(), 10);
}

/// Each qubit's offsets are stratified around its own band independently, so some snapshots have one qubit inside
/// its band and the other outside.
#[test]
fn selective_qubits_are_sampled_independently() {
    let mut cfg = default_config("spin_chain", 2).unwrap();
    cfg.parameters.coverage = vec![Coverage::Selective; 2];
    cfg.optimization.h0_snapshots = 100;
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    let two_pi = 2.0 * std::f64::consts::PI;
    let half_band = two_pi * cfg.parameters.pulse_bandwidth[0] / 2.0;
    let mut mixed = 0;
    for k in 0..p.n_snapshots {
        let h0 = &p.batch.iter().find(|b| b.snapshot == k).unwrap().h0;
        // H0 = δ1·Z1 + δ2·Z2 + coupling: H0[0,0] = (δ1 + δ2)/2 and H0[1,1] = (δ1 − δ2)/2.
        let (a, b) = (h0.get(0, 0).re, h0.get(1, 1).re);
        let inside = |d: f64, q: usize| (d - p.delta[q]).abs() <= half_band;
        if inside(a + b, 0) != inside(a - b, 1) {
            mixed += 1;
        }
    }
    assert!(
        mixed > 20,
        "{mixed} of 100 snapshots mix in-band and out-of-band qubits"
    );
}

/// Leaked population lies outside every qubit's Bloch sphere: |2⟩ reads zero on each Pauli axis, not +1.
#[test]
fn leaked_population_has_no_pauli_expectation() {
    let p = Problem::build_with_seed(&default_config("duffing_transmon", 1).unwrap(), 1).unwrap();
    let o = C::new(0.0, 0.0);
    let two = CMat::column(vec![o, o, C::new(1.0, 0.0)]);
    for op in &p.observables[0] {
        assert_eq!(two.inner(&op.matmul(&two).unwrap()).re, 0.0);
    }
}

/// Sample t sits in the middle of step t, so the carrier advances by exactly offset·dt per step.
#[test]
fn waveform_samples_sit_at_the_step_midpoints() {
    let mut cfg = example("single_qubit_parameters");
    cfg.parameters.pulse_offset = vec![1e6];
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    for (t, &time) in p.times.iter().enumerate() {
        assert!((time - (t as f64 + 0.5) * p.dt).abs() < 1e-21, "{t}");
    }
    let per_step = p.modulation.get(1, 0) / p.modulation.get(0, 0);
    let want = C::from_polar(1.0, 2.0 * std::f64::consts::PI * 1e6 * p.dt);
    assert!((per_step - want).norm() < 1e-12);
}
