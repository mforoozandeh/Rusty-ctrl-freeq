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

#[test]
fn gate_targets_are_the_gate_applied_to_the_initial_state() {
    let mut cfg = example("two_qubit_parameters");
    cfg.initial_states = vec![vec!["-Z".into(), "Z".into()]];
    let p = Problem::build_with_seed(&cfg, 1).unwrap();
    // CNOT |10> = |11>
    let want = product_state(&["-Z".to_string(), "-Z".to_string()]).unwrap();
    assert!(p.batch[0].target.max_abs_diff(&want) < 1e-15);
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
    assert_eq!(p.batch[0].initial.shape(), (9, 1));
    assert_eq!(p.batch[0].target.shape(), (9, 1));
    let mut liou = cfg.clone();
    liou.optimization.space = Space::Liouville;
    let p = Problem::build_with_seed(&liou, 1).unwrap();
    assert_eq!(p.batch[0].initial.shape(), (9, 9));
    assert!((p.batch[0].initial.trace().re - 1.0).abs() < 1e-15);
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
