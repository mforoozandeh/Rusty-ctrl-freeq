use super::*;
use crate::config::{Config, examples};

#[test]
fn waveforms_have_one_column_per_qubit() {
    let cfg = Config::from_json(examples().iter().find(|e| e.0 == "two_qubit_parameters").unwrap().1).unwrap();
    let m = CostModel::new(Problem::build_with_seed(&cfg, 2).unwrap());
    let w = m.waveforms(&m.problem().x0).unwrap();
    assert_eq!(w.cx.shape(), (100, 2));
    for r in 0..100 {
        assert!((w.amp.get(r, 1) - w.cx.get(r, 1).hypot(w.cy.get(r, 1))).abs() < 1e-15);
    }
}

#[test]
fn the_fidelity_of_a_zero_pulse_is_the_overlap_of_free_evolution() {
    // Z -> -Z with no pulse: the state stays at |0> (up to phase), fidelity 0.
    let cfg = Config::from_json(examples()[0].1).unwrap();
    let m = CostModel::new(Problem::build_with_seed(&cfg, 2).unwrap());
    let e = m.value(&vec![0.0; m.dim()]).unwrap();
    assert!(e.fidelity.abs() < 1e-12 && e.penalty == 0.0 && e.cost == -e.fidelity);
}

/// Steps six times longer than T1 relax |1> to |0> almost completely.  The fidelity is the exact 1 − e^(−6), where
/// an explicit Euler step of the dissipator gave 2.
#[test]
fn long_lindblad_steps_stay_physical() {
    let mut cfg = Config::from_json(examples().iter().find(|e| e.0 == "single_qubit_dissipative").unwrap().1).unwrap();
    cfg.parameters.pulse_duration = vec![6e-6];
    cfg.parameters.point_in_pulse = vec![3];
    cfg.parameters.n_para = vec![2];
    cfg.parameters.t1 = Some(vec![1e-6]);
    cfg.parameters.t2 = Some(vec![2e-6]);
    cfg.initial_states = vec![vec!["-Z".into()]];
    cfg.target_states = crate::config::Targets::Axis(vec![vec!["Z".into()]]);
    let m = CostModel::new(Problem::build_with_seed(&cfg, 1).unwrap());
    let e = m.value(&vec![0.0; m.dim()]).unwrap();
    assert!((e.fidelity - (1.0 - (-6.0f64).exp())).abs() < 1e-12, "{}", e.fidelity);
}

/// One qubit, no pulse, and a drift of `δ` for `T` with `δ·T` set by `turn`: a Z rotation by `2π·turn`.
fn single_z_config(turn: f64, target: &str, dissipative: bool) -> Config {
    let name = if dissipative {
        "single_qubit_dissipative"
    } else {
        "single_qubit_parameters"
    };
    let mut cfg = Config::from_json(examples().iter().find(|e| e.0 == name).unwrap().1).unwrap();
    let duration = cfg.parameters.pulse_duration[0];
    cfg.parameters.delta = Some(vec![turn / duration]);
    cfg.parameters.coverage = vec![crate::config::Coverage::Single];
    cfg.initial_states = vec![vec!["Z".into()]];
    cfg.target_states = crate::config::Targets::Gate(vec![target.into()]);
    cfg
}

fn zero_pulse_fidelity(cfg: &Config) -> f64 {
    let m = CostModel::new(Problem::build_with_seed(cfg, 1).unwrap());
    m.value(&vec![0.0; m.dim()]).unwrap().fidelity
}

/// A Z-gate target with |0> as the initial state: doing nothing leaves |0> alone, yet the identity's average gate
/// fidelity to Z is (2 + |Tr Z|²)/6 = 1/3.  Half a turn about z is Z up to a global phase: fidelity 1.
#[test]
fn a_gate_target_is_scored_by_the_average_gate_fidelity() {
    for space in [crate::config::Space::Hilbert, crate::config::Space::Liouville] {
        let mut idle = single_z_config(0.0, "Z", false);
        let mut z = single_z_config(0.5, "Z", false);
        idle.optimization.space = space;
        z.optimization.space = space;
        assert!((zero_pulse_fidelity(&idle) - 1.0 / 3.0).abs() < 1e-12, "{space:?}");
        assert!((zero_pulse_fidelity(&z) - 1.0).abs() < 1e-12, "{space:?}");
    }
}

/// Under relaxation the average gate fidelity is 1/2 + (2·e^(−T/T2) + e^(−T/T1))/6, the trace of the channel's
/// Bloch-vector map over six; the Z rotation commutes with the relaxation and drops out.
#[test]
fn the_gate_fidelity_counts_relaxation() {
    let cfg = single_z_config(0.5, "Z", true);
    let p = &cfg.parameters;
    let (t, t1, t2) = (
        p.pulse_duration[0],
        p.t1.as_ref().unwrap()[0],
        p.t2.as_ref().unwrap()[0],
    );
    let want = 0.5 + (2.0 * (-t / t2).exp() + (-t / t1).exp()) / 6.0;
    let got = zero_pulse_fidelity(&cfg);
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

/// `CX` and `CNOT` are one gate, so both spellings ask for the gate, not for state transfer.  With no drift and no
/// pulse the identity's average gate fidelity to CNOT is (4 + |Tr CNOT|²)/20 = 0.4.
#[test]
fn gate_aliases_name_the_same_gate() {
    let mut cfg = Config::from_json(examples().iter().find(|e| e.0 == "two_qubit_parameters").unwrap().1).unwrap();
    cfg.parameters.delta = Some(vec![0.0, 0.0]);
    cfg.parameters.j = Some(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    cfg.initial_states = vec![vec!["Z".into(), "Z".into()], vec!["Z".into(), "-Z".into()]];
    for names in [["CNOT", "CNOT"], ["CNOT", "CX"], ["CX", "CNOT"]] {
        cfg.target_states = crate::config::Targets::Gate(names.map(String::from).to_vec());
        let got = zero_pulse_fidelity(&cfg);
        assert!((got - 0.4).abs() < 1e-12, "{names:?}: {got}");
    }
}
