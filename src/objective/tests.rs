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

fn config(name: &str) -> Config {
    Config::from_json(examples().iter().find(|e| e.0 == name).unwrap().1).unwrap()
}

/// The two-qubit Duffing transmon fixture: three levels per qubit, so the gate's columns can leak.
fn duffing() -> Config {
    let v: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/two_duffing.json")).unwrap();
    Config::from_json(&v["config"].to_string()).unwrap()
}

/// `‖r‖² = cost + 1`, the last residual squared is the penalty, and the `Eval` is `value()`'s.
fn assert_least_squares_form(cfg: &Config, scale: f64) -> Eval {
    let m = CostModel::new(Problem::build_with_seed(cfg, 3).unwrap());
    let x: Vec<f64> = m
        .problem()
        .x0
        .iter()
        .enumerate()
        .map(|(i, v)| scale * (v + 0.3 * (1.7 * i as f64).sin()))
        .collect();
    let (e, r) = m.residuals(&x).unwrap();
    let want = m.value(&x).unwrap();
    let norm: f64 = r.iter().map(|v| v * v).sum();
    assert!(
        (norm - (e.cost + 1.0)).abs() < 1e-12,
        "‖r‖² = {norm}, cost + 1 = {}",
        e.cost + 1.0
    );
    assert!((r.last().unwrap().powi(2) - e.penalty).abs() < 1e-12);
    for (a, b) in [
        (e.cost, want.cost),
        (e.fidelity, want.fidelity),
        (e.penalty, want.penalty),
    ] {
        assert!((a - b).abs() < 1e-12, "{e:?} vs {want:?}");
    }
    e
}

#[test]
fn the_residuals_are_the_cost_in_least_squares_form() {
    assert_least_squares_form(&config("single_qubit_parameters"), 1.0); // a state target, 100 drift snapshots
    assert_least_squares_form(&config("single_qubit_parameters_multiple_initial_targ"), 1.0); // several states
    assert_least_squares_form(&config("single_qubit_parameters_polar_phase"), 1.0);
    assert_least_squares_form(&config("two_qubit_parameters"), 1.0); // a CNOT
    assert_least_squares_form(&config("two_qubit_parameters_multiple_initial_targ"), 1.0); // gates per state
    assert_least_squares_form(&duffing(), 1.0); // a gate with leakage levels
    let mut rabi = config("single_qubit_parameters");
    rabi.optimization.h0_snapshots = 5;
    rabi.optimization.omega_r_snapshots = 3;
    rabi.parameters.sigma_omega_r_max = vec![0.1];
    assert_least_squares_form(&rabi, 1.0);
}

#[test]
fn the_last_residual_is_the_root_of_the_amplitude_penalty() {
    let e = assert_least_squares_form(&config("single_qubit_parameters"), 40.0);
    assert!(e.penalty > 0.0, "the scaled pulse should exceed the limit: {e:?}");
}

#[test]
fn liouville_problems_have_no_least_squares_form() {
    let mut liouville = config("single_qubit_parameters");
    liouville.optimization.space = crate::config::Space::Liouville;
    for cfg in [config("single_qubit_dissipative"), liouville] {
        let m = CostModel::new(Problem::build_with_seed(&cfg, 1).unwrap());
        let err = m.residuals(&m.problem().x0).unwrap_err();
        assert!(matches!(err, Error::NotSupported(_)), "{err}");
    }
}

/// The identity behind the residuals needs orthonormal columns on both sides.
#[test]
fn hilbert_states_and_targets_have_orthonormal_columns() {
    let mut cfgs: Vec<Config> = examples()
        .iter()
        .map(|e| Config::from_json(e.1).unwrap())
        .filter(|c| c.space() == crate::config::Space::Hilbert)
        .collect();
    cfgs.push(duffing());
    for cfg in cfgs {
        let p = Problem::build_with_seed(&cfg, 1).unwrap();
        for e in &p.batch {
            for m in [&e.initial, &e.target] {
                let g = m.adjoint().matmul(m).unwrap();
                assert!(g.max_abs_diff(&crate::linalg::CMat::identity(g.rows)) < 1e-12);
            }
        }
    }
}
