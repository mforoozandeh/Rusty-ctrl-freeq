//! Configuration parsing, round trips and validation.

use ctrl_freeq::config::{Config, Targets, examples};

#[test]
fn every_bundled_example_parses_validates_and_round_trips() {
    for (name, json) in examples() {
        let cfg = Config::from_json(json).unwrap_or_else(|e| panic!("{name}: {e}"));
        let problems = cfg.validate();
        assert!(problems.is_empty(), "{name}: {problems:?}");
        let again = Config::from_json(&cfg.to_json()).unwrap();
        assert_eq!(cfg, again, "{name}");
    }
}

#[test]
fn no_bundled_example_names_a_qiskit_algorithm() {
    for (name, json) in examples() {
        assert!(!json.contains("qiskit"), "{name}");
    }
}

fn single_qubit() -> Config {
    Config::from_json(examples()[0].1).unwrap()
}

/// The Python GUI can write every target key, filling the unused ones with nulls, and nulls elsewhere too.
#[test]
fn gui_shaped_json_parses() {
    let json = r#"{
        "hamiltonian_type": "spin_chain",
        "qubits": ["q1", "q2"],
        "compute_resource": "cpu",
        "parameters": {
            "Delta": [1e7, 1e7], "sigma_Delta": [0.0, 0.0], "Omega_R_max": [4e7, 4e7],
            "pulse_duration": [2e-7, 2e-7], "point_in_pulse": [100, 100], "wf_type": ["cheb", "cheb"],
            "wf_mode": ["cart", "cart"], "amplitude_envelope": ["gn", "gn"], "amplitude_order": [1, 1],
            "coverage": ["single", "single"], "sw": [5e6, 5e6], "pulse_offset": [0.0, 0.0],
            "pulse_bandwidth": [5e5, 5e5], "ratio_factor": [0.5, 0.5], "sigma_Omega_R_max": [0.0, 0.0],
            "profile_order": [2, 2], "n_para": [16, 16], "J": [[0.0, 1.6e7], [0.0, 0.0]],
            "coupling_type": "XY", "sigma_J": null, "T1": [], "T2": []
        },
        "initial_states": [["Z", "-Z"]],
        "target_states": {"Axis": [null], "Phi": [null], "Beta": [null], "Gate": ["CNOT"]},
        "optimization": {"space": "hilbert", "dissipation_mode": "non-dissipative", "H0_snapshots": 10,
                         "Omega_R_snapshots": 1, "algorithm": "l-bfgs", "max_iter": 100, "targ_fid": 0.999}
    }"#;
    let cfg = Config::from_json(json).unwrap();
    assert_eq!(cfg.target_states, Targets::Gate(vec!["CNOT".into()]));
    assert!(cfg.validate().is_empty(), "{:?}", cfg.validate());
    // Only the chosen target key is written back.
    let out = cfg.to_json();
    assert!(out.contains("\"Gate\"") && !out.contains("\"Axis\"") && !out.contains("\"Phi\""));
}

#[test]
fn phi_beta_targets_parse_together() {
    let mut cfg = single_qubit();
    let json = cfg.to_json().replace(
        r#""Axis": [
      [
        "-Z"
      ]
    ]"#,
        r#""Phi": [["x"]], "Beta": [[90.0]]"#,
    );
    cfg = Config::from_json(&json).unwrap();
    assert!(matches!(cfg.target_states, Targets::PhiBeta { .. }));
    assert!(cfg.validate().is_empty());
}

/// Each rule produces a sentence naming what is wrong.
#[test]
fn validation_messages_name_the_problem() {
    let expect = |cfg: &Config, needle: &str| {
        let problems = cfg.validate();
        assert!(
            problems.iter().any(|p| p.contains(needle)),
            "wanted \"{needle}\" in {problems:?}"
        );
    };

    let mut c = single_qubit();
    c.parameters.sw.push(1.0);
    expect(&c, "sw has 2 entries for 1 qubits");

    let mut c = single_qubit();
    c.optimization.algorithm = "qiskit-spsa".into();
    expect(&c, "is not supported; choose one of l-bfgs");

    let mut c = single_qubit();
    c.parameters.n_para = vec![15];
    expect(&c, "n_para must be even");

    let mut c = single_qubit();
    c.parameters.wf_type = vec!["bessel".into()];
    expect(&c, "wf_type \"bessel\"");

    let mut c = single_qubit();
    c.parameters.point_in_pulse = vec![4];
    expect(&c, "basis functions need at least as many points");

    let mut c = single_qubit();
    c.target_states = Targets::Gate(vec!["CNOT".into()]);
    expect(&c, "gate \"CNOT\" is not one of X, Y, Z");

    let mut c = single_qubit();
    c.initial_states = vec![vec!["Q".into()]];
    expect(&c, "\"Q\" is not one of");

    let mut c = single_qubit();
    c.optimization.targ_fid = 1.5;
    expect(&c, "targ_fid");

    let mut c = single_qubit();
    c.optimization.max_iter = 0;
    expect(&c, "max_iter");

    let mut c = Config::from_json(examples().iter().find(|e| e.0 == "single_qubit_dissipative").unwrap().1).unwrap();
    c.parameters.t2 = Some(vec![1.0]);
    c.parameters.t1 = Some(vec![0.1]);
    expect(&c, "T2 must not exceed 2·T1");
    c.parameters.t1 = None;
    expect(&c, "needs T1 and T2");

    let mut c = single_qubit();
    c.hamiltonian_type = Some("duffing_transmon".into());
    expect(&c, "needs anharmonicities");

    let mut c = single_qubit();
    c.hamiltonian_type = Some("trapped_ion".into());
    expect(&c, "hamiltonian_type \"trapped_ion\"");

    let mut c = single_qubit();
    c.target_states = Targets::Axis(vec![vec!["Z".into()], vec!["X".into()]]);
    expect(&c, "2 target entries for 1 initial states");
}

#[test]
fn syntax_errors_are_config_errors() {
    assert!(matches!(Config::from_json("{"), Err(ctrl_freeq::Error::Config(_))));
    let bad_mode = examples()[0].1.replace("\"cart\"", "\"spherical\"");
    let err = Config::from_json(&bad_mode).unwrap_err().to_string();
    assert!(err.contains("spherical"), "{err}");
}

/// A smooth band-selective profile has no in-band region for a state or gate to apply in; rotations scale with it.
#[test]
fn band_selective_coverage_needs_rotation_targets() {
    let mut cfg = single_qubit();
    cfg.parameters.coverage = vec![ctrl_freeq::config::Coverage::BandSelective];
    assert!(cfg.validate().iter().any(|p| p.contains("band_selective")), "axis");
    cfg.target_states = Targets::Gate(vec!["X".into()]);
    assert!(cfg.validate().iter().any(|p| p.contains("band_selective")), "gate");
    cfg.target_states = Targets::PhiBeta {
        phi: vec![vec!["x".into()]],
        beta: vec![vec![180.0]],
    };
    assert!(cfg.validate().is_empty(), "{:?}", cfg.validate());
}

/// A band-selective profile needs a width to be half of, and an order to be a super-Gaussian of.
#[test]
fn band_selective_coverage_needs_a_bandwidth_and_an_order() {
    let mut cfg = single_qubit();
    cfg.parameters.coverage = vec![ctrl_freeq::config::Coverage::BandSelective];
    cfg.target_states = Targets::PhiBeta {
        phi: vec![vec!["x".into()]],
        beta: vec![vec![180.0]],
    };
    assert!(cfg.validate().is_empty(), "{:?}", cfg.validate());
    cfg.parameters.profile_order = vec![0];
    assert!(
        cfg.validate().iter().any(|p| p.contains("profile_order")),
        "{:?}",
        cfg.validate()
    );
    cfg.parameters.profile_order = vec![2];
    cfg.parameters.pulse_bandwidth = vec![0.0];
    assert!(
        cfg.validate().iter().any(|p| p.contains("pulse_bandwidth")),
        "{:?}",
        cfg.validate()
    );
}
