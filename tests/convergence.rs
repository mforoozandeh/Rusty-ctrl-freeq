//! The release gate: real problems reach the target with every optimiser that should manage it.
//!
//! These run the full bundled examples, so they are skipped in debug builds; run them with
//! `cargo test --release --test convergence`.

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::optim::{Exit, NoProgress, RecordingSink};
use ctrl_freeq::run;

fn example(name: &str) -> Config {
    let mut c = Config::from_json(examples().iter().find(|e| e.0 == name).unwrap().1).unwrap();
    c.seed = Some(7);
    c
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug builds; run with --release")]
fn the_single_qubit_example_converges_with_every_optimiser() {
    for alg in ctrl_freeq::optim::optimizer_names() {
        let mut cfg = example("single_qubit_parameters");
        cfg.optimization.algorithm = alg.to_string();
        let r = run(&cfg, &mut NoProgress).unwrap();
        assert_eq!(r.exit, Exit::TargetReached, "{alg}: F = {}", r.fidelity);
        assert!(r.fidelity - r.penalty >= 0.999, "{alg}");
        assert_eq!(r.seed, 7);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug builds; run with --release")]
fn the_two_qubit_cnot_converges_with_every_gradient_optimiser() {
    for alg in ["l-bfgs", "newton-cg", "newton-exact"] {
        let mut cfg = example("two_qubit_parameters");
        cfg.optimization.algorithm = alg.to_string();
        let r = run(&cfg, &mut NoProgress).unwrap();
        assert_eq!(r.exit, Exit::TargetReached, "{alg}: F = {}", r.fidelity);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug builds; run with --release")]
fn every_bundled_example_runs() {
    for (name, _) in examples() {
        let mut cfg = example(name);
        cfg.optimization.max_iter = 2;
        cfg.optimization.h0_snapshots = cfg.optimization.h0_snapshots.min(10);
        let mut sink = RecordingSink::default();
        let r = run(&cfg, &mut sink).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!sink.reports.is_empty(), "{name}");
        assert_eq!(r.fidelity_history.len(), sink.reports.len(), "{name}");
        assert!(!r.solution.is_empty(), "{name}");
    }
}

/// The same seed gives the same run, bit for bit.
#[test]
fn a_seeded_run_repeats_exactly() {
    let mut cfg = example("single_qubit_parameters_multiple_initial_targ");
    cfg.optimization.max_iter = 15;
    let a = run(&cfg, &mut NoProgress).unwrap();
    let b = run(&cfg, &mut NoProgress).unwrap();
    assert_eq!(a.solution, b.solution);
    assert_eq!(a.fidelity_history, b.fidelity_history);
}

#[test]
fn a_gpu_request_runs_on_the_cpu_with_a_notice() {
    let mut cfg = example("single_qubit_parameters_multiple_initial_targ");
    cfg.compute_resource = Some("gpu".into());
    cfg.cpu_cores = Some(2);
    cfg.optimization.max_iter = 3;
    let r = run(&cfg, &mut NoProgress).unwrap();
    assert_eq!(r.notices, vec!["GPU is not supported; running on CPU.".to_string()]);
}
