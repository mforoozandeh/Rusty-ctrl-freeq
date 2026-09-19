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
