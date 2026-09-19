use super::*;
use crate::autodiff::C;
use crate::setup::{gate, product_state, spin_ops};

fn j2(u: f64, l: f64) -> Vec<Vec<f64>> {
    vec![vec![0.0, u], vec![l, 0.0]]
}

#[test]
fn drifts_are_hermitian() {
    let j = upper_coupling(&j2(3.0, 0.0), 2).unwrap();
    let models: Vec<Box<dyn HamiltonianModel>> = vec![
        Box::new(SpinChain::new(2, Coupling::Xyz)),
        Box::new(Superconducting::new(
            2,
            "XY+ZZ".into(),
            Some(vec![-10.0, -12.0]),
            None,
            Some(vec![0.1, 0.2]),
        )),
        Box::new(DuffingTransmon::new(2, vec![-10.0, -12.0]).unwrap()),
    ];
    for m in &models {
        let h = m.drift(&[1.0, -2.0], Some(&j)).unwrap();
        assert!(h.is_hermitian(1e-14), "{}", m.name());
        assert_eq!(h.rows, m.dim());
        assert_eq!(m.control_ops().len(), m.control_channels().len());
        for op in m.control_ops() {
            assert!(op.is_hermitian(1e-14));
        }
    }
}

#[test]
fn spin_chain_xy_drift_is_the_hand_built_matrix() {
    let ops = spin_ops(2);
    let j = upper_coupling(&j2(5.0, 0.0), 2).unwrap();
    let h = SpinChain::new(2, Coupling::Xy).drift(&[1.0, 2.0], Some(&j)).unwrap();
    let mut want = ops[0][2].scale_re(1.0).add(&ops[1][2].scale_re(2.0)).unwrap();
    want.axpy_re(5.0, &ops[0][0].matmul(&ops[1][0]).unwrap());
    want.axpy_re(5.0, &ops[0][1].matmul(&ops[1][1]).unwrap());
    assert!(h.max_abs_diff(&want) < 1e-15);
}

#[test]
fn either_triangle_of_j_gives_the_same_coupling() {
    let upper = upper_coupling(&j2(5.0, 0.0), 2).unwrap();
    let lower = upper_coupling(&j2(0.0, 5.0), 2).unwrap();
    let sym = upper_coupling(&j2(5.0, 5.0), 2).unwrap();
    assert_eq!(upper, lower);
    assert_eq!(upper, sym);
    assert!(upper_coupling(&j2(5.0, 4.0), 2).unwrap_err().contains("asymmetric"));
}

#[test]
fn zz_crosstalk_takes_priority_over_the_anharmonicity_formula() {
    let g = upper_coupling(&j2(3.0, 0.0), 2).unwrap();
    let mut zz = RMat::<f64>::zeros(2, 2);
    zz.set(0, 1, 0.7);
    let with_formula = Superconducting::new(2, "ZZ".into(), Some(vec![-10.0, -10.0]), None, None);
    let with_calibrated = Superconducting::new(2, "ZZ".into(), Some(vec![-10.0, -10.0]), Some(zz), None);
    let zz_op = |m: &Superconducting| {
        let h = m.drift(&[0.0, 0.0], Some(&g)).unwrap();
        // |00> sees +ζ/4.
        h.get(0, 0).re * 4.0
    };
    assert!((zz_op(&with_formula) - 2.0 * 9.0 * (-0.2)).abs() < 1e-12);
    assert!((zz_op(&with_calibrated) - 0.7).abs() < 1e-12);
}

#[test]
fn stark_shift_adds_a_power_channel_per_qubit() {
    let m = Superconducting::new(2, "XY".into(), None, None, Some(vec![0.1, 0.2]));
    let ch = m.control_channels();
    assert_eq!(ch.len(), 6);
    assert_eq!(
        ch[2],
        ControlChannel {
            qubit: 0,
            source: Source::Power,
            rabi_power: 2,
            coeff: 0.1
        }
    );
}

#[test]
fn duffing_embeds_the_computational_basis() {
    let m = DuffingTransmon::new(2, vec![-10.0, -12.0]).unwrap();
    assert_eq!(m.dim(), 9);
    let s = |axes: &[&str]| product_state(&axes.iter().map(|a| a.to_string()).collect::<Vec<_>>()).unwrap();
    // |01> (computational index 1) -> ternary index 1; |10> (index 2) -> ternary index 3.
    let e01 = m.embed_state(&s(&["Z", "-Z"])).unwrap();
    let e10 = m.embed_state(&s(&["-Z", "Z"])).unwrap();
    assert_eq!(e01.get(1, 0), C::new(1.0, 0.0));
    assert_eq!(e10.get(3, 0), C::new(1.0, 0.0));
    let g = m.embed_gate(&gate("iSWAP", 2).unwrap()).unwrap();
    assert!(g.matmul(&g.adjoint()).unwrap().max_abs_diff(&CMat::identity(9)) < 1e-14);
    assert!(m.leakage(&e01).abs() < 1e-15);
}

#[test]
fn default_configs_validate() {
    for name in model_names() {
        for n in [1, 2] {
            let c = default_config(name, n).unwrap();
            assert!(c.validate().is_empty(), "{name} {n}: {:?}", c.validate());
            assert!(model_from_config(&c).is_ok());
        }
    }
    assert!(default_config("nope", 1).is_err());
}

#[test]
fn the_legacy_path_defaults_to_z_coupling() {
    let mut c = default_config("spin_chain", 2).unwrap();
    c.hamiltonian_type = None;
    c.parameters.coupling_type = None;
    let m = model_from_config(&c).unwrap();
    let j = upper_coupling(&j2(4.0, 0.0), 2).unwrap();
    let h = m.drift(&[0.0, 0.0], Some(&j)).unwrap();
    let ops = spin_ops(2);
    let want = ops[0][2].matmul(&ops[1][2]).unwrap().scale_re(4.0);
    assert!(h.max_abs_diff(&want) < 1e-15);
}
