//! The objective's gradient and Hessian-vector product against finite differences, for every combination of state
//! space, Hamiltonian model and waveform mode, on small problems.

use ctrl_freeq::config::{Config, Dissipation, Space, Targets, WaveformMode};
use ctrl_freeq::hamiltonian::default_config;
use ctrl_freeq::objective::CostModel;
use ctrl_freeq::setup::Problem;

#[derive(Clone, Copy, Debug)]
enum Model {
    Legacy,
    SuperconductingStark,
    Duffing,
}

fn small(model: Model, space: Space, dissipative: bool, mode: WaveformMode) -> Config {
    let mut c = match model {
        Model::Legacy => {
            let mut c = default_config("spin_chain", 2).unwrap();
            c.hamiltonian_type = None;
            c
        }
        Model::SuperconductingStark => {
            let mut c = default_config("superconducting", 2).unwrap();
            c.parameters.coupling_type = Some("XY+ZZ".into());
            c.parameters.anharmonicities = Some(vec![-330e6, -310e6]);
            c.parameters.stark_shift_coeffs = Some(vec![1e-9, 2e-9]);
            c
        }
        Model::Duffing => default_config("duffing_transmon", 2).unwrap(),
    };
    let p = &mut c.parameters;
    p.point_in_pulse = vec![12; 2];
    p.n_para = vec![4; 2];
    p.wf_mode = vec![mode; 2];
    p.pulse_offset = vec![0.3e6, -0.2e6];
    p.sigma_omega_r_max = vec![1e6, 2e6];
    c.optimization.h0_snapshots = 2;
    c.optimization.omega_r_snapshots = 2;
    c.optimization.space = space;
    if dissipative {
        c.optimization.dissipation_mode = Some(Dissipation::Dissipative);
        c.parameters.t1 = Some(vec![2e-6, 3e-6]);
        c.parameters.t2 = Some(vec![1e-6, 2e-6]);
    }
    let gate = if matches!(model, Model::Legacy) {
        "CNOT"
    } else {
        "iSWAP"
    };
    c.initial_states = vec![vec!["Z".into(), "-Z".into()], vec!["X".into(), "Y".into()]];
    c.target_states = Targets::Gate(vec![gate.into(), gate.into()]);
    c.seed = Some(11);
    assert!(c.validate().is_empty(), "{:?}", c.validate());
    c
}

fn model(cfg: &Config) -> CostModel {
    CostModel::new(Problem::build(cfg).unwrap())
}

fn assert_close(what: &str, got: f64, want: f64, rel: f64) {
    let scale = got.abs().max(want.abs()).max(1e-3);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: got {got}, finite differences {want}"
    );
}

fn check_gradient(cfg: &Config, label: &str) {
    let m = model(cfg);
    let x: Vec<f64> = m.problem().x0.iter().map(|v| 0.6 * v).collect();
    let (e, g) = m.value_and_gradient(&x).unwrap();
    assert_eq!(e.cost, m.value(&x).unwrap().cost, "{label}: value paths agree");
    assert!(
        e.fidelity > 0.0 && e.fidelity <= 1.0 + 1e-12,
        "{label}: fidelity {}",
        e.fidelity
    );
    let h = 1e-6;
    for i in 0..x.len() {
        let mut p = x.clone();
        let mut q = x.clone();
        p[i] += h;
        q[i] -= h;
        let fd = (m.value(&p).unwrap().cost - m.value(&q).unwrap().cost) / (2.0 * h);
        assert_close(&format!("{label} d/dx{i}"), g[i], fd, 1e-5);
    }
}

#[test]
fn gradients_match_finite_differences_everywhere() {
    let spaces = [
        (Space::Hilbert, false),
        (Space::Liouville, false),
        (Space::Liouville, true),
    ];
    for model in [Model::Legacy, Model::SuperconductingStark, Model::Duffing] {
        for &(space, dissipative) in &spaces {
            if dissipative && matches!(model, Model::Duffing) {
                continue;
            }
            for mode in WaveformMode::ALL {
                let cfg = small(model, space, dissipative, mode);
                check_gradient(&cfg, &format!("{model:?} {space:?} dissipative={dissipative} {mode:?}"));
            }
        }
    }
}

#[test]
fn a_large_amplitude_activates_the_penalty_gradient() {
    let cfg = small(Model::Legacy, Space::Hilbert, false, WaveformMode::Cart);
    let m = model(&cfg);
    let x: Vec<f64> = m.problem().x0.iter().map(|v| 8.0 * v).collect();
    let (e, _) = m.value_and_gradient(&x).unwrap();
    assert!(e.penalty > 0.0);
    check_gradient_at(&m, &x, "penalised");
}

fn check_gradient_at(m: &CostModel, x: &[f64], label: &str) {
    let (_, g) = m.value_and_gradient(x).unwrap();
    let h = 1e-6;
    for i in 0..x.len() {
        let mut p = x.to_vec();
        let mut q = x.to_vec();
        p[i] += h;
        q[i] -= h;
        let fd = (m.value(&p).unwrap().cost - m.value(&q).unwrap().cost) / (2.0 * h);
        assert_close(&format!("{label} d/dx{i}"), g[i], fd, 1e-5);
    }
}

#[test]
fn hessian_vector_products_match_finite_differences_of_the_gradient() {
    for (space, dissipative) in [(Space::Hilbert, false), (Space::Liouville, true)] {
        for mode in [WaveformMode::Cart, WaveformMode::PolarPhase] {
            let cfg = small(Model::Legacy, space, dissipative, mode);
            let m = model(&cfg);
            let x: Vec<f64> = m.problem().x0.iter().map(|v| 0.6 * v).collect();
            let v: Vec<f64> = (0..x.len()).map(|i| ((i * 7 % 5) as f64 - 2.0) / 3.0).collect();
            let hv = m.hvp(&x, &v).unwrap();
            let h = 1e-5;
            let plus: Vec<f64> = x.iter().zip(&v).map(|(a, d)| a + h * d).collect();
            let minus: Vec<f64> = x.iter().zip(&v).map(|(a, d)| a - h * d).collect();
            let (_, gp) = m.value_and_gradient(&plus).unwrap();
            let (_, gm) = m.value_and_gradient(&minus).unwrap();
            for i in 0..x.len() {
                let fd = (gp[i] - gm[i]) / (2.0 * h);
                assert_close(&format!("{space:?} {mode:?} Hv[{i}]"), hv[i], fd, 1e-5);
            }
            let hess = m.hessian_matrix(&x).unwrap();
            for i in 0..x.len() {
                let row: f64 = (0..x.len()).map(|j| hess[(i, j)] * v[j]).sum();
                assert_close("H·v from the matrix", row, hv[i], 1e-9);
            }
        }
    }
}

#[test]
fn wrong_lengths_are_errors() {
    let m = model(&small(Model::Legacy, Space::Hilbert, false, WaveformMode::Cart));
    assert!(m.value(&[0.0; 3]).is_err());
    assert!(m.hvp(&m.problem().x0, &[0.0; 3]).is_err());
}
