//! Edits to a configuration that keep it consistent: every per-qubit array as long as the qubit list, couplings
//! square, targets matching the initial states, model-specific fields present when the model needs them.
//!
//! The interface changes the configuration only through these for anything structural, so it never produces a
//! file the Python package or the Rust library would reject for shape reasons.

use ctrl_freeq::config::{Config, Dissipation, Targets};
use ctrl_freeq::setup::gate_names;

/// Target methods, as the interface names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetMethod {
    /// A product state per initial state.
    Axis,
    /// A gate per initial state.
    Gate,
    /// A rotation per qubit per initial state.
    Rotation,
}

impl TargetMethod {
    /// The method `targets` uses.
    pub fn of(targets: &Targets) -> Self {
        match targets {
            Targets::Axis(_) => TargetMethod::Axis,
            Targets::Gate(_) => TargetMethod::Gate,
            Targets::PhiBeta { .. } => TargetMethod::Rotation,
        }
    }

    /// Label for the interface.
    pub fn label(self) -> &'static str {
        match self {
            TargetMethod::Axis => "State",
            TargetMethod::Gate => "Gate",
            TargetMethod::Rotation => "Rotation",
        }
    }
}

/// Coupling types the model accepts, the first being its default.
pub fn coupling_types(model: Option<&str>) -> &'static [&'static str] {
    match model {
        Some("superconducting") => &["XY", "ZZ", "XY+ZZ"],
        Some("duffing_transmon") => &["XY"],
        Some(_) => &["XY", "Z", "XYZ"],
        None => &["Z", "XY", "XYZ"],
    }
}

fn resize<T: Clone>(v: &mut Vec<T>, n: usize, fill: T) {
    let fill = v.last().cloned().unwrap_or(fill);
    v.resize(n, fill);
}

fn resize_opt<T: Clone>(v: &mut Option<Vec<T>>, n: usize, fill: T) {
    if let Some(v) = v {
        resize(v, n, fill);
    }
}

fn resize_matrix(m: &mut Vec<Vec<f64>>, n: usize) {
    m.resize(n, Vec::new());
    for row in m.iter_mut() {
        row.resize(n, 0.0);
    }
}

/// The default target for one initial state under `method` with `n` qubits.
fn default_axis(n: usize) -> Vec<String> {
    vec!["-Z".to_string(); n]
}

fn default_gate(n: usize) -> Option<String> {
    gate_names(n).first().map(|g| g.to_string())
}

/// Set the number of qubits, copying the last qubit's settings to new ones and trimming removed ones.
pub fn set_qubit_count(cfg: &mut Config, n: usize) {
    let n = n.max(1);
    let old = cfg.n_qubits();
    cfg.qubits = (1..=n).map(|i| format!("q{i}")).collect();
    let p = &mut cfg.parameters;
    // A new qubit sits 10 MHz above the last one, so couplings between them are not degenerate by default.
    if let Some(d) = &mut p.delta {
        while d.len() < n {
            let next = d.last().copied().unwrap_or(0.0) + 10e6;
            d.push(next);
        }
        d.truncate(n);
    }
    resize_opt(&mut p.sigma_delta, n, 0.0);
    resize(&mut p.omega_r_max, n, 40e6);
    resize(&mut p.sigma_omega_r_max, n, 0.0);
    resize(&mut p.pulse_duration, n, 2e-7);
    resize(&mut p.point_in_pulse, n, 100);
    resize(&mut p.wf_type, n, "cheb".into());
    resize(&mut p.wf_mode, n, ctrl_freeq::config::WaveformMode::Cart);
    resize(&mut p.amplitude_envelope, n, "gn".into());
    resize(&mut p.amplitude_order, n, 1);
    resize(&mut p.coverage, n, ctrl_freeq::config::Coverage::Single);
    resize(&mut p.sw, n, 5e6);
    resize(&mut p.pulse_offset, n, 0.0);
    resize(&mut p.pulse_bandwidth, n, 5e5);
    resize(&mut p.ratio_factor, n, 0.5);
    resize(&mut p.profile_order, n, 2);
    resize(&mut p.n_para, n, 16);
    resize_opt(&mut p.t1, n, 1e-3);
    resize_opt(&mut p.t2, n, 5e-4);
    resize_opt(&mut p.anharmonicities, n, -330e6);
    resize_opt(&mut p.stark_shift_coeffs, n, 0.0);
    let mut j = p.j.take().unwrap_or_default();
    resize_matrix(&mut j, n);
    // A new neighbour is coupled to the previous qubit like the last pair was, or at 10 MHz.
    for i in old.max(1)..n {
        let last = if i >= 2 { j[i - 2][i - 1] } else { 0.0 };
        j[i - 1][i] = if last != 0.0 { last } else { 10e6 };
    }
    p.j = Some(j);
    if let Some(zz) = &mut p.zz_crosstalk {
        resize_matrix(zz, n);
    }
    if n > 1 && p.coupling_type.is_none() {
        p.coupling_type = Some(coupling_types(cfg.hamiltonian_type.as_deref())[0].to_string());
    }
    if n > 1 && p.sigma_j.is_none() {
        p.sigma_j = Some(0.0);
    }
    for s in cfg.initial_states.iter_mut() {
        s.resize(n, "Z".into());
    }
    match &mut cfg.target_states {
        Targets::Axis(axes) => axes.iter_mut().for_each(|a| a.resize(n, "-Z".into())),
        Targets::PhiBeta { phi, beta } => {
            phi.iter_mut().for_each(|a| a.resize(n, "x".into()));
            beta.iter_mut().for_each(|b| b.resize(n, 90.0));
        }
        Targets::Gate(gates) => {
            let valid = gate_names(n);
            match default_gate(n) {
                Some(first) => gates
                    .iter_mut()
                    .filter(|g| !valid.contains(&g.as_str()))
                    .for_each(|g| *g = first.clone()),
                // No gates for this many qubits: fall back to state targets.
                None => cfg.target_states = Targets::Axis(vec![default_axis(n); cfg.initial_states.len()]),
            }
        }
    }
}

/// Switch the Hamiltonian model, filling in what the new one needs.  `None` is the Python package's original
/// spin-chain path.
pub fn set_model(cfg: &mut Config, model: Option<&str>) {
    let n = cfg.n_qubits();
    cfg.hamiltonian_type = model.map(str::to_string);
    let valid = coupling_types(model);
    let p = &mut cfg.parameters;
    if n > 1 && !p.coupling_type.as_deref().is_some_and(|c| valid.contains(&c)) {
        p.coupling_type = Some(valid[0].to_string());
    }
    if model == Some("duffing_transmon") {
        if p.anharmonicities.as_ref().is_none_or(|a| a.len() != n) {
            p.anharmonicities = Some(vec![-330e6; n]);
        }
        if cfg.optimization.dissipation_mode == Some(Dissipation::Dissipative) {
            cfg.optimization.dissipation_mode = Some(Dissipation::NonDissipative);
        }
    }
    if model != Some("superconducting") {
        p.zz_crosstalk = None;
        p.stark_shift_coeffs = None;
        if model != Some("duffing_transmon") {
            p.anharmonicities = None;
        }
    }
}

/// Turn T1/T2 relaxation on or off, with the Python example's times when none are set.
pub fn set_dissipative(cfg: &mut Config, on: bool) {
    let n = cfg.n_qubits();
    cfg.optimization.dissipation_mode = Some(if on {
        Dissipation::Dissipative
    } else {
        Dissipation::NonDissipative
    });
    if on {
        let p = &mut cfg.parameters;
        if p.t1.as_ref().is_none_or(|t| t.len() != n) {
            p.t1 = Some(vec![1e-3; n]);
        }
        if p.t2.as_ref().is_none_or(|t| t.len() != n) {
            p.t2 = Some(vec![5e-4; n]);
        }
    }
}

/// Change the target method, with a sensible target for each initial state.
pub fn set_target_method(cfg: &mut Config, method: TargetMethod) {
    let n = cfg.n_qubits();
    let k = cfg.initial_states.len();
    cfg.target_states = match method {
        TargetMethod::Axis => Targets::Axis(vec![default_axis(n); k]),
        TargetMethod::Gate => match default_gate(n) {
            Some(g) => Targets::Gate(vec![g; k]),
            None => return,
        },
        TargetMethod::Rotation => Targets::PhiBeta {
            phi: vec![vec!["x".to_string(); n]; k],
            beta: vec![vec![90.0; n]; k],
        },
    };
}

/// Add an initial state (all qubits along Z) with a matching target.
pub fn add_initial_state(cfg: &mut Config) {
    let n = cfg.n_qubits();
    cfg.initial_states.push(vec!["Z".to_string(); n]);
    match &mut cfg.target_states {
        Targets::Axis(a) => a.push(a.last().cloned().unwrap_or_else(|| default_axis(n))),
        Targets::Gate(g) => g.push(g.last().cloned().or_else(|| default_gate(n)).unwrap_or_default()),
        Targets::PhiBeta { phi, beta } => {
            phi.push(phi.last().cloned().unwrap_or_else(|| vec!["x".into(); n]));
            beta.push(beta.last().cloned().unwrap_or_else(|| vec![90.0; n]));
        }
    }
}

/// Remove initial state `i` and its target; the last one stays.
pub fn remove_initial_state(cfg: &mut Config, i: usize) {
    if cfg.initial_states.len() <= 1 || i >= cfg.initial_states.len() {
        return;
    }
    cfg.initial_states.remove(i);
    match &mut cfg.target_states {
        Targets::Axis(a) => {
            a.remove(i);
        }
        Targets::Gate(g) => {
            g.remove(i);
        }
        Targets::PhiBeta { phi, beta } => {
            phi.remove(i);
            beta.remove(i);
        }
    }
}

/// Set the pulse duration and point count on every qubit, as the Python GUI does.
pub fn set_pulse(cfg: &mut Config, duration_s: f64, points: usize) {
    let n = cfg.n_qubits();
    cfg.parameters.pulse_duration = vec![duration_s; n];
    cfg.parameters.point_in_pulse = vec![points; n];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::presets;

    #[test]
    fn edits_keep_every_preset_valid() {
        for (name, preset) in presets() {
            for n in [1, 2, 3] {
                let mut c = preset.clone();
                set_qubit_count(&mut c, n);
                assert!(c.validate().is_empty(), "{name} with {n} qubits: {:?}", c.validate());
                for model in [
                    None,
                    Some("spin_chain"),
                    Some("superconducting"),
                    Some("duffing_transmon"),
                ] {
                    let mut m = c.clone();
                    set_model(&mut m, model);
                    assert!(
                        m.validate().is_empty(),
                        "{name}, {n} qubits, {model:?}: {:?}",
                        m.validate()
                    );
                }
                for method in [TargetMethod::Axis, TargetMethod::Gate, TargetMethod::Rotation] {
                    let mut t = c.clone();
                    set_target_method(&mut t, method);
                    add_initial_state(&mut t);
                    assert!(
                        t.validate().is_empty(),
                        "{name}, {n} qubits, {method:?}: {:?}",
                        t.validate()
                    );
                    remove_initial_state(&mut t, 0);
                    assert!(t.validate().is_empty());
                }
                let mut d = c.clone();
                set_dissipative(&mut d, true);
                if d.hamiltonian_type.as_deref() != Some("duffing_transmon") {
                    assert!(
                        d.validate().is_empty(),
                        "{name}, {n} qubits, dissipative: {:?}",
                        d.validate()
                    );
                }
            }
        }
    }

    #[test]
    fn four_qubit_gates_fall_back_to_state_targets() {
        let mut c = ctrl_freeq::hamiltonian::default_config("spin_chain", 2).unwrap();
        set_qubit_count(&mut c, 4);
        assert!(matches!(c.target_states, Targets::Axis(_)));
        assert!(c.validate().is_empty(), "{:?}", c.validate());
    }
}
