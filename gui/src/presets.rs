//! Ready-made configurations: the Python package's examples, a Toffoli gate and each model's defaults.

use ctrl_freeq::config::{Config, Coverage, Targets, examples};
use ctrl_freeq::hamiltonian::{default_config, model_names};

use crate::edit::set_pulse;

/// `(label, configuration)` for every preset, in menu order.
pub fn presets() -> Vec<(String, Config)> {
    let mut out: Vec<(String, Config)> = examples()
        .iter()
        .filter(|(name, _)| *name != "four_qubit_parameters_polar_phase")
        .filter_map(|(name, json)| Config::from_json(json).ok().map(|c| (name.replace('_', " "), c)))
        .collect();
    if let Some(c) = toffoli() {
        out.push(("three qubit toffoli".into(), c));
    }
    for model in model_names() {
        for n in [1, 2] {
            if let Ok(c) = default_config(model, n) {
                let qubits = if n == 1 { "1 qubit" } else { "2 qubits" };
                out.push((format!("{} default, {qubits}", model.replace('_', " ")), c));
            }
        }
    }
    out
}

/// A Toffoli on a chain of three XY-coupled spins, from |110⟩ so the target flips.  Pulses of 400 ns with 32
/// parameters per qubit reach the 0.999 target in under a hundred L-BFGS iterations; the spin-chain default's
/// 200 ns and 16 parameters stall well short of it.
fn toffoli() -> Option<Config> {
    let mut c = default_config("spin_chain", 3).ok()?;
    c.initial_states = vec![vec!["-Z".into(), "-Z".into(), "Z".into()]];
    c.target_states = Targets::Gate(vec!["Toff".into()]);
    c.parameters.coverage = vec![Coverage::Single; 3];
    c.parameters.n_para = vec![32; 3];
    set_pulse(&mut c, 400e-9, 200);
    Some(c)
}

/// What the interface shows on first start.
pub fn initial() -> Config {
    presets()
        .into_iter()
        .find(|(name, _)| name == "single qubit parameters")
        .map(|(_, c)| c)
        .unwrap_or_else(|| default_config("spin_chain", 1).expect("the spin chain default is valid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_menu_has_a_valid_toffoli_and_no_four_qubit_example() {
        let all = presets();
        let (_, toff) = all
            .iter()
            .find(|(name, _)| name == "three qubit toffoli")
            .expect("listed");
        assert!(toff.validate().is_empty(), "{:?}", toff.validate());
        assert_eq!(toff.target_states.single_gate(), Some("Toff"));
        assert!(all.iter().all(|(_, c)| c.n_qubits() < 4));
    }
}
