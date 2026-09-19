//! Ready-made configurations: the Python package's examples and each model's defaults.

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::hamiltonian::{default_config, model_names};

/// `(label, configuration)` for every preset, in menu order.
pub fn presets() -> Vec<(String, Config)> {
    let mut out: Vec<(String, Config)> = examples()
        .iter()
        .filter_map(|(name, json)| Config::from_json(json).ok().map(|c| (name.replace('_', " "), c)))
        .collect();
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

/// What the interface shows on first start.
pub fn initial() -> Config {
    presets()
        .into_iter()
        .find(|(name, _)| name == "single qubit parameters")
        .map(|(_, c)| c)
        .unwrap_or_else(|| default_config("spin_chain", 1).expect("the spin chain default is valid"))
}
