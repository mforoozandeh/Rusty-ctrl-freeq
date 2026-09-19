//! The example configurations bundled with the Python package, as JSON.
//!
//! They are copied unchanged except that the Python examples naming `qiskit-cobyla` use `cobyla` here.

/// `(file stem, JSON)` for every bundled example.
pub fn examples() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "single_qubit_parameters",
            include_str!("examples/single_qubit_parameters.json"),
        ),
        (
            "single_qubit_parameters_multiple_initial_targ",
            include_str!("examples/single_qubit_parameters_multiple_initial_targ.json"),
        ),
        (
            "single_qubit_parameters_polar_phase",
            include_str!("examples/single_qubit_parameters_polar_phase.json"),
        ),
        (
            "single_qubit_dissipative",
            include_str!("examples/single_qubit_dissipative.json"),
        ),
        (
            "two_qubit_parameters",
            include_str!("examples/two_qubit_parameters.json"),
        ),
        (
            "two_qubit_parameters_multiple_initial_targ",
            include_str!("examples/two_qubit_parameters_multiple_initial_targ.json"),
        ),
        (
            "two_qubit_parameters_polar_phase",
            include_str!("examples/two_qubit_parameters_polar_phase.json"),
        ),
        (
            "four_qubit_parameters_polar_phase",
            include_str!("examples/four_qubit_parameters_polar_phase.json"),
        ),
    ]
}
