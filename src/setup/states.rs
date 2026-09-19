//! Spin operators, product states, gates and rotations.
//!
//! Qubit 0 is the most significant factor of every Kronecker product, as in the Python package.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::autodiff::C;
use crate::error::{Error, Result};
use crate::linalg::{CMat, Mat, expm};

fn c(re: f64, im: f64) -> C<f64> {
    C::new(re, im)
}

fn mat(n: usize, entries: &[C<f64>]) -> CMat<f64> {
    Mat::from_vec(n, n, entries.to_vec()).expect("literal matrices have the right size")
}

/// The Pauli matrices `[σx, σy, σz]`.
pub fn paulis() -> [CMat<f64>; 3] {
    let (o, l, i) = (c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0));
    [mat(2, &[o, l, l, o]), mat(2, &[o, -i, i, o]), mat(2, &[l, o, o, -l])]
}

/// `op` acting on qubit `qubit` of `n`, identity on the rest.
pub fn embed(op: &CMat<f64>, qubit: usize, n: usize) -> CMat<f64> {
    let id = CMat::<f64>::identity(op.rows);
    let mut out = CMat::<f64>::identity(1);
    for q in 0..n {
        out = out.kron(if q == qubit { op } else { &id });
    }
    out
}

/// Per qubit, the spin operators `[X, Y, Z] = [σx, σy, σz]/2` in the `2ⁿ`-dimensional space.
pub fn spin_ops(n: usize) -> Vec<[CMat<f64>; 3]> {
    let p = paulis();
    (0..n)
        .map(|q| [0, 1, 2].map(|k| embed(&p[k].scale_re(0.5), q, n)))
        .collect()
}

/// Per qubit, the Pauli observables `[σx, σy, σz]` in the `2ⁿ`-dimensional space.
pub fn pauli_ops(n: usize) -> Vec<[CMat<f64>; 3]> {
    let p = paulis();
    (0..n).map(|q| [0, 1, 2].map(|k| embed(&p[k], q, n))).collect()
}

/// The single-qubit state along `axis` (`Z` = |0⟩, `-Z` = |1⟩, `X`, `-X`, `Y`, `-Y`).
fn axis_state(axis: &str) -> Result<[C<f64>; 2]> {
    let s = FRAC_1_SQRT_2;
    Ok(match axis {
        "Z" => [c(1.0, 0.0), c(0.0, 0.0)],
        "-Z" => [c(0.0, 0.0), c(1.0, 0.0)],
        "X" => [c(s, 0.0), c(s, 0.0)],
        "-X" => [c(s, 0.0), c(-s, 0.0)],
        "Y" => [c(s, 0.0), c(0.0, s)],
        "-Y" => [c(s, 0.0), c(0.0, -s)],
        _ => return Err(Error::Config(format!("unknown state axis \"{axis}\""))),
    })
}

/// The product state vector for one axis per qubit.
pub fn product_state(axes: &[String]) -> Result<CMat<f64>> {
    let mut out = CMat::<f64>::identity(1);
    for a in axes {
        let s = axis_state(a)?;
        out = out.kron(&Mat::column(s.to_vec()));
    }
    Ok(out)
}

/// The product density matrix `⊗ ½(I ± σ)` for one axis per qubit.
pub fn product_density(axes: &[String]) -> Result<CMat<f64>> {
    let psi = product_state(axes)?;
    psi.matmul(&psi.adjoint())
}

/// Gate names available for `n` qubits, in the order the interface lists them.
pub fn gate_names(n: usize) -> &'static [&'static str] {
    match n {
        1 => &["X", "Y", "Z", "H", "S", "T"],
        2 => &["CNOT", "CX", "CZ", "SWAP", "iSWAP", "√iSWAP", "ECR"],
        3 => &["Toff"],
        _ => &[],
    }
}

/// The canonical name of gate `name`, which some gates have more than one of: `CX` is `CNOT`.
///
/// Targets name one gate when their canonical names agree, whatever the configuration spelled; see
/// [`Targets::single_gate`](crate::config::Targets::single_gate).
pub fn canonical_gate(name: &str) -> &str {
    match name {
        "CX" => "CNOT",
        other => other,
    }
}

/// The unitary for gate `name` on `n` qubits.
pub fn gate(name: &str, n: usize) -> Result<CMat<f64>> {
    let (o, l, i) = (c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0));
    let s = FRAC_1_SQRT_2;
    let (sr, si) = (c(s, 0.0), c(0.0, s));
    let g = match (n, name) {
        (1, "X") => mat(2, &[o, l, l, o]),
        (1, "Y") => mat(2, &[o, -i, i, o]),
        (1, "Z") => mat(2, &[l, o, o, -l]),
        (1, "H") => mat(2, &[sr, sr, sr, -sr]),
        (1, "S") => mat(2, &[l, o, o, i]),
        (1, "T") => mat(2, &[l, o, o, C::from_polar(1.0, std::f64::consts::FRAC_PI_4)]),
        (2, "CNOT" | "CX") => mat(4, &[l, o, o, o, o, l, o, o, o, o, o, l, o, o, l, o]),
        (2, "CZ") => mat(4, &[l, o, o, o, o, l, o, o, o, o, l, o, o, o, o, -l]),
        (2, "SWAP") => mat(4, &[l, o, o, o, o, o, l, o, o, l, o, o, o, o, o, l]),
        (2, "iSWAP") => mat(4, &[l, o, o, o, o, o, i, o, o, i, o, o, o, o, o, l]),
        (2, "√iSWAP") => mat(4, &[l, o, o, o, o, sr, si, o, o, si, sr, o, o, o, o, l]),
        (2, "ECR") => mat(4, &[o, o, sr, si, o, o, si, sr, sr, -si, o, o, -si, sr, o, o]),
        (3, "Toff") => {
            let mut m = CMat::<f64>::identity(8);
            m.set(6, 6, o);
            m.set(7, 7, o);
            m.set(6, 7, l);
            m.set(7, 6, l);
            m
        }
        _ => {
            return Err(Error::Config(format!(
                "gate \"{name}\" is not available for {n} qubits"
            )));
        }
    };
    Ok(g)
}

/// `Π_q exp(∓i·β_q·Op_q)` over the qubits in order, with `Op` the spin operator for each axis and the sign `−`
/// for `x`, `y`, `z` and `+` for `-x`, `-y`, `-z`.  `beta` is in radians.
pub fn rotation(axes: &[String], beta: &[f64], n: usize) -> Result<CMat<f64>> {
    let ops = spin_ops(n);
    let mut u = CMat::<f64>::identity(1 << n);
    for (q, (axis, &b)) in axes.iter().zip(beta).enumerate() {
        let (k, sign) = match axis.as_str() {
            "x" => (0, -1.0),
            "-x" => (0, 1.0),
            "y" => (1, -1.0),
            "-y" => (1, 1.0),
            "z" => (2, -1.0),
            "-z" => (2, 1.0),
            _ => return Err(Error::Config(format!("unknown rotation axis \"{axis}\""))),
        };
        let uq = expm(&ops[q][k].scale(c(0.0, sign * b)))?;
        u = u.matmul(&uq)?;
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn axis_states_are_eigenstates_of_the_paulis() {
        let p = paulis();
        for (axis, k, eig) in [
            ("Z", 2, 1.0),
            ("-Z", 2, -1.0),
            ("X", 0, 1.0),
            ("-X", 0, -1.0),
            ("Y", 1, 1.0),
            ("-Y", 1, -1.0),
        ] {
            let psi = product_state(&strs(&[axis])).unwrap();
            let got = p[k].matmul(&psi).unwrap();
            assert!(got.max_abs_diff(&psi.scale_re(eig)) < 1e-15, "{axis}");
        }
    }

    #[test]
    fn densities_are_pure_with_unit_trace() {
        let rho = product_density(&strs(&["X", "-Y"])).unwrap();
        assert!((rho.trace() - c(1.0, 0.0)).norm() < 1e-15);
        let rho2 = rho.matmul(&rho).unwrap();
        assert!((rho2.trace() - c(1.0, 0.0)).norm() < 1e-15);
        assert!(rho.is_hermitian(1e-15));
    }

    #[test]
    fn gates_are_unitary() {
        for n in 1..=3 {
            for name in gate_names(n) {
                let g = gate(name, n).unwrap();
                let gg = g.matmul(&g.adjoint()).unwrap();
                assert!(gg.max_abs_diff(&CMat::identity(g.rows)) < 1e-15, "{name}");
            }
        }
        assert!(gate("CNOT", 1).is_err());
        assert!(gate("Toff", 4).is_err());
    }

    #[test]
    fn cnot_flips_the_second_qubit_when_the_first_is_one() {
        let g = gate("CNOT", 2).unwrap();
        let psi = product_state(&strs(&["-Z", "Z"])).unwrap(); // |10>
        let out = g.matmul(&psi).unwrap();
        let want = product_state(&strs(&["-Z", "-Z"])).unwrap(); // |11>
        assert!(out.max_abs_diff(&want) < 1e-15);
    }

    #[test]
    fn a_180_degree_x_rotation_inverts_z() {
        let u = rotation(&strs(&["x"]), &[std::f64::consts::PI], 1).unwrap();
        let psi = product_state(&strs(&["Z"])).unwrap();
        let out = u.matmul(&psi).unwrap();
        // |0> -> -i|1>
        assert!((out.get(1, 0).norm() - 1.0).abs() < 1e-14);
        assert!(out.get(0, 0).norm() < 1e-14);
    }

    #[test]
    fn spin_ops_have_the_half_factor_and_qubit_zero_is_most_significant() {
        let ops = spin_ops(2);
        // Z on qubit 0 = diag(1, 1, -1, -1)/2.
        let z0 = &ops[0][2];
        assert_eq!(z0.get(0, 0), c(0.5, 0.0));
        assert_eq!(z0.get(1, 1), c(0.5, 0.0));
        assert_eq!(z0.get(2, 2), c(-0.5, 0.0));
    }
}
