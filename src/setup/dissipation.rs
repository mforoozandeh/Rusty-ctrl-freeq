//! Lindblad collapse operators for T1 relaxation and pure dephasing.

use super::states::{embed, paulis};
use crate::autodiff::C;
use crate::error::{Error, Result};
use crate::linalg::{CMat, Mat};

/// The collapse operators for per-qubit `t1` and `t2` in seconds.
///
/// For qubit `q`: amplitude damping `sqrt(γ₁)·σ⁻_q` with `γ₁ = 1/T1`, and pure dephasing `sqrt(γφ/2)·σz_q` with
/// `γφ = 1/T2 − 1/(2·T1)` when positive.  The dephasing operator makes the Lindblad dissipator decay the
/// off-diagonal elements at rate `γφ`, the physical T2 convention.
pub fn collapse_operators(t1: &[f64], t2: &[f64]) -> Result<Vec<CMat<f64>>> {
    let n = t1.len();
    if t2.len() != n {
        return Err(Error::Config("T1 and T2 need one entry per qubit".into()));
    }
    let o = C::new(0.0, 0.0);
    let sigma_minus = Mat::from_vec(2, 2, vec![o, C::new(1.0, 0.0), o, o])?;
    let sigma_z = paulis()[2].clone();
    let mut ops = Vec::new();
    for q in 0..n {
        if !(t1[q] > 0.0 && t2[q] > 0.0) || t2[q] > 2.0 * t1[q] {
            return Err(Error::Config(format!(
                "qubit {}: T1 and T2 must be positive with T2 at most 2·T1",
                q + 1
            )));
        }
        let gamma1 = 1.0 / t1[q];
        ops.push(embed(&sigma_minus, q, n).scale_re(gamma1.sqrt()));
        let gamma_phi = 1.0 / t2[q] - 1.0 / (2.0 * t1[q]);
        if gamma_phi > 0.0 {
            ops.push(embed(&sigma_z, q, n).scale_re((gamma_phi / 2.0).sqrt()));
        }
    }
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t1_equal_t2_gives_damping_and_half_rate_dephasing() {
        let ops = collapse_operators(&[1.0], &[1.0]).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].get(0, 1), C::new(1.0, 0.0));
        assert!((ops[1].get(0, 0).re - 0.5).abs() < 1e-15);
        assert!((ops[1].get(1, 1).re + 0.5).abs() < 1e-15);
    }

    #[test]
    fn t2_at_twice_t1_has_no_dephasing() {
        assert_eq!(collapse_operators(&[1.0], &[2.0]).unwrap().len(), 1);
        assert!(collapse_operators(&[1.0], &[2.5]).is_err());
    }
}
