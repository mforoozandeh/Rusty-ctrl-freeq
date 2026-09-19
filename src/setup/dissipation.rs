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
    use crate::autodiff::Step;

    #[test]
    fn t1_equal_t2_gives_damping_and_half_rate_dephasing() {
        let ops = collapse_operators(&[1.0], &[1.0]).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].get(0, 1), C::new(1.0, 0.0));
        assert!((ops[1].get(0, 0).re - 0.5).abs() < 1e-15);
        assert!((ops[1].get(1, 1).re + 0.5).abs() < 1e-15);
    }

    /// One step is the exact relaxation channel: populations relax at 1/T1 and coherences at 1/T2, also over steps
    /// longer than T1, where an explicit Euler step turns populations negative.
    #[test]
    fn a_step_is_the_exact_relaxation_channel() {
        let (t1, t2) = (1.0, 1.5);
        for dt in [0.01, 0.7, 2.0] {
            let ops = crate::autodiff::LindbladOps::new(&collapse_operators(&[t1], &[t2]).unwrap(), dt).unwrap();
            let coherence = C::new(0.2, 0.4);
            let rho = Mat::from_vec(
                2,
                2,
                vec![C::new(0.3, 0.0), coherence, coherence.conj(), C::new(0.7, 0.0)],
            )
            .unwrap();
            let out = ops.apply(&rho, Step::Full, false).unwrap();
            let excited = 0.7 * (-dt / t1).exp();
            assert!((out.get(1, 1).re - excited).abs() < 1e-14, "dt {dt}");
            assert!((out.get(0, 0).re - (1.0 - excited)).abs() < 1e-14, "dt {dt}");
            assert!((out.get(0, 1) - coherence * (-dt / t2).exp()).norm() < 1e-14, "dt {dt}");
        }
    }

    /// The Liouvillian `−i[H, ·] + D` as a `d²×d²` matrix on the row-major `vec(ρ)`.
    fn liouvillian(h: &CMat<f64>, collapse: &[CMat<f64>]) -> CMat<f64> {
        let d = h.rows;
        let i = C::new(0.0, 1.0);
        let mut out = CMat::<f64>::zeros(d * d, d * d);
        for from in 0..d * d {
            let mut e = CMat::<f64>::zeros(d, d);
            e.data[from] = C::new(1.0, 0.0);
            let mut col = h.matmul(&e).unwrap().sub(&e.matmul(h).unwrap()).unwrap().scale(-i);
            for l in collapse {
                let (l_dag,) = (l.adjoint(),);
                let l_dag_l = l_dag.matmul(l).unwrap();
                col = col.add(&l.matmul(&e).unwrap().matmul(&l_dag).unwrap()).unwrap();
                col.axpy_re(
                    -0.5,
                    &l_dag_l.matmul(&e).unwrap().add(&e.matmul(&l_dag_l).unwrap()).unwrap(),
                );
            }
            for (to, v) in col.data.iter().enumerate() {
                out.set(to, from, *v);
            }
        }
        out
    }

    /// `exp(total·L)` applied to `rho`, the exact evolution of a constant Hamiltonian with relaxation.
    fn exactly_propagated(h: &CMat<f64>, collapse: &[CMat<f64>], total: f64, rho: &CMat<f64>) -> CMat<f64> {
        let d = rho.rows;
        let map = crate::linalg::expm(&liouvillian(h, collapse).scale_re(total)).unwrap();
        let mut out = CMat::<f64>::zeros(d, d);
        for to in 0..d * d {
            out.data[to] = (0..d * d).fold(C::new(0.0, 0.0), |acc, from| acc + map.get(to, from) * rho.data[from]);
        }
        out
    }

    /// Alternating the unitary step with half a dissipation step at each end is second order: halving the step
    /// quarters the error.  The objective and the plots propagate exactly this way.
    #[test]
    fn strang_splitting_converges_at_second_order() {
        let (t1, t2, total) = (1.0, 1.5, 0.4);
        let collapse = collapse_operators(&[t1], &[t2]).unwrap();
        // A drive about x, which does not commute with the damping.
        let h = paulis()[0].scale_re(1.7);
        let half = C::new(0.5, 0.0);
        let rho0 = Mat::from_vec(2, 2, vec![half, half, half, half]).unwrap();
        let exact = exactly_propagated(&h, &collapse, total, &rho0);
        let errors: Vec<f64> = [10usize, 20, 40]
            .iter()
            .map(|&steps| {
                let dt = total / steps as f64;
                let ops = crate::autodiff::LindbladOps::new(&collapse, dt).unwrap();
                let u = crate::linalg::expm_mi_dt(&h, dt).unwrap();
                let mut rho = rho0.clone();
                for _ in 0..steps {
                    let entering = ops.apply(&rho, Step::Half, false).unwrap();
                    let turned = u.matmul(&entering).unwrap().matmul(&u.adjoint()).unwrap();
                    rho = ops.apply(&turned, Step::Half, false).unwrap();
                }
                rho.max_abs_diff(&exact)
            })
            .collect();
        let ratios = [errors[0] / errors[1], errors[1] / errors[2]];
        assert!(
            ratios.iter().all(|r| (3.2..4.8).contains(r)),
            "errors {errors:?} fall by {ratios:?}, not about 4"
        );
    }

    /// A step backwards in time is not a channel: it would take population below zero.
    #[test]
    fn the_channel_refuses_a_negative_or_infinite_step() {
        let ops = collapse_operators(&[1.0], &[2.0]).unwrap();
        assert!(crate::autodiff::LindbladOps::new(&ops, -0.1).is_err());
        assert!(crate::autodiff::LindbladOps::new(&ops, f64::NAN).is_err());
        assert!(crate::autodiff::LindbladOps::new(&ops, f64::INFINITY).is_err());
        assert!(crate::autodiff::LindbladOps::new(&ops, 0.1).is_ok());
    }

    #[test]
    fn t2_at_twice_t1_has_no_dephasing() {
        assert_eq!(collapse_operators(&[1.0], &[2.0]).unwrap().len(), 1);
        assert!(collapse_operators(&[1.0], &[2.5]).is_err());
    }
}
