use rand::SeedableRng;

use super::*;
use crate::config::WaveformMode::{Cart, Polar, PolarPhase};

fn rng() -> Rng {
    Rng::seed_from_u64(1)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

#[test]
fn polynomials_match_their_closed_forms() {
    let x = [-0.9, -0.3, 0.0, 0.45, 1.0];
    let cheb = basis("cheb").unwrap().matrix(&x, 6, &mut rng());
    let leg = basis("leg").unwrap().matrix(&x, 4, &mut rng());
    let herm = basis("hermite").unwrap().matrix(&x, 4, &mut rng());
    let gegen = basis("gegen").unwrap().matrix(&x, 3, &mut rng());
    let poly = basis("poly").unwrap().matrix(&x, 4, &mut rng());
    let chirp = basis("chirp").unwrap().matrix(&x, 3, &mut rng());
    for (r, &v) in x.iter().enumerate() {
        for i in 0..6 {
            assert!(close(cheb.get(r, i), (i as f64 * v.acos()).cos()), "T{i}({v})");
        }
        assert!(close(leg.get(r, 2), 0.5 * (3.0 * v * v - 1.0)));
        assert!(close(leg.get(r, 3), 0.5 * (5.0 * v * v * v - 3.0 * v)));
        assert!(close(herm.get(r, 2), v * v - 1.0));
        assert!(close(herm.get(r, 3), v * v * v - 3.0 * v));
        // C_2^(1/2) is P_2.
        assert!(close(gegen.get(r, 2), 0.5 * (3.0 * v * v - 1.0)));
        assert!(close(poly.get(r, 3), v * v * v));
        assert!(close(
            chirp.get(r, 2),
            (2.0 * std::f64::consts::PI * 2.0 * 0.25 * v * v).cos()
        ));
    }
}

#[test]
fn fourier_columns_are_constant_then_cosine_sine_pairs() {
    let x = [0.1, 0.7];
    let f = basis("fou").unwrap().matrix(&x, 5, &mut rng());
    let tau = 2.0 * std::f64::consts::PI;
    for (r, &v) in x.iter().enumerate() {
        assert_eq!(f.get(r, 0), 1.0);
        assert!(close(f.get(r, 1), (tau * v).cos()));
        assert!(close(f.get(r, 2), (tau * v).sin()));
        assert!(close(f.get(r, 3), (tau * 2.0 * v).cos()));
        assert!(close(f.get(r, 4), (tau * 2.0 * v).sin()));
    }
}

/// The Python `update_n_para` table and the matching column counts.
#[test]
fn parameter_and_column_counts_follow_the_python_rules() {
    let cheb = basis("cheb").unwrap();
    let fou = basis("fou").unwrap();
    assert_eq!((cheb.columns(16, Cart), cheb.parameter_count(16, Cart)), (8, 16));
    assert_eq!((cheb.columns(16, Polar), cheb.parameter_count(16, Polar)), (8, 16));
    assert_eq!(
        (cheb.columns(16, PolarPhase), cheb.parameter_count(16, PolarPhase)),
        (16, 17)
    );
    // Fourier cart/polar: n = (16 − 2)/4 = 3 → 7 columns, 14 parameters.
    assert_eq!((fou.columns(16, Cart), fou.parameter_count(16, Cart)), (7, 14));
    // Fourier polar_phase: n = (16 − 1)/2 = 7 → 15 columns, 16 parameters.
    assert_eq!(
        (fou.columns(16, PolarPhase), fou.parameter_count(16, PolarPhase)),
        (15, 16)
    );
    assert!(cheb.check_n_para(15, Cart).is_err());
    assert!(cheb.check_n_para(15, PolarPhase).is_ok());
    assert!(fou.check_n_para(1, Cart).is_err());
}

#[test]
fn initial_parameters_have_the_parameter_count() {
    let mut r = rng();
    for name in basis_names() {
        let b = basis(name).unwrap();
        for mode in WaveformMode::ALL {
            for n_para in [6, 9, 16] {
                if b.check_n_para(n_para, mode).is_err() {
                    continue;
                }
                let raw = raw_coefficients(n_para, &mut r);
                assert!(raw.iter().all(|v| (-1.0..1.0).contains(v)));
                let x = initial_params(b.as_ref(), &raw, mode);
                assert_eq!(x.len(), b.parameter_count(n_para, mode), "{name} {mode:?} {n_para}");
                if mode == PolarPhase {
                    assert_eq!(x[x.len() - 1], x[x.len() - 2]);
                }
            }
        }
    }
}

#[test]
fn qubit_bases_are_orthonormal_where_they_should_be() {
    let n_pulse = 40;
    let x = linspace(-1.0, 1.0, n_pulse);
    let env = envelope("gn", &x, 1).unwrap();
    let is_orthonormal = |q: &RMat<f64>| {
        let qtq = q.transpose().matmul(q).unwrap();
        (0..q.cols).all(|i| (0..q.cols).all(|j| (qtq.get(i, j) - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12))
    };
    for mode in WaveformMode::ALL {
        let qb = qubit_basis(basis("leg").unwrap().as_ref(), &env, 8, mode, n_pulse, &mut rng()).unwrap();
        assert!(is_orthonormal(&qb.q[1]), "{mode:?}");
        match mode {
            PolarPhase => assert!((0..n_pulse).all(|r| qb.q[0].get(r, 3) == env[r])),
            _ => assert!(is_orthonormal(&qb.q[0])),
        }
    }
}

#[test]
fn envelopes_span_epsilon_to_one_and_peak_in_the_middle() {
    let x = linspace(-1.0, 1.0, 101);
    for name in ["gn", "hs"] {
        for order in [1, 2, 3] {
            let e = envelope(name, &x, order).unwrap();
            let max = e.iter().copied().fold(f64::MIN, f64::max);
            let min = e.iter().copied().fold(f64::MAX, f64::min);
            assert!(
                (max - 1.0).abs() < 1e-15 && (min - f64::EPSILON).abs() < 1e-15,
                "{name}{order}"
            );
            if order % 2 == 0 || name == "gn" {
                assert_eq!(e[50], max, "{name}{order} peaks at x = 0");
            }
        }
    }
    let q = envelope("quad", &x, 1).unwrap();
    assert_eq!(q[50], 1.0 + f64::EPSILON);
    assert!(envelope("nope", &x, 1).is_err());
}

#[test]
fn linspace_matches_numpy() {
    let v = linspace(-1.0, 1.0, 5);
    assert_eq!(v, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    assert_eq!(linspace(0.0, 1.0, 1), vec![0.0]);
}

/// Two points sit at x = ±1, where a symmetric envelope has one value: there is nothing to taper, so it is flat.
#[test]
fn a_constant_envelope_is_flat() {
    for name in ["gn", "hs"] {
        assert_eq!(envelope(name, &[-1.0, 1.0], 1).unwrap(), vec![1.0, 1.0], "{name}");
    }
}

/// Chirps are even in x, so ten symmetric points hold only five independent ones.  QR would still return eight
/// orthonormal columns, the missing three arbitrary and not even symmetric.
#[test]
fn linearly_dependent_bases_are_refused() {
    let n_pulse = 10;
    let env = envelope("gn", &linspace(-1.0, 1.0, n_pulse), 1).unwrap();
    for mode in WaveformMode::ALL {
        // Eight columns in every mode.
        let n_para = if mode == PolarPhase { 8 } else { 16 };
        let qb = qubit_basis(
            basis("chirp").unwrap().as_ref(),
            &env,
            n_para,
            mode,
            n_pulse,
            &mut rng(),
        );
        assert!(
            matches!(qb, Err(Error::Config(ref m)) if m.contains("independent")),
            "{mode:?}: {qb:?}"
        );
    }
    let many = 50;
    let env = envelope("gn", &linspace(-1.0, 1.0, many), 1).unwrap();
    assert!(qubit_basis(basis("chirp").unwrap().as_ref(), &env, 16, Cart, many, &mut rng()).is_ok());
}
