//! Parity with the Python package on deterministic problems.
//!
//! The fixtures in `tests/fixtures` come from `tools/python_reference/generate.py`, which runs the Python ctrl-freeq
//! 0.3.0 on each configuration at a fixed parameter vector.  Every quantity the Rust setup and objective produce is
//! compared: basis matrices, drift Hamiltonians, initial states, targets, modulation, the cost and its gradient.

use std::path::Path;

use ctrl_freeq::autodiff::C;
use ctrl_freeq::config::{Config, Space};
use ctrl_freeq::linalg::CMat;
use ctrl_freeq::objective::CostModel;
use ctrl_freeq::setup::Problem;
use serde_json::Value;

fn fixtures() -> Vec<(String, Value)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut out: Vec<(String, Value)> = std::fs::read_dir(&dir)
        .expect("tests/fixtures exists")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .map(|e| {
            let name = e.path().file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(e.path()).unwrap();
            (name, serde_json::from_str(&text).unwrap())
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(out.len() >= 9, "expected the generated fixtures");
    out
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn complex(v: &Value) -> C<f64> {
    let a = v.as_array().unwrap();
    C::new(a[0].as_f64().unwrap(), a[1].as_f64().unwrap())
}

/// A complex matrix (or a column vector) from nested `[re, im]` pairs.
fn cmat(v: &Value) -> CMat<f64> {
    let rows = v.as_array().unwrap();
    if rows[0].as_array().unwrap()[0].is_number() {
        CMat::column(rows.iter().map(complex).collect())
    } else {
        let r = rows.len();
        let c = rows[0].as_array().unwrap().len();
        CMat::from_fn(r, c, |i, j| complex(&rows[i][j]))
    }
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-3)
}

#[test]
fn setup_and_objective_match_python() {
    for (name, fx) in fixtures() {
        let cfg = Config::from_json(&fx["config"].to_string()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let problem = Problem::build_with_seed(&cfg, 0).unwrap();
        let liouville = problem.space == Space::Liouville;

        // Basis matrices, entry for entry.  Chirps are nearly linearly dependent, so their QR carries rounding
        // noise of a few 1e-12.
        for (q, qb) in problem.qubits.iter().enumerate() {
            for k in 0..2 {
                let want = &fx["mat"][q][k];
                let rows = want.as_array().unwrap();
                assert_eq!(rows.len(), qb.q[k].rows, "{name}: basis rows");
                for (r, row) in rows.iter().enumerate() {
                    for (c, v) in f64s(row).iter().enumerate() {
                        let got = qb.q[k].get(r, c);
                        assert!(
                            (got - v).abs() < 1e-10,
                            "{name}: qubit {q} matrix {k} ({r},{c}): {got} vs {v}"
                        );
                    }
                }
            }
        }

        // Batch: drift, initial state and target per element.
        let h0 = fx["h0"].as_array().unwrap();
        assert_eq!(h0.len(), problem.batch.len(), "{name}: batch size");
        for (i, e) in problem.batch.iter().enumerate() {
            let scale = e.h0.norm1();
            assert!(e.h0.max_abs_diff(&cmat(&h0[i])) <= 1e-12 * scale, "{name}: H0[{i}]");
            assert!(
                e.initial.max_abs_diff(&cmat(&fx["initials"][i])) < 1e-12,
                "{name}: initial[{i}]"
            );
            assert!(
                e.target.max_abs_diff(&cmat(&fx["targets"][i])) < 1e-12,
                "{name}: target[{i}]"
            );
        }

        // Modulation.
        let modulation = fx["modulation"].as_array().unwrap();
        for (t, row) in modulation.iter().enumerate() {
            for (q, v) in row.as_array().unwrap().iter().enumerate() {
                assert!(
                    (problem.modulation.get(t, q) - complex(v)).norm() < 1e-12,
                    "{name}: modulation"
                );
            }
        }

        // Cost and gradient.  Python's Liouville fidelity goes through two eigendecompositions of rank-deficient
        // matrices, which costs it a few digits; Rust's Re Tr(ρσ) is exact for the pure targets.
        let model = CostModel::new(problem);
        let x = f64s(&fx["x"]);
        let (e, g) = model.value_and_gradient(&x).unwrap();
        let rel = if liouville { 1e-7 } else { 1e-10 };
        let cost = fx["cost"].as_f64().unwrap();
        assert!(close(e.cost, cost, rel), "{name}: cost {} vs {cost}", e.cost);
        assert!(
            close(e.fidelity, fx["fidelity"].as_f64().unwrap(), rel),
            "{name}: fidelity"
        );
        assert!(
            close(e.penalty, fx["penalty"].as_f64().unwrap(), rel),
            "{name}: penalty"
        );
        if let Some(want) = fx["gradient"].as_array() {
            let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let grel = if liouville { 1e-5 } else { 1e-8 };
            for (i, w) in want.iter().enumerate() {
                let w = w.as_f64().unwrap();
                assert!(
                    (g[i] - w).abs() <= grel * gmax.max(1e-3),
                    "{name}: gradient[{i}] {} vs {w}",
                    g[i]
                );
            }
        }
    }
}
