//! Time the objective, its gradient and its Hessian on one thread and on all of them, and check that both give the
//! same bits.
//!
//! `cargo run --release --example timing`

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::objective::CostModel;
use ctrl_freeq::setup::Problem;
use ctrl_freeq::time::Instant;

fn mean_seconds(reps: usize, mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() -> ctrl_freeq::Result<()> {
    let json = examples()
        .iter()
        .find(|e| e.0 == "two_qubit_parameters")
        .expect("bundled")
        .1;
    let mut cfg = Config::from_json(json)?;
    cfg.parameters.coverage = vec![ctrl_freeq::config::Coverage::Broadband; 2];
    cfg.optimization.h0_snapshots = 64;
    let all = ctrl_freeq::run::available_threads();
    let mut outputs = Vec::new();
    println!(
        "two qubits, 64 snapshots x 100 steps, {} parameters",
        Problem::build_with_seed(&cfg, 1)?.n_params()
    );
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}",
        "threads", "value", "gradient", "hessian"
    );
    for threads in [1, all] {
        let model = CostModel::new(Problem::build_with_seed(&cfg, 1)?).with_threads(threads)?;
        let x = model.problem().x0.clone();
        let v = mean_seconds(10, || drop(model.value(&x)));
        let g = mean_seconds(5, || drop(model.value_and_gradient(&x)));
        let h = mean_seconds(1, || drop(model.hessian_matrix(&x)));
        println!("{threads:>8}  {v:>9.4}s  {g:>9.4}s  {h:>9.4}s");
        let (e, grad) = model.value_and_gradient(&x)?;
        let hess = model.hessian_matrix(&x)?;
        outputs.push((
            e.cost.to_bits(),
            grad.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            hess.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ));
    }
    if outputs[0] != outputs[1] {
        eprintln!("results differ between 1 and {all} threads");
        std::process::exit(1);
    }
    println!("identical results on 1 and {all} threads");
    Ok(())
}
