//! Results do not depend on the number of threads: value, gradient and Hessian are bit-for-bit identical on one
//! thread and on several.
#![cfg(feature = "parallel")]

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::objective::CostModel;
use ctrl_freeq::setup::Problem;

fn on_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn one_thread_and_four_give_identical_bits() {
    let mut cfg = Config::from_json(examples().iter().find(|e| e.0 == "single_qubit_parameters").unwrap().1).unwrap();
    cfg.parameters.point_in_pulse = vec![30];
    cfg.parameters.n_para = vec![6];
    cfg.optimization.h0_snapshots = 37;
    let model = CostModel::new(Problem::build_with_seed(&cfg, 5).unwrap());
    let x = model.problem().x0.clone();
    let run = || {
        let (e, g) = model.value_and_gradient(&x).unwrap();
        let h = model.hessian_matrix(&x).unwrap();
        (
            e.cost.to_bits(),
            g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        )
    };
    let one = on_threads(1, run);
    let four = on_threads(4, run);
    assert_eq!(one, four);
}
