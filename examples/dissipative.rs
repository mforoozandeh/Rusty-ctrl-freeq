//! An inversion under T1 and T2 relaxation, in Liouville space with Lindblad evolution.
//!
//! `cargo run --release --example dissipative`

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::optim::ConsoleSink;

fn main() -> ctrl_freeq::Result<()> {
    let json = examples()
        .iter()
        .find(|e| e.0 == "single_qubit_dissipative")
        .expect("bundled")
        .1;
    let mut cfg = Config::from_json(json)?;
    cfg.seed = Some(7);
    println!(
        "single qubit with T1 = 1 ms, T2 = 0.5 ms, {}",
        cfg.optimization.algorithm
    );
    let r = ctrl_freeq::run(&cfg, &mut ConsoleSink::new(20))?;
    println!(
        "fidelity {:.6} after {} iterations in {:.2} s: {}",
        r.fidelity,
        r.iterations,
        r.elapsed_s,
        r.exit.message()
    );
    if r.fidelity - r.penalty < cfg.optimization.targ_fid {
        std::process::exit(1);
    }
    Ok(())
}
