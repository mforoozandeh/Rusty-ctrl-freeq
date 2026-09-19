//! A broadband inversion: one qubit, Z to −Z, robust across a 5 MHz band of offsets.
//!
//! `cargo run --release --example single_qubit [algorithm]`

use ctrl_freeq::config::{Config, examples};
use ctrl_freeq::optim::ConsoleSink;

fn main() -> ctrl_freeq::Result<()> {
    let json = examples()
        .iter()
        .find(|e| e.0 == "single_qubit_parameters")
        .expect("bundled")
        .1;
    let mut cfg = Config::from_json(json)?;
    cfg.seed = Some(7);
    if let Some(alg) = std::env::args().nth(1) {
        cfg.optimization.algorithm = alg;
    }
    println!("single qubit, broadband Z -> -Z, {}", cfg.optimization.algorithm);
    let r = ctrl_freeq::run(&cfg, &mut ConsoleSink::new(10))?;
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
