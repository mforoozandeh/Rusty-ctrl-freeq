//! An iSWAP on two exchange-coupled transmons, with and without the third level.
//!
//! `cargo run --release --example superconducting_iswap`

use ctrl_freeq::hamiltonian::default_config;
use ctrl_freeq::optim::ConsoleSink;

fn main() -> ctrl_freeq::Result<()> {
    let mut failed = false;
    for model in ["superconducting", "duffing_transmon"] {
        let mut cfg = default_config(model, 2)?;
        cfg.seed = Some(7);
        cfg.optimization.algorithm = "newton-cg".into();
        println!("{model}: iSWAP, {}", cfg.optimization.algorithm);
        let r = ctrl_freeq::run(&cfg, &mut ConsoleSink::new(5))?;
        println!(
            "fidelity {:.6} after {} iterations in {:.2} s: {}\n",
            r.fidelity,
            r.iterations,
            r.elapsed_s,
            r.exit.message()
        );
        failed |= r.fidelity - r.penalty < cfg.optimization.targ_fid;
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}
