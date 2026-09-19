# Rusty-ctrl-freeq

Quantum gate and pulse design by optimal control, in Rust.  A port of the Python
[ctrl-freeq](https://github.com/mforoozandeh/ctrl-freeq) package that reads the same configuration files, with its
own automatic differentiation, and builds for WebAssembly.  The crate is published as `ctrl-freeq` and imported as
`ctrl_freeq`, the same names as the Python package.

A pulse on each qubit is expanded in a few smooth basis functions.  It is propagated through a batch of drift
Hamiltonians that sample the uncertainty in offsets, couplings and drive strength, and scored by the mean fidelity
to a target state or gate.  An optimiser adjusts the basis coefficients using exact gradients and Hessians.

## Features

- **Platforms:** spin chains (Z, XY or XYZ coupling), two-level transmons (exchange and ZZ coupling, AC Stark
  shift), and three-level transmons.
- **Evolution:** state vectors, density matrices, and Lindblad relaxation with T1 and T2.
- **Robustness:** offsets sampled per qubit (single, broadband, selective, band-selective), couplings with a
  spread, and several drive strengths.
- **Targets:** product states, gates (X Y Z H S T, CNOT CZ SWAP iSWAP √iSWAP ECR, Toffoli), or rotations.
- **Waveforms:** Cartesian, polar, or phase-only over Chebyshev, Legendre, Fourier, polynomial, Hermite,
  Gegenbauer, chirp or random bases.
- **Optimisers:** `l-bfgs`, `newton-cg`, `newton-exact`, `cobyla`, `bobyqa`, all with progress reports,
  cancellation and a target-fidelity stop.
- **Parallel:** batches and Hessians run on all cores, with bit-for-bit identical results on any thread count.

Differences from the Python package are listed in [DEVIATIONS.md](DEVIATIONS.md).

## Quick start

```toml
[dependencies]
ctrl-freeq = "0.1"
```

```rust
use ctrl_freeq::config::Config;
use ctrl_freeq::optim::ConsoleSink;

fn main() -> ctrl_freeq::Result<()> {
    let json = std::fs::read_to_string("single_qubit_parameters.json")?;
    let mut cfg = Config::from_json(&json)?;
    cfg.seed = Some(7); // reproducible
    let result = ctrl_freeq::run(&cfg, &mut ConsoleSink::new(10))?;
    println!("fidelity {:.6}: {}", result.fidelity, result.exit.message());
    let plots = ctrl_freeq::analyse(&cfg, &result)?; // pulses, dynamics, excitation profiles
    println!("{} time points", plots.times_ns.len());
    Ok(())
}
```

The Python package's example configurations are bundled: `ctrl_freeq::config::examples()`.  For a model's default
setup use `ctrl_freeq::hamiltonian::default_config("superconducting", 2)`.

## The interface

A desktop and browser application edits a configuration, runs it with live convergence, and plots the pulses,
the state dynamics and the excitation profile.  Configurations load and save in the Python package's JSON format;
results export as JSON and the waveforms as CSV.

```bash
cargo run -p ctrl-freeq-gui --release          # desktop
trunk serve --config gui/Trunk.toml            # browser, at http://127.0.0.1:8080
trunk build --release --config gui/Trunk.toml  # static files in gui/dist for any web host
```

In the browser everything runs locally, in a Web Worker, on one thread; nothing is uploaded.  Cancelling there
keeps the convergence so far but not the partial pulse, because a busy worker can only be stopped by ending it;
natively, a cancelled run returns and plots its best pulse.  Pushes to `main` publish the site to Cloudflare Pages;
see [deploy/README.md](deploy/README.md).

## Examples

```bash
cargo run --release --example single_qubit            # broadband inversion
cargo run --release --example two_qubit_cnot          # CNOT on coupled spins
cargo run --release --example dissipative             # inversion with T1 and T2
cargo run --release --example superconducting_iswap   # iSWAP on two- and three-level transmons
cargo run --release --example timing                  # speed on 1 and N threads
```

## Threads and WebAssembly

The `parallel` feature, on by default, spreads batch elements and Hessian columns over rayon's thread pool;
`cpu_cores` in the configuration limits it.  Without the feature, and always on `wasm32-unknown-unknown`,
everything runs on the calling thread.  The library needs no filesystem, clock or entropy source in the browser.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), which includes step-by-step recipes for adding an optimiser, a basis or a
Hamiltonian model.

## License

Apache-2.0.  The vendored COBYLA in `src/optim/cobyla` is MIT-licensed; see the licence file beside it.
