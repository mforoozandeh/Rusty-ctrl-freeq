# Contributing

## Development

```bash
cargo test                      # library tests; the slow convergence tests are skipped in debug builds
cargo test --release            # everything, including the convergence gate
cargo test --release --lib --no-default-features   # the sequential build wasm32 uses
cargo clippy --all-targets -- -D warnings
cargo fmt --all
cargo build --target wasm32-unknown-unknown         # the library must keep building for the browser
cargo run --release --example timing                # speed, and identical results on 1 and N threads
```

The Python parity fixtures in `tests/fixtures` are generated from the Python package; see
`tools/python_reference/README.md`.

The interface is a separate workspace member, so the commands above leave it alone:

```bash
cargo test -p ctrl-freeq-gui
cargo clippy -p ctrl-freeq-gui --all-targets -- -D warnings
cargo clippy -p ctrl-freeq-gui --target wasm32-unknown-unknown -- -D warnings
cargo run -p ctrl-freeq-gui --release
trunk serve --config gui/Trunk.toml
```

A new optimiser, basis or Hamiltonian model appears in the interface without interface changes: its dropdowns
read `optimizer_names()`, `basis_names()` and `model_names()`.  Only model-specific fields need a line in
`gui/src/inputs.rs`.

## Where things live

| Module | What it owns |
|---|---|
| `config` | The JSON schema, examples, validation |
| `setup` | Config → `Problem`: sampling, states, targets, batch |
| `basis` | Basis functions, envelopes, QR |
| `hamiltonian` | Physical platforms |
| `linalg` | Small dense matrices, `expm` and its adjoint |
| `autodiff` | `Scalar`, `Dual`, the tape and its operations, `batch_mean` |
| `objective` | The cost graph; value, gradient, Hessian-vector product, Hessian |
| `optim` | The optimiser interface and the algorithms |
| `run`, `analysis` | One-call runs, and the data plots need |
| `parallel` | The only code that knows about threads |
| `gui/` | The interface: `run.rs` is shared by the native thread and the Web Worker; `runner_*.rs` carry it |

## Adding an optimiser

1. Create `src/optim/<name>.rs` with a type implementing `Optimizer`.  Use `Monitor` for bookkeeping:

   ```rust
   pub struct Spsa;

   impl Optimizer for Spsa {
       fn name(&self) -> &'static str { "spsa" }
       fn derivatives(&self) -> Derivatives { Derivatives::None }
       fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
           let mut monitor = Monitor::new(ctl);
           let mut x = x0.to_vec();
           let mut k = 0;
           while monitor.exit.is_none() {
               // ... evaluate with obj.value(&x)?, then:
               // monitor.evaluated(&x, eval);   // counts it and keeps the best point
               // k += 1; monitor.report(k, eval, None);   // progress, target, cancel and max_iter checks
           }
           monitor.finish(k, Exit::Converged)
       }
   }
   ```

2. Add it to `optimizer()` and `optimizer_names()` in `src/optim/mod.rs`.  Validation and the GUI's algorithm list
   read `optimizer_names()`, so nothing else changes.
3. `tests/optimizers.rs` runs every name in `optimizer_names()` through Rosenbrock, the target stop, cancellation
   and the iteration limit.  If the method is derivative-free, add it to `is_gradient_based`'s exclusions.

## Adding a basis

1. Add a type implementing `Basis` in `src/basis/`: `name` and `matrix` at least; override `columns`,
   `parameter_count`, `check_n_para` and `initial_coefficients` if its parameter counting differs from the
   default, as the Fourier basis's does.
2. Add it to `basis()` and `basis_names()` in `src/basis/mod.rs`.
3. Test its columns against a closed form in `src/basis/tests.rs`.

## Adding a Hamiltonian model

1. Add a type implementing `HamiltonianModel` in `src/hamiltonian/`: the drift for given offsets and couplings,
   the control operators, and the control channels.  A channel says which waveform component drives it and how it
   scales with the Rabi frequency, so models need no autodiff code.  Override the `embed_*` methods if the model's
   space is larger than the qubits' (see `duffing.rs`).
2. Add its name to `HAMILTONIAN_TYPES` in `src/config/mod.rs`, and add cases to `model_from_config` and
   `default_config` in `src/hamiltonian/mod.rs`.
3. Check model-specific parameters in `src/config/validate.rs`.
4. Add it to the model list in `tests/gradients.rs`; the gradient check then covers it in every state space and
   waveform mode.

## Style

- No commits of generated fixtures without the generator change that produced them.
- Public items carry documentation (`#![deny(missing_docs)]`).
- The library never panics on user input; errors are `ctrl_freeq::Error`.
