//! # ctrl-freeq
//!
//! Quantum gate and pulse design by optimal control.  A Rust port of the Python
//! [ctrl-freeq](https://pypi.org/project/ctrl-freeq/) package.
//!
//! A control pulse is expanded in a small basis (Chebyshev, Legendre, Fourier, ...), propagated through a batch of
//! drift Hamiltonians that sample the uncertainty in offsets, couplings and field strength, and scored by the mean
//! fidelity to a target state or gate.  The optimiser adjusts the basis coefficients; gradients and Hessians come
//! from the crate's own reverse-mode automatic differentiation.
//!
//! ## Quick start
//!
//! ```
//! use ctrl_freeq::config::{Config, examples};
//! use ctrl_freeq::optim::NoProgress;
//!
//! # fn main() -> ctrl_freeq::Result<()> {
//! // One of the Python package's bundled configurations: three initial states on one qubit.
//! let json = examples().iter().find(|e| e.0 == "single_qubit_parameters_multiple_initial_targ").unwrap().1;
//! let mut cfg = Config::from_json(json)?;
//! cfg.seed = Some(7); // reproducible
//! cfg.optimization.max_iter = 20;
//! let result = ctrl_freeq::run(&cfg, &mut NoProgress)?;
//! assert!(result.fidelity > 0.5);
//! let plots = ctrl_freeq::analyse(&cfg, &result)?;
//! assert_eq!(plots.pulses.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! ## Where to look
//!
//! * [`config`]: the configuration, in the Python package's JSON schema.
//! * [`run()`] and [`analyse()`]: optimise, then get the data for plots.
//! * [`optim`]: the optimisers and the interface a new one implements.
//! * [`hamiltonian`] and [`basis`]: the physical platforms and waveform bases, and how to add one.
//! * [`objective`] and [`autodiff`]: the cost function and the differentiation underneath it.
//!
//! ## Threads
//!
//! With the `parallel` feature, on by default, the batch elements and the Hessian's columns are evaluated on
//! [rayon](https://docs.rs/rayon)'s thread pool.  Results are bit-for-bit the same on one thread as on many.
//! `wasm32` has no threads and always runs sequentially.
#![deny(missing_docs)]

pub mod analysis;
pub mod autodiff;
pub mod basis;
pub mod config;
pub mod error;
pub mod hamiltonian;
pub mod linalg;
pub mod objective;
pub mod optim;
mod parallel;
pub mod run;
pub mod setup;
pub mod time;

pub use analysis::{Analysis, analyse};
pub use config::Config;
pub use error::{Error, Result};
pub use run::{RunResult, run};
