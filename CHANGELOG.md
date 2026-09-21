# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-21

Corrections to the physics, and the reporting that goes with them.  What still differs from the Python package is
listed in [DEVIATIONS.md](DEVIATIONS.md).

### Added

- `Dynamics::leakage`, with its spread over the batch snapshots, and a trace in the interface: the population
  outside the computational subspace over time.
- `RunResult::peak_amplitude` per qubit, and a notice when a pulse exceeds its maximum Rabi frequency, which is
  penalised in the cost rather than enforced.

### Changed

- A gate target named for every initial state is scored by the average gate fidelity over the computational basis,
  leakage and relaxation included, rather than state by state.  Different gates for different initial states still
  mean state transfer.
- Two-level transmons share the three-level model's conventions: qubit frequency `+δ`, exchange hopping `g`, and a
  Stark coefficient that adds to `δ`.  The ZZ estimate from the anharmonicities has the right sign and includes the
  detuning, and is refused above a mixing ratio of 0.1 with `|20⟩` and `|02⟩` - a small-mixing heuristic, not an
  accuracy bound.
- Waveform samples sit at the middle of each time step, and `Analysis::state_times_ns` times the dynamics at the
  step boundaries, starting from the initial state.
- `pulse_bandwidth` is the band-selective profile's full width at half maximum, for every super-Gaussian order; it
  was the width at a quarter of the maximum for order 1, and narrower above.  `band_selective` coverage now needs a
  positive bandwidth and an order of at least 1.
- `band_selective` coverage requires rotation targets; selective axis targets apply qubit by qubit, and gates only
  where every qubit is in its band.

### Fixed

- Lindblad relaxation uses the dissipator's exact channel in Strang splitting, so states stay physical, fidelities
  are at most 1, and the error is second order in the time step.
- Gate names are compared canonically, so `CNOT` and `CX` together still ask for the gate.
- A coupling spread with fixed offsets gets every drift snapshot, not one.
- Selective offsets are shuffled per qubit, so qubits fall in and out of their bands independently.
- Leaked three-level population no longer reads +1 on every Pauli axis.
- Linearly dependent basis functions are refused instead of completed arbitrarily, and two-point pulses no longer
  produce NaN envelopes.
- `LindbladOps::new` rejects a negative or non-finite step, which is not a channel.

## [0.1.0] - 2026-09-19

First release of the Rust port of ctrl-freeq 0.3.0.

### Added

- Reads and writes the Python package's JSON configurations; optional `"seed"` for reproducible runs.
- Physics: Hilbert and Liouville spaces, Lindblad T1/T2 relaxation; spin chains (Z, XY, XYZ coupling), two-level
  transmons (XY and ZZ coupling, Stark shift), three-level transmons; single, broadband, selective and
  band-selective coverage; axis, gate and rotation targets; drift and Rabi snapshots; carrier modulation.
- Waveforms: `cart`, `polar`, `polar_phase` over Chebyshev, Legendre, Fourier, polynomial, Hermite, Gegenbauer,
  chirp and random bases, with `gn`, `hs` and `quad` envelopes.
- Automatic differentiation: a reverse-mode tape over real and complex matrices, with exact Hessian-vector
  products from dual numbers.
- Optimisers: L-BFGS, Newton-CG, trust-region Newton, COBYLA, BOBYQA, behind one interface with progress
  reporting, cancellation and a target-fidelity stop.
- Parallel batch evaluation on rayon, bit-for-bit identical on any number of threads; builds for wasm32.
- Plot data: pulses, state dynamics and excitation profiles.
- Interface (`gui/`), native and in the browser: every configuration field, presets, JSON load and save, live
  convergence, pulse, dynamics and excitation-profile plots, results JSON and waveform CSV export.  The browser
  build runs the optimisation in a Web Worker.

[unreleased]: https://github.com/mforoozandeh/Rusty-ctrl-freeq/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mforoozandeh/Rusty-ctrl-freeq/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mforoozandeh/Rusty-ctrl-freeq/releases/tag/v0.1.0
