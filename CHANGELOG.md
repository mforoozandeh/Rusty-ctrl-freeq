# Changelog

## Unreleased

Physics corrections; each is described in `DEVIATIONS.md`.

- Changed: a gate target named for every initial state is scored by the average gate fidelity over the
  computational basis (leakage and relaxation included), not state by state; different gates per initial state
  still mean state transfer.
- Changed: two-level transmons share the three-level model's conventions: qubit frequency `+δ`, exchange hopping
  `g`, Stark coefficient adding to `δ`.  The ZZ estimate from the anharmonicities has the right sign and includes
  the detuning, and is refused above a mixing ratio of 0.1 with `|20⟩` and `|02⟩` - a small-mixing heuristic, not
  an accuracy bound.
- Changed: waveform samples sit at the middle of each time step; `Analysis::state_times_ns` times the dynamics at
  the step boundaries, from the initial state.
- Changed: `band_selective` coverage requires rotation targets; selective axis targets apply qubit by qubit and
  gates only where every qubit is in its band.
- Fixed: Lindblad relaxation uses the dissipator's exact channel, so states stay physical and fidelities at most 1.
- Fixed: gate names are compared canonically, so `CNOT` and `CX` together still ask for the gate.
- Fixed: a coupling spread with fixed offsets gets every drift snapshot, not one.
- Fixed: `LindbladOps::new` rejects a negative or non-finite step, which is not a channel.
- Fixed: selective offsets are shuffled per qubit, so qubits fall in and out of their bands independently.
- Fixed: leaked three-level population no longer reads +1 on every Pauli axis.
- Fixed: linearly dependent basis functions are refused instead of completed arbitrarily; two-point pulses no
  longer produce NaN envelopes.

## 0.1.0 - 2026-09-19

First release of the Rust port of ctrl-freeq 0.3.0.

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
