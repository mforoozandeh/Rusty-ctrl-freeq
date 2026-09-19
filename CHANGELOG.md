# Changelog

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
