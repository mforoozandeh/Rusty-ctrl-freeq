# Differences from Python 0.4.0

## Numerics and sampling

- **Two-level propagators:** Rust handles non-zero-trace Hamiltonians; Python's closed form assumes a traceless
  matrix. This affects custom inputs, not the built-in two-level models.
- **Perturbative ZZ:** Python skips pairs where either anharmonicity is zero; Rust evaluates the formula and
  mixing guard.
- **Selective sampling:** Rust requires an even tail-sample count; Python allows odd counts. For 10 snapshots
  and `ratio_factor = 0.3`, Rust allocates 4 tail samples and Python 3, changing the ensemble weights.
- **Randomness:** Rust exposes and reports a run seed and uses ChaCha8. Python has no configuration seed field
  and uses NumPy's generator; sampled values differ.

## Optimisers

Rust supports `l-bfgs`, `newton-cg`, `newton-exact`, `cobyla` and `bobyqa`. Python offers additional
torchmin/Qiskit methods but no `bobyqa`.

- **`newton-cg`:** Rust uses Armijo backtracking; Python uses strong Wolfe.
- **`newton-exact`:** Rust uses a trust region with Steihaug's solver; Python uses a line search.
- **`l-bfgs`:** Rust uses 10 correction pairs and Moré–Thuente; Python defaults to 100 pairs and strong Wolfe.
- **Derivative-free bounds:** Rust's COBYLA and BOBYQA impose `±max(100, 2·max|x_initial|)` on parameters.
  Python's wrapper supplies no such box.
- **Returned point:** Rust returns the best point evaluated, including line-search trials. Python returns the
  solver or early-stop point, without that guarantee.

## Interfaces and runtime

- **Python-only APIs:** Rust has no equivalent to `PiecewiseAPI` or the per-snapshot `zz_instances` override.
- **Amplitude reporting:** Rust reports normalised peaks and the largest violation. Python additionally reports
  excess, limit status, physical peaks and their range across sampled Rabi gains.
- **GPU:** Python supports CUDA. Rust runs on CPU, including when GPU execution is requested.
- **Cancellation:** Rust desktop returns the best pulse so far; browser cancellation retains history but no
  partial pulse. Python's GUI has no cancellation control.
