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

Rust supports `l-bfgs`, `newton-cg`, `newton-exact`, `cobyla`, `bobyqa`, `nelder-mead`, `spsa` and `cma-es`.
Python offers additional torchmin methods but no `bobyqa` or `cma-es`.

- **`newton-cg`:** Rust uses Armijo backtracking; Python uses strong Wolfe.
- **`newton-exact`:** Rust uses a trust region with Steihaug's solver; Python uses a line search.
- **`l-bfgs`:** Rust uses 10 correction pairs and Moré–Thuente; Python defaults to 100 pairs and strong Wolfe.
- **Derivative-free bounds:** every Rust derivative-free optimiser imposes `±max(100, 2·max|x_initial|)` on
  parameters. Python's wrapper supplies no such box.
- **`nelder-mead`:** scipy's initial simplex - each coordinate of the starting point moved 5%, or to `0.00025`
  where it is zero - and convergence once the simplex spans less than `1e-8` in both the parameters and the
  objective.
- **`spsa`:** the power series, Bernoulli perturbation, gradient sample and calibration follow Qiskit's SPSA,
  with three deliberate differences, all because this objective is deterministic rather than shot-noise
  limited. The perturbation is `0.01` rather than Qiskit's `0.2`. Blocking is always on rather than off by
  default - a third evaluation per iteration rejects any step that raises the objective, without which a
  calibrated step can land somewhere far steeper than where it was measured and the iterate runs away - and its
  allowed increase is zero, so Qiskit's 25 evaluations estimating the objective's standard deviation are not
  spent. The stability constant spans a tenth of the iterations, as Spall recommends, rather than Qiskit's
  zero; `a` is rescaled so the first step is unchanged. Expect SPSA to find a basin rather than converge inside
  one: on a curved, ill-conditioned valley it settles two orders of magnitude short of the simplex and
  population methods.
- **`cma-es`:** Hansen's reference defaults for the population, weights, learning rates and damping, with the
  distribution starting spherical at standard deviation `0.5`. Python has no CMA-ES.
- **Stochastic runs:** `spsa` and `cma-es` draw from the run's seed, so a configuration with a `seed` repeats
  exactly. Python's Qiskit optimisers have no seed field.
- **Returned point:** Rust returns the best point evaluated, including line-search trials. Python returns the
  solver or early-stop point, without that guarantee.

## Interfaces and runtime

- **Python-only APIs:** Rust has no equivalent to `PiecewiseAPI` or the per-snapshot `zz_instances` override.
- **Amplitude reporting:** Rust reports normalised peaks and the largest violation. Python additionally reports
  excess, limit status, physical peaks and their range across sampled Rabi gains.
- **GPU:** Python supports CUDA. Rust runs on CPU, including when GPU execution is requested.
- **Cancellation:** Rust desktop returns the best pulse so far; browser cancellation retains history but no
  partial pulse. Python's GUI has no cancellation control.
