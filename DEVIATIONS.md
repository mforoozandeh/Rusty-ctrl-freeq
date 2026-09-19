# Differences from the Python package

ctrl-freeq for Rust reads and writes the Python package's configuration files and reproduces its physics: the
parity tests in `tests/parity.rs` compare basis matrices, drift Hamiltonians, states, targets, modulation, the
cost and its gradient against the Python package 0.3.0 on nine deterministic problems.  This file lists where
the two differ, and why.

## Numerics

- **Liouville-space fidelity.** Python computes the Uhlmann fidelity `(Tr √(√ρ σ √ρ))²` through two
  eigendecompositions.  Every target the configuration can describe is a pure state, and for a pure `σ` the
  Uhlmann fidelity equals `Re Tr(ρσ)` exactly, so that is what Rust computes.  It avoids the square root's infinite
  derivative at the zero eigenvalues of pure states.  Setup refuses mixed targets rather than silently changing
  meaning.  On the parity problems the two agree to about 1e-7; Python loses the last digits in its
  eigendecompositions.
- **Two-level propagators.** The closed form for `exp(−i·dt·H)` also handles Hamiltonians with a non-zero trace;
  Python's assumes `H₀₀ = −H₁₁`.  Identical for every model in the package.

## Behaviour fixed

- **Several initial states with several drift snapshots.** Python stacks the drift Hamiltonians by initial state
  but the initial states and targets by offset, so with more than one initial state and more than one snapshot a
  drift is paired with the wrong target and some combinations never appear.  Rust builds the batch as every
  (initial state, snapshot, Rabi snapshot) combination.  The two agree whenever there is one initial state or one
  snapshot, which covers every bundled example.
- **Density matrices for the three-level transmon.** Python embeds initial density matrices into the `3ⁿ` space as
  if they were state vectors.  Rust embeds them as `P·ρ·Pᵀ`.
- **Couplings with a spread.** Python adds noise to both triangles of the coupling matrix independently and then
  rejects the result as asymmetric whenever both triangles were given.  Rust reads the couplings once, from either
  triangle, and draws each coupling's noise once.  All three models accept either triangle; Python's
  superconducting and transmon models read the upper one only.

## Optimisers

- **Names.** `l-bfgs`, `newton-cg`, `newton-exact`, `cobyla` and `bobyqa`.  Any other name from the Python package
  is rejected with the list of supported ones.  `bobyqa` is new.
- **`newton-cg`** is the line-search truncated Newton method of Nocedal and Wright (Algorithm 7.1) on exact
  Hessian-vector products, as in torchmin, with Armijo backtracking where torchmin uses a strong-Wolfe search.
- **`newton-exact`** is Newton's method in a trust region on the exact Hessian (argmin's trust region with
  Steihaug's subproblem solver).  torchmin's `newton-exact` uses a line search instead; the trust region copes
  with indefinite Hessians, which fidelity landscapes have.
- **`l-bfgs`** keeps 10 correction pairs (argmin, Moré–Thuente line search); torchmin keeps 100.
- **`cobyla`** starts with a trust radius of 1.0, as scipy does.  Its source is vendored from the `cobyla` crate so
  runs can stop at the target fidelity or on request.
- **Derivative-free iterations.** For `cobyla` and `bobyqa` one iteration is one function evaluation and
  `max_iter` limits evaluations, as in the Python package's Qiskit wrapper.  Both keep parameters inside a box of
  ±100 (or twice the largest initial value), far outside where basis coefficients go.
- **Stopping.** Every optimiser stops when fidelity minus penalty reaches `targ_fid`, as Python does, and returns
  the best point it evaluated.

## Randomness

- **Seeds.** An optional top-level `"seed"` makes a run reproducible.  Without one a seed is drawn from the clock
  and reported with the result.  The Python package ignores the field.
- **Values.** Rust draws from ChaCha8, so the sampled offsets, couplings, Rabi frequencies and initial coefficients
  differ in value from numpy's, with the same distributions and structure.

## Plots and runs

- **Time step.** The dynamics plots use the optimiser's time step, `duration / points`.  Python's plotter uses the
  spacing of its time grid, which is slightly larger.
- **GPU.** `compute_resource: "gpu"` is accepted; the run uses the CPU and says so.  `cpu_cores` sets the number of
  threads.

## Kept as in Python, worth knowing

- **Selective coverage and axis or gate targets.** An offset gets the target only where the first qubit's
  excitation profile is exactly 1, otherwise the target is the initial state.  With `band_selective` coverage the
  profile is a smooth super-Gaussian that is exactly 1 only at the centre, so almost every offset asks for the
  identity.  Rotation (`Phi`/`Beta`) targets scale smoothly with the profile and do not have this problem.
