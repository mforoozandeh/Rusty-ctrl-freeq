# Differences from the Python package

ctrl-freeq for Rust reads and writes the Python package's configuration files and reproduces its physics: the
parity tests in `tests/parity.rs` compare basis matrices, drift Hamiltonians, states, targets, modulation, the
cost and its gradient against the Python package 0.3.0 on nine deterministic problems, skipping only what a fix
below changes.  This file lists where the two differ, and why.

## Numerics

- **Liouville-space fidelity.** Python computes the Uhlmann fidelity `(Tr √(√ρ σ √ρ))²` through two
  eigendecompositions.  Every target state the configuration can describe is pure, and for a pure `σ` the
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
- **Gate targets.** Python scores a gate target state by state: each initial state against the gate applied to it.
  That is not the gate: with `|0⟩` as the initial state and a Z-gate target, doing nothing scores 1, although the
  identity's average gate fidelity to Z is 1/3.  When every initial state names the same gate, Rust scores the
  pulse by its average gate fidelity over the computational basis, averaged over the snapshots: Pedersen's formula
  on the propagated basis in Hilbert space, which counts leakage as loss, and Nielsen's formula over the Pauli
  operators in Liouville space, which counts relaxation.  The initial states then only choose what the dynamics
  plots show.  Different gates for different initial states can only mean state transfer, and keep Python's
  scoring.
- **Relaxation.** Python follows each unitary step with an explicit Euler step of the Lindblad dissipator, which
  does not keep states physical: a step longer than T1 turns populations negative, and fidelities can exceed 1.
  Rust applies the dissipator's exact channel `exp(dt·D)`, which is completely positive and trace preserving, and
  alternates it with the unitary step to the same first order in `dt`.  On the parity problem the two differ by
  1e-4, the size of the Euler error there.
- **Two-level transmons.** Python's two-level model writes `δ·Z + g·(X·X + Y·Y)` with spin-½ operators, while its
  three-level model writes `δ·n + g·(a†b + ab†)`: one configuration had the opposite detuning and half the exchange
  in the two-level model.  Rust's two-level model is the three-level one restricted to `|0⟩, |1⟩`:
  `−δ·Z + 2g·(X·X + Y·Y)`, with the Stark term `−s·Ω²·(cx² + cy²)·Z` so that `s` still adds to `δ`.  Python
  estimates the static ZZ from the anharmonicities as `2g²·(1/α_i + 1/α_j)`, which has the wrong sign and ignores
  the detuning.  Rust uses the second-order result `2g²·(α_i + α_j)/((Δ + α_i)·(Δ − α_j))` with `Δ = δ_i − δ_j`,
  which matches the three-level spectrum.  Its error grows roughly as the square of the mixing ratio `√2·g/Δ` of
  `|11⟩` with `|20⟩` and `|02⟩`, so it is refused above a ratio of 0.1.  That cutoff keeps the mixing small rather
  than bounding the error, which reaches about 2% at the cutoff itself; a calibrated `zz_crosstalk` or the
  three-level model covers the rest.
  Calibrated `zz_crosstalk` values mean what they meant.
- **Time grid.** Python samples the waveforms at `linspace(ε, T, N)`, spaced `T/(N−1)`, although each of the `N`
  steps lasts `T/N`, so a carrier offset turned `N/(N−1)` times too fast.  Rust samples each step at its middle,
  `(t + ½)·T/N`, and the dynamics plots show the states at the step boundaries `0, T/N, …, T`, starting from the
  initial state.
- **Coupling spread with fixed offsets.** Python builds a single drift snapshot whenever every qubit has single
  coverage and no offset spread, even with a coupling spread, so the pulse was made robust to one random coupling.
  Rust builds every snapshot whenever anything in the drift is uncertain.
- **Selective sampling.** Python lists each qubit's offsets left of its band, in it, then right of it, and pairs
  the qubits' offsets by index, so with equal ratios the qubits were always in or out of their bands together.
  Rust shuffles each qubit's offsets, so the snapshots also cover one qubit in its band and another outside.
- **Coverage with axis or gate targets.** Python applies an axis or gate target where the first qubit's excitation
  profile is exactly 1, and asks for the initial state elsewhere, whatever the other qubits' profiles.  With
  `band_selective` coverage the profile is a smooth super-Gaussian that is exactly 1 only at the centre, so almost
  every offset asked for the identity.  Rust applies an axis target qubit by qubit - each qubit's target axis in its
  band, its initial axis outside - and a gate only where every qubit is in its band, the identity elsewhere.
  `band_selective` coverage is refused with axis and gate targets; rotation (`Phi`/`Beta`) targets scale smoothly
  with its profile.
- **Observables of three-level transmons.** Python embeds the Pauli operators for the plots like gates, with the
  identity outside the computational subspace, so leaked population read +1 on every axis.  Rust projects them on
  the computational subspace: leaked population reads 0.
- **Dependent basis functions.** Where the requested basis functions are not linearly independent on the pulse's
  points - chirps, which are even, on few points, or high-order polynomials - QR completes the basis with arbitrary
  directions.  Rust refuses such bases, judging the rank as numpy's `matrix_rank` does.
- **Two-point pulses.** On two points the `gn` and `hs` envelopes are constant, and rescaling them divided by zero.
  Rust makes them flat.

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
  differ in value from numpy's, with the same distributions.

## Plots and runs

- **GPU.** `compute_resource: "gpu"` is accepted; the run uses the CPU and says so.  `cpu_cores` sets the number of
  threads.
- **Cancelling in the browser.** The web interface runs the optimisation in a Web Worker, which can only be stopped
  by ending it, so a cancelled browser run keeps its convergence history but not its partial pulse.  The desktop
  interface returns and plots the best pulse so far.  The Python GUI cannot cancel a run.
