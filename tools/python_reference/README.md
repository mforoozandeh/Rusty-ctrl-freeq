# Python reference fixtures

`generate.py` evaluates the Python ctrl-freeq package on nine deterministic problems at a fixed parameter vector
and writes everything the Rust parity test (`tests/parity.rs`) compares: basis matrices, drift Hamiltonians,
initial states, targets, modulation, cost, fidelity, penalty and gradient.

Regenerate from the Rust repository root, with the Python repository checked out and its environment installed:

```bash
PYTHONDONTWRITEBYTECODE=1 ../../PycharmProjects/ctrl-freeq/.venv/bin/python \
    tools/python_reference/generate.py ../../PycharmProjects/ctrl-freeq tests/fixtures
```

The script imports the package from `<python-repo>/src` and writes only into the output directory.  The committed
fixtures were generated from ctrl-freeq 0.3.0.
