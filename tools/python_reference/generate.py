"""Generate reference values from the Python ctrl-freeq package for the Rust parity tests.

Usage (from the Rust repository root):

    PYTHONDONTWRITEBYTECODE=1 <python-repo>/.venv/bin/python tools/python_reference/generate.py \
        <python-repo> tests/fixtures

The Python package is imported read-only from ``<python-repo>/src``; nothing is written anywhere but the output
directory.  Every case is deterministic - fixed offsets, no spread in the Rabi frequency or couplings - so the
Python and Rust setups produce the same numbers, and the parameter vector is fixed rather than random.
"""

import copy
import json
import math
import os
import sys

sys.dont_write_bytecode = True


def main():
    repo, out_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.join(repo, "src"))
    os.makedirs(out_dir, exist_ok=True)

    import numpy as np
    import torch

    from ctrl_freeq.api import CtrlFreeQAPI
    from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
        CtrlFreeQ,
        exp_mat_exact,
        exp_mat_torch,
        fidelity_hilbert,
        fidelity_liouville,
        state_hilbert,
        state_liouville,
        state_lindblad,
    )
    from ctrl_freeq.make_pulse.waveform_gen_torch import (
        waveform_gen_cart,
        waveform_gen_polar,
        waveform_gen_polar_phase,
    )
    from ctrl_freeq.setup.iterator_generation.generate_iterator import (
        h0_omega_1_iterator_torch,
    )
    from ctrl_freeq.setup.operator_generation.generate_operators import (
        create_hamiltonian_basis_torch,
    )
    from ctrl_freeq.utils.conversion import array_to_tensor
    from ctrl_freeq.utils.utility_functions import convert_attributes_to_numpy

    def cplx(a):
        a = np.asarray(a)
        if a.ndim == 0:
            return [float(a.real), float(a.imag)]
        return [cplx(v) for v in a]

    def real(a):
        return np.asarray(a, dtype=float).tolist()

    def evaluate(config):
        api = CtrlFreeQAPI(copy.deepcopy(config))
        p = api.parameters
        convert_attributes_to_numpy(p)
        model = getattr(p, "hamiltonian_model", None)
        rabi = array_to_tensor(p.Omega_R)
        H0 = array_to_tensor(p.H0)
        initials = array_to_tensor(p.initials)
        targets = array_to_tensor(p.targets)
        mat = array_to_tensor(p.mat)
        dt = array_to_tensor(p.pulse_duration / p.np_pulse)
        me = array_to_tensor(p.modulation_exponent)
        n_h0, n_rabi = H0.size(0), rabi.size(0)
        H0, initials, targets = h0_omega_1_iterator_torch(H0, n_rabi, initials, targets)
        dim = model.dim if model is not None else 2**p.n_qubits
        u_fun = exp_mat_exact if dim == 2 else exp_mat_torch
        if getattr(p, "dissipation_mode", "non-dissipative") == "dissipative":
            fid_fun, state_fun = fidelity_liouville, state_lindblad
            collapse = array_to_tensor(p.collapse_operators)
        elif p.space == "hilbert":
            fid_fun, state_fun, collapse = fidelity_hilbert, state_hilbert, None
        else:
            fid_fun, state_fun, collapse = fidelity_liouville, state_liouville, None
        modes = {"polar_phase": waveform_gen_polar_phase, "polar": waveform_gen_polar, "cart": waveform_gen_cart}
        wf_fun = [modes[m] for m in p.wf_mode]
        if model is not None:
            control_ops, op = model.control_ops_tensor(), None
        else:
            control_ops, op = None, create_hamiltonian_basis_torch(p.n_qubits)
        inst = CtrlFreeQ(
            p.n_para_updated, p.n_qubits, op, rabi, p.np_pulse, n_h0, n_rabi, mat, H0, dt, initials, targets,
            wf_fun, u_fun, state_fun, fid_fun, p.targ_fid, me,
            collapse_ops=collapse, hamiltonian_model=model, control_ops=control_ops,
        )
        n = int(sum(p.n_para_updated))
        x = torch.linspace(-0.8, 0.9, n, dtype=torch.float64, requires_grad=True)
        cost = inst.objective_function(x)
        cost.backward()
        grad = x.grad.detach().numpy()
        return {
            "config": config,
            "x": real(x.detach().numpy()),
            "cost": float(cost.detach()),
            "fidelity": float(inst.fid.detach()),
            "penalty": float(torch.as_tensor(inst.pen).detach()),
            "gradient": real(grad) if np.all(np.isfinite(grad)) else None,
            "mat": [real(m) for m in p.mat],
            "h0": cplx(H0.numpy()),
            "initials": cplx(initials.numpy()),
            "targets": cplx(targets.numpy()),
            "modulation": cplx(p.modulation_exponent),
        }

    def base(n):
        q = lambda v: [v] * n
        return {
            "qubits": [f"q{i + 1}" for i in range(n)],
            "parameters": {
                "Delta": [10e6 * (i + 1) for i in range(n)],
                "sigma_Delta": q(0.0),
                "Omega_R_max": q(40e6),
                "sigma_Omega_R_max": q(0.0),
                "pulse_duration": q(2e-7),
                "point_in_pulse": q(50),
                "wf_type": q("cheb"),
                "wf_mode": q("cart"),
                "amplitude_envelope": q("gn"),
                "amplitude_order": q(1),
                "coverage": q("single"),
                "sw": q(5e6),
                "pulse_offset": q(0.0),
                "pulse_bandwidth": q(5e5),
                "ratio_factor": q(0.5),
                "profile_order": q(2),
                "n_para": q(8),
                "J": [[0.0] * n for _ in range(n)],
            },
            "initial_states": [q("Z")],
            "target_states": {"Axis": [q("-Z")]},
            "optimization": {
                "space": "hilbert",
                "H0_snapshots": 1,
                "Omega_R_snapshots": 1,
                "algorithm": "l-bfgs",
                "max_iter": 10,
                "targ_fid": 0.999,
            },
        }

    def two_qubit(**params):
        c = base(2)
        c["parameters"]["J"] = [[0.0, 16.67e6], [0.0, 0.0]]
        c["parameters"]["coupling_type"] = "XY"
        c["parameters"]["sigma_J"] = 0.0
        c["initial_states"] = [["-Z", "Z"], ["X", "Y"]]
        c["target_states"] = {"Gate": ["CNOT", "CNOT"]}
        c["parameters"].update(params)
        return c

    cases = {}

    c = base(1)
    cases["single_cheb_cart_axis"] = c

    c = base(1)
    c["parameters"]["wf_type"] = ["leg"]
    c["parameters"]["wf_mode"] = ["polar"]
    c["parameters"]["amplitude_envelope"] = ["hs"]
    c["parameters"]["amplitude_order"] = [2]
    c["target_states"] = {"Gate": ["H"]}
    cases["single_leg_polar_gate"] = c

    c = base(1)
    c["parameters"]["wf_type"] = ["fou"]
    c["parameters"]["wf_mode"] = ["polar_phase"]
    c["parameters"]["n_para"] = [7]
    c["parameters"]["amplitude_envelope"] = ["quad"]
    c["initial_states"] = [["Z"], ["X"]]
    c["target_states"] = {"Axis": [["X"], ["-Y"]]}
    cases["single_fourier_polar_phase"] = c

    c = two_qubit(pulse_offset=[1e6, -0.5e6])
    cases["two_cheb_cart_cnot_offset"] = c

    c = two_qubit(wf_type=["hermite", "gegen"], wf_mode=["polar", "polar"])
    c["optimization"]["space"] = "liouville"
    c["initial_states"] = [["Z", "Z"]]
    c["target_states"] = {"Phi": [["x", "y"]], "Beta": [[90.0, 45.0]]}
    cases["two_liouville_phi_beta"] = c

    c = base(1)
    c["optimization"]["space"] = "liouville"
    c["optimization"]["dissipation_mode"] = "dissipative"
    c["parameters"]["T1"] = [1e-6]
    c["parameters"]["T2"] = [1.5e-6]
    cases["single_dissipative"] = c

    c = two_qubit()
    c["hamiltonian_type"] = "superconducting"
    c["parameters"]["coupling_type"] = "XY+ZZ"
    c["parameters"]["anharmonicities"] = [-330e6, -310e6]
    c["parameters"]["stark_shift_coeffs"] = [1e-9, 2e-9]
    c["target_states"] = {"Gate": ["iSWAP", "iSWAP"]}
    cases["two_superconducting_stark"] = c

    c = two_qubit()
    c["hamiltonian_type"] = "duffing_transmon"
    c["parameters"]["anharmonicities"] = [-330e6, -310e6]
    c["target_states"] = {"Gate": ["iSWAP", "iSWAP"]}
    cases["two_duffing"] = c

    c = two_qubit(wf_type=["poly", "chirp"])
    c["parameters"]["coupling_type"] = "XYZ"
    c["target_states"] = {"Gate": ["SWAP", "CZ"]}
    cases["two_poly_chirp_xyz"] = c

    for name, config in cases.items():
        result = evaluate(config)
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(result, f)
        g = "finite" if result["gradient"] is not None else "non-finite"
        print(f"{name}: cost {result['cost']:.12g}, fidelity {result['fidelity']:.12g}, gradient {g}")


if __name__ == "__main__":
    main()
