//! Configuration checks.  Each problem is one sentence naming the field (and qubit, where relevant), so the GUI can
//! list them all at once.

use super::{Config, Coverage, HAMILTONIAN_TYPES, ROTATION_AXES, STATE_AXES, Targets};

pub(super) fn validate(cfg: &Config) -> Vec<String> {
    let mut v = Checks::default();
    let n = cfg.n_qubits();
    if n == 0 {
        v.push("at least one qubit is needed".into());
        return v.0;
    }
    let p = &cfg.parameters;
    let model = cfg.hamiltonian_type.as_deref();

    if let Some(m) = model {
        if !HAMILTONIAN_TYPES.contains(&m) {
            v.push(format!(
                "hamiltonian_type \"{m}\" is not one of {}",
                HAMILTONIAN_TYPES.join(", ")
            ));
        }
    }

    // Per-qubit array lengths.
    v.len("Omega_R_max", p.omega_r_max.len(), n);
    v.len("sigma_Omega_R_max", p.sigma_omega_r_max.len(), n);
    v.len("pulse_duration", p.pulse_duration.len(), n);
    v.len("point_in_pulse", p.point_in_pulse.len(), n);
    v.len("wf_type", p.wf_type.len(), n);
    v.len("wf_mode", p.wf_mode.len(), n);
    v.len("amplitude_envelope", p.amplitude_envelope.len(), n);
    v.len("amplitude_order", p.amplitude_order.len(), n);
    v.len("coverage", p.coverage.len(), n);
    v.len("sw", p.sw.len(), n);
    v.len("pulse_offset", p.pulse_offset.len(), n);
    v.len("pulse_bandwidth", p.pulse_bandwidth.len(), n);
    v.len("ratio_factor", p.ratio_factor.len(), n);
    v.len("profile_order", p.profile_order.len(), n);
    v.len("n_para", p.n_para.len(), n);
    if let Some(d) = &p.delta {
        v.len("Delta", d.len(), n);
    }
    if let Some(d) = &p.sigma_delta {
        v.len("sigma_Delta", d.len(), n);
        v.non_negative("sigma_Delta", d);
    }
    v.non_negative("sigma_Omega_R_max", &p.sigma_omega_r_max);
    if let Some(j) = &p.j {
        if let Err(e) = crate::hamiltonian::upper_coupling(j, n) {
            v.push(e);
        }
    }
    if let Some(s) = p.sigma_j {
        if !(s >= 0.0 && s.is_finite()) {
            v.push("sigma_J must be zero or positive".into());
        }
    }
    if v.has_length_problems() {
        return v.0;
    }

    // Pulse grid.
    let n_pulse = p.point_in_pulse[0];
    if n_pulse < 2 {
        v.push("point_in_pulse must be at least 2".into());
    }
    if !(p.pulse_duration[0] > 0.0 && p.pulse_duration[0].is_finite()) {
        v.push("pulse_duration must be positive".into());
    }

    // Bases, modes and envelopes.
    for q in 0..n {
        let name = &p.wf_type[q];
        match crate::basis::basis(name) {
            Err(_) => v.push(format!(
                "qubit {}: wf_type \"{name}\" is not one of {}",
                q + 1,
                crate::basis::basis_names().join(", ")
            )),
            Ok(b) => {
                let mode = p.wf_mode[q];
                if let Err(e) = b.check_n_para(p.n_para[q], mode) {
                    v.push(format!("qubit {}: {e}", q + 1));
                } else if b.columns(p.n_para[q], mode) > n_pulse {
                    v.push(format!(
                        "qubit {}: {} basis functions need at least as many points in the pulse (point_in_pulse = {n_pulse})",
                        q + 1,
                        b.columns(p.n_para[q], mode)
                    ));
                }
            }
        }
        if !crate::basis::envelope_names().contains(&p.amplitude_envelope[q].as_str()) {
            v.push(format!(
                "qubit {}: amplitude_envelope \"{}\" is not one of {}",
                q + 1,
                p.amplitude_envelope[q],
                crate::basis::envelope_names().join(", ")
            ));
        }
        if p.amplitude_order[q] == 0 {
            v.push(format!("qubit {}: amplitude_order must be at least 1", q + 1));
        }
        if matches!(p.coverage[q], Coverage::Selective | Coverage::BandSelective) {
            if !(0.0..=1.0).contains(&p.ratio_factor[q]) {
                v.push(format!("qubit {}: ratio_factor must be between 0 and 1", q + 1));
            }
            if p.pulse_bandwidth[q].is_nan() || p.pulse_bandwidth[q] < 0.0 {
                v.push(format!("qubit {}: pulse_bandwidth must not be negative", q + 1));
            }
        }
        if p.sw[q].is_nan() || p.sw[q] < 0.0 {
            v.push(format!("qubit {}: sw must not be negative", q + 1));
        }
    }

    // Model-specific parameters.
    let coupling = p.coupling_type.as_deref();
    match model {
        Some("superconducting") => {
            if let Some(c) = coupling {
                if n > 1 && !["XY", "ZZ", "XY+ZZ"].contains(&c) {
                    v.push(format!("coupling_type \"{c}\" is not one of XY, ZZ, XY+ZZ"));
                }
            }
            v.optional_len("anharmonicities", p.anharmonicities.as_deref(), n);
            v.optional_len("stark_shift_coeffs", p.stark_shift_coeffs.as_deref(), n);
            if let Some(zz) = &p.zz_crosstalk {
                if zz.len() != n || zz.iter().any(|row| row.len() != n) {
                    v.push(format!("zz_crosstalk must be a {n}x{n} matrix"));
                }
            }
        }
        Some("duffing_transmon") => {
            match &p.anharmonicities {
                None => v.push("the duffing_transmon model needs anharmonicities".into()),
                Some(a) => v.len("anharmonicities", a.len(), n),
            }
            if cfg.is_dissipative() {
                v.push("dissipative mode is not supported for the duffing_transmon model".into());
            }
        }
        _ => {
            if let Some(c) = coupling {
                if n > 1 && !["Z", "XY", "XYZ"].contains(&c.to_uppercase().as_str()) {
                    v.push(format!("coupling_type \"{c}\" is not one of Z, XY, XYZ"));
                }
            }
        }
    }

    // Dissipation.
    if cfg.is_dissipative() {
        match (&p.t1, &p.t2) {
            (Some(t1), Some(t2)) if t1.len() == n && t2.len() == n => {
                for q in 0..n {
                    if !(t1[q] > 0.0 && t1[q].is_finite()) {
                        v.push(format!("qubit {}: T1 must be positive and finite", q + 1));
                    }
                    if !(t2[q] > 0.0 && t2[q].is_finite()) {
                        v.push(format!("qubit {}: T2 must be positive and finite", q + 1));
                    } else if t2[q] > 2.0 * t1[q] {
                        v.push(format!("qubit {}: T2 must not exceed 2·T1", q + 1));
                    }
                }
            }
            _ => v.push(format!("dissipative mode needs T1 and T2 for each of the {n} qubits")),
        }
    }

    // States and targets.
    if cfg.initial_states.is_empty() {
        v.push("at least one initial state is needed".into());
    }
    for (i, s) in cfg.initial_states.iter().enumerate() {
        v.axes(&format!("initial state {}", i + 1), s, n, &STATE_AXES);
    }
    let n_init = cfg.initial_states.len();
    if cfg.target_states.len() != n_init {
        v.push(format!(
            "{} target entries for {n_init} initial states",
            cfg.target_states.len()
        ));
    }
    match &cfg.target_states {
        Targets::Axis(axes) => {
            for (i, s) in axes.iter().enumerate() {
                v.axes(&format!("target {}", i + 1), s, n, &STATE_AXES);
            }
        }
        Targets::Gate(gates) => {
            let valid = crate::setup::gate_names(n);
            for g in gates {
                if !valid.contains(&g.as_str()) {
                    if valid.is_empty() {
                        v.push(format!("gate targets are not available for {n} qubits"));
                    } else {
                        v.push(format!(
                            "gate \"{g}\" is not one of {} for {n} qubits",
                            valid.join(", ")
                        ));
                    }
                }
            }
        }
        Targets::PhiBeta { phi, beta } => {
            if phi.len() != beta.len() {
                v.push("Phi and Beta must have the same number of entries".into());
            }
            for (i, s) in phi.iter().enumerate() {
                v.axes(&format!("rotation {} axes", i + 1), s, n, &ROTATION_AXES);
            }
            for (i, b) in beta.iter().enumerate() {
                if b.len() != n || b.iter().any(|x| !x.is_finite()) {
                    v.push(format!("rotation {} needs {n} finite angles", i + 1));
                }
            }
        }
    }

    // Optimisation.
    let o = &cfg.optimization;
    if o.h0_snapshots == 0 {
        v.push("H0_snapshots must be at least 1".into());
    }
    if o.omega_r_snapshots == 0 {
        v.push("Omega_R_snapshots must be at least 1".into());
    }
    if o.max_iter == 0 {
        v.push("max_iter must be at least 1".into());
    }
    if !(o.targ_fid > 0.0 && o.targ_fid <= 1.0) {
        v.push("targ_fid must be in (0, 1]".into());
    }
    let names = crate::optim::optimizer_names();
    if !names.contains(&o.algorithm.as_str()) {
        v.push(format!(
            "algorithm \"{}\" is not supported; choose one of {}",
            o.algorithm,
            names.join(", ")
        ));
    }
    if let Some(r) = &cfg.compute_resource {
        if r != "cpu" && r != "gpu" {
            v.push(format!("compute_resource \"{r}\" is not cpu or gpu"));
        }
    }
    if cfg.cpu_cores == Some(0) {
        v.push("cpu_cores must be at least 1".into());
    }
    v.0
}

#[derive(Default)]
struct Checks(Vec<String>, bool);

impl Checks {
    fn push(&mut self, s: String) {
        self.0.push(s);
    }

    fn len(&mut self, field: &str, got: usize, n: usize) {
        if got != n {
            self.1 = true;
            self.0.push(format!("{field} has {got} entries for {n} qubits"));
        }
    }

    fn optional_len(&mut self, field: &str, v: Option<&[f64]>, n: usize) {
        if let Some(v) = v {
            self.len(field, v.len(), n);
        }
    }

    fn has_length_problems(&self) -> bool {
        self.1
    }

    fn non_negative(&mut self, field: &str, v: &[f64]) {
        if v.iter().any(|x| !(*x >= 0.0 && x.is_finite())) {
            self.0.push(format!("{field} must be zero or positive"));
        }
    }

    fn axes(&mut self, what: &str, s: &[String], n: usize, allowed: &[&str]) {
        if s.len() != n {
            self.0.push(format!("{what} has {} axes for {n} qubits", s.len()));
            return;
        }
        for a in s {
            if !allowed.contains(&a.as_str()) {
                self.0
                    .push(format!("{what}: \"{a}\" is not one of {}", allowed.join(", ")));
            }
        }
    }
}
