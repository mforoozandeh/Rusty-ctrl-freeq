//! The optimisation problem as the Python package writes it: the same JSON schema, field for field.
//!
//! Frequencies are in Hz and times in seconds, exactly as in the Python configuration files; the setup converts
//! them to angular frequencies.  Per-qubit settings are parallel arrays indexed by qubit.  One field is new: an
//! optional top-level `"seed"`, which makes the random parts of a run - the initial coefficients and the sampled
//! offsets - reproducible.  The Python package ignores it.

mod examples;

pub use examples::examples;

use serde::de::{self, Deserializer};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A complete optimisation problem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// The physical platform: `spin_chain`, `superconducting` or `duffing_transmon`.  Absent means the Python
    /// package's original spin-chain path, whose default coupling is `Z`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hamiltonian_type: Option<String>,
    /// One name per qubit; the count sets the number of qubits.
    pub qubits: Vec<String>,
    /// `cpu` or `gpu`.  This version always computes on the CPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_resource: Option<String>,
    /// Number of CPU threads to use natively.  Absent means all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<usize>,
    /// Seed for every random choice in a run.  Absent means one is drawn from the clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Physical and pulse parameters.
    pub parameters: Parameters,
    /// Initial states: for each, one axis (`Z`, `-Z`, `X`, `-X`, `Y`, `-Y`) per qubit.
    pub initial_states: Vec<Vec<String>>,
    /// What each initial state should become.
    pub target_states: Targets,
    /// Optimiser settings.
    pub optimization: Optimization,
}

/// Physical and pulse parameters.  Arrays have one entry per qubit unless noted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameters {
    /// Offset (chemical shift, detuning) per qubit in Hz.  Absent means zero.
    #[serde(rename = "Delta", default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<Vec<f64>>,
    /// Standard deviation of the offset in Hz.  Absent means zero.
    #[serde(rename = "sigma_Delta", default, skip_serializing_if = "Option::is_none")]
    pub sigma_delta: Option<Vec<f64>>,
    /// Maximum Rabi frequency (drive amplitude) in Hz.
    #[serde(rename = "Omega_R_max")]
    pub omega_r_max: Vec<f64>,
    /// Standard deviation of the Rabi frequency in Hz.
    #[serde(rename = "sigma_Omega_R_max")]
    pub sigma_omega_r_max: Vec<f64>,
    /// Pulse duration in seconds.  Only the first entry is used, as in Python.
    pub pulse_duration: Vec<f64>,
    /// Number of time points in the pulse.  Only the first entry is used, as in Python.
    pub point_in_pulse: Vec<usize>,
    /// Basis per qubit: `cheb`, `leg`, `poly`, `fou`, `hermite`, `gegen`, `chirp` or `random`.
    pub wf_type: Vec<String>,
    /// Waveform parametrisation per qubit.
    pub wf_mode: Vec<WaveformMode>,
    /// Amplitude envelope per qubit: `gn`, `hs` or `quad`.
    pub amplitude_envelope: Vec<String>,
    /// Envelope order per qubit.
    pub amplitude_order: Vec<u32>,
    /// Offset coverage per qubit.
    pub coverage: Vec<Coverage>,
    /// Sweep width (offset range) in Hz.
    pub sw: Vec<f64>,
    /// Carrier offset of the pulse in Hz.
    pub pulse_offset: Vec<f64>,
    /// Excitation bandwidth in Hz, for the selective coverages.
    pub pulse_bandwidth: Vec<f64>,
    /// Fraction of offsets sampled outside the band, for the selective coverages.
    pub ratio_factor: Vec<f64>,
    /// Super-Gaussian order of the band-selective profile.
    pub profile_order: Vec<u32>,
    /// Number of basis coefficients per qubit, as the user sets it.
    pub n_para: Vec<usize>,
    /// Coupling matrix in Hz, `n × n`; upper, lower or symmetric.  Absent means zero.
    #[serde(rename = "J", default, skip_serializing_if = "Option::is_none")]
    pub j: Option<Vec<Vec<f64>>>,
    /// Coupling type: `Z`, `XY` or `XYZ` for spin chains; `XY`, `ZZ` or `XY+ZZ` for superconducting qubits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupling_type: Option<String>,
    /// Standard deviation of the non-zero couplings in Hz.
    #[serde(rename = "sigma_J", default, skip_serializing_if = "Option::is_none")]
    pub sigma_j: Option<f64>,
    /// Amplitude-damping time T1 in seconds, dissipative mode only.
    #[serde(rename = "T1", default, skip_serializing_if = "Option::is_none")]
    pub t1: Option<Vec<f64>>,
    /// Coherence time T2 in seconds, dissipative mode only.
    #[serde(rename = "T2", default, skip_serializing_if = "Option::is_none")]
    pub t2: Option<Vec<f64>>,
    /// Transmon anharmonicities in Hz (superconducting and Duffing models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anharmonicities: Option<Vec<f64>>,
    /// Calibrated static ZZ coupling matrix in Hz (superconducting model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zz_crosstalk: Option<Vec<Vec<f64>>>,
    /// AC Stark shift coefficients, dimensionless (superconducting model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stark_shift_coeffs: Option<Vec<f64>>,
}

/// Optimiser settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Optimization {
    /// State space.
    pub space: Space,
    /// Whether T1/T2 relaxation is included.  Absent means non-dissipative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dissipation_mode: Option<Dissipation>,
    /// Number of sampled drift Hamiltonians (offsets and couplings).
    #[serde(rename = "H0_snapshots")]
    pub h0_snapshots: usize,
    /// Number of sampled Rabi frequencies.
    #[serde(rename = "Omega_R_snapshots")]
    pub omega_r_snapshots: usize,
    /// Optimiser name; see [`optimizer_names`](crate::optim::optimizer_names).
    pub algorithm: String,
    /// Iteration limit.  For derivative-free optimisers an iteration is one function evaluation.
    pub max_iter: usize,
    /// Target fidelity; the run stops once fidelity minus penalty reaches it.
    pub targ_fid: f64,
}

/// State space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Space {
    /// State vectors.
    Hilbert,
    /// Density matrices.
    Liouville,
}

/// Whether relaxation is modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dissipation {
    /// Unitary evolution only.
    #[serde(rename = "non-dissipative")]
    NonDissipative,
    /// Lindblad evolution with T1/T2; forces Liouville space.
    #[serde(rename = "dissipative")]
    Dissipative,
}

/// How a qubit's waveform is parametrised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveformMode {
    /// Two basis expansions for the x and y quadratures.
    #[serde(rename = "cart")]
    Cart,
    /// Two basis expansions for amplitude and phase.
    #[serde(rename = "polar")]
    Polar,
    /// A fixed envelope scaled by one amplitude parameter, and a basis expansion for the phase.
    #[serde(rename = "polar_phase")]
    PolarPhase,
}

impl WaveformMode {
    /// Every mode, in the order the interface lists them.
    pub const ALL: [WaveformMode; 3] = [WaveformMode::Cart, WaveformMode::Polar, WaveformMode::PolarPhase];

    /// The JSON name.
    pub fn name(self) -> &'static str {
        match self {
            WaveformMode::Cart => "cart",
            WaveformMode::Polar => "polar",
            WaveformMode::PolarPhase => "polar_phase",
        }
    }
}

/// How the offsets of a qubit are sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coverage {
    /// Normally distributed about the offset.
    Single,
    /// Uniform across the sweep width.
    Broadband,
    /// Inside the band normally distributed, outside uniform; the target applies only inside the band.
    Selective,
    /// Inside and outside the band uniform; the target follows a super-Gaussian profile.
    BandSelective,
}

impl Coverage {
    /// Every coverage, in the order the interface lists them.
    pub const ALL: [Coverage; 4] = [
        Coverage::Single,
        Coverage::Broadband,
        Coverage::Selective,
        Coverage::BandSelective,
    ];

    /// The JSON name.
    pub fn name(self) -> &'static str {
        match self {
            Coverage::Single => "single",
            Coverage::Broadband => "broadband",
            Coverage::Selective => "selective",
            Coverage::BandSelective => "band_selective",
        }
    }
}

/// What each initial state should become.
#[derive(Clone, Debug, PartialEq)]
pub enum Targets {
    /// A product state, one axis per qubit, per initial state.
    Axis(Vec<Vec<String>>),
    /// A gate name per initial state; see [`Targets::single_gate`] for what one name for every state means.
    Gate(Vec<String>),
    /// A rotation about an axis per qubit (`x`, `-x`, `y`, `-y`, `z`, `-z`) by an angle in degrees, per initial state.
    PhiBeta {
        /// Rotation axes.
        phi: Vec<Vec<String>>,
        /// Rotation angles in degrees.
        beta: Vec<Vec<f64>>,
    },
}

impl Targets {
    /// The JSON key naming the method: `Axis`, `Gate` or `Phi`.
    pub fn method(&self) -> &'static str {
        match self {
            Targets::Axis(_) => "Axis",
            Targets::Gate(_) => "Gate",
            Targets::PhiBeta { .. } => "Phi",
        }
    }

    /// Number of target entries.
    pub fn len(&self) -> usize {
        match self {
            Targets::Axis(v) => v.len(),
            Targets::Gate(v) => v.len(),
            Targets::PhiBeta { phi, .. } => phi.len(),
        }
    }

    /// Whether there are no target entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The gate every initial state names, when they all name the same one.  Such a target asks for the gate
    /// itself, scored by its average gate fidelity over the computational basis; different gates for different
    /// initial states ask for each state to reach its gate's image.
    pub fn single_gate(&self) -> Option<&str> {
        match self {
            Targets::Gate(names) => {
                // By canonical name, so that `CNOT` and `CX` are the one gate they describe.
                let first = crate::setup::canonical_gate(names.first()?);
                names
                    .iter()
                    .all(|n| crate::setup::canonical_gate(n) == first)
                    .then_some(first)
            }
            _ => None,
        }
    }
}

impl Serialize for Targets {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Targets::Axis(v) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry("Axis", v)?;
                m.end()
            }
            Targets::Gate(v) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry("Gate", v)?;
                m.end()
            }
            Targets::PhiBeta { phi, beta } => {
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("Phi", phi)?;
                m.serialize_entry("Beta", beta)?;
                m.end()
            }
        }
    }
}

/// The raw target map.  The Python GUI can write every key, with nulls for the unused ones.
#[derive(Deserialize)]
struct RawTargets {
    #[serde(rename = "Axis", default)]
    axis: Option<Vec<Option<Vec<String>>>>,
    #[serde(rename = "Gate", default)]
    gate: Option<Vec<Option<String>>>,
    #[serde(rename = "Phi", default)]
    phi: Option<Vec<Option<Vec<String>>>>,
    #[serde(rename = "Beta", default)]
    beta: Option<Vec<Option<Vec<f64>>>>,
}

/// The entries of a list if it is present, non-empty and has no nulls.
fn complete<T>(v: Option<Vec<Option<T>>>) -> Option<Vec<T>> {
    let v = v?;
    if v.is_empty() {
        return None;
    }
    v.into_iter().collect()
}

impl<'de> Deserialize<'de> for Targets {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = RawTargets::deserialize(d)?;
        if let Some(axis) = complete(raw.axis) {
            return Ok(Targets::Axis(axis));
        }
        if let Some(gate) = complete(raw.gate) {
            return Ok(Targets::Gate(gate));
        }
        if let (Some(phi), Some(beta)) = (complete(raw.phi), complete(raw.beta)) {
            return Ok(Targets::PhiBeta { phi, beta });
        }
        Err(de::Error::custom(
            "target_states needs a complete \"Axis\", \"Gate\", or \"Phi\" and \"Beta\" list",
        ))
    }
}

/// Axis names for states.
pub const STATE_AXES: [&str; 6] = ["Z", "-Z", "X", "-X", "Y", "-Y"];
/// Axis names for rotations.
pub const ROTATION_AXES: [&str; 6] = ["x", "-x", "y", "-y", "z", "-z"];
/// Hamiltonian model names.
pub const HAMILTONIAN_TYPES: [&str; 3] = ["spin_chain", "superconducting", "duffing_transmon"];

impl Config {
    /// Parse a configuration.  Only the JSON syntax and field types are checked; see [`validate`](Self::validate).
    pub fn from_json(json: &str) -> Result<Config> {
        serde_json::from_str(json).map_err(|e| Error::Config(e.to_string()))
    }

    /// Serialise as pretty-printed JSON in the Python schema.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a Config always serialises")
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Whether T1/T2 relaxation is on.
    pub fn is_dissipative(&self) -> bool {
        self.optimization.dissipation_mode == Some(Dissipation::Dissipative)
    }

    /// The effective state space: dissipative runs are always in Liouville space.
    pub fn space(&self) -> Space {
        if self.is_dissipative() {
            Space::Liouville
        } else {
            self.optimization.space
        }
    }

    /// Every problem with this configuration, as sentences naming the field.  Empty means it can run.
    pub fn validate(&self) -> Vec<String> {
        validate::validate(self)
    }

    /// [`validate`](Self::validate) as a `Result`.
    pub fn check(&self) -> Result<()> {
        let problems = self.validate();
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Error::Config(problems.join("; ")))
        }
    }
}

mod validate;
