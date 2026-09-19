//! One call from a configuration to an optimised pulse.

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::Result;
use crate::objective::CostModel;
use crate::optim::{Exit, IterationReport, ProgressSink, RunControl, optimizer};
use crate::setup::{Problem, resolve_seed};
use crate::time::Instant;

/// What a run produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    /// The seed every random quantity was drawn from; put it in the configuration to repeat the run exactly.
    pub seed: u64,
    /// The optimiser used.
    pub algorithm: String,
    /// The optimised parameters.
    pub solution: Vec<f64>,
    /// Mean fidelity at the solution.
    pub fidelity: f64,
    /// Amplitude penalty at the solution.
    pub penalty: f64,
    /// Fidelity at each reported iteration.
    pub fidelity_history: Vec<f64>,
    /// Iterations completed.
    pub iterations: usize,
    /// Objective evaluations.
    pub evaluations: usize,
    /// Why the run ended.
    pub exit: Exit,
    /// Wall-clock seconds.
    pub elapsed_s: f64,
    /// Things the user should know, such as a request for a GPU this version cannot honour.
    pub notices: Vec<String>,
    /// Per qubit, the largest amplitude of the optimised pulse as a fraction of that qubit's `Omega_R_max`.
    /// Anything above 1 asks for more than the configured maximum Rabi frequency: the limit is penalised in the
    /// cost, not enforced, so a result can exceed it.
    #[serde(default)]
    pub peak_amplitude: Vec<f64>,
}

/// What to tell the user when a pulse asks for more than the maximum Rabi frequency it was given.
fn amplitude_notice(peaks: &[f64]) -> Option<String> {
    let (q, &peak) = peaks
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .filter(|&(_, &p)| p > 1.0)?;
    Some(format!(
        "The pulse exceeds the maximum Rabi frequency: qubit {}'s amplitude peaks at {peak:.2} of it.  The limit \
         is penalised in the cost, not enforced.",
        q + 1
    ))
}

/// Number of threads a run uses when the configuration does not set `cpu_cores`.
pub fn available_threads() -> usize {
    crate::parallel::available_threads()
}

/// Passes reports on and keeps the fidelity history.
struct History<'a> {
    inner: &'a mut dyn ProgressSink,
    fidelities: Vec<f64>,
}

impl ProgressSink for History<'_> {
    fn on_iteration(&mut self, r: &IterationReport) {
        self.fidelities.push(r.fidelity);
        self.inner.on_iteration(r);
    }
    fn should_cancel(&self) -> bool {
        self.inner.should_cancel()
    }
}

/// Optimise the pulse `cfg` describes, reporting each iteration to `sink`.
///
/// The configuration's seed is used if it has one; otherwise one is drawn and returned in the result.
pub fn run(cfg: &Config, sink: &mut dyn ProgressSink) -> Result<RunResult> {
    let started = Instant::now();
    let seed = resolve_seed(cfg);
    let problem = Problem::build_with_seed(cfg, seed)?;
    let x0 = problem.x0.clone();
    let mut model = CostModel::new(problem);
    if let Some(threads) = cfg.cpu_cores {
        model = model.with_threads(threads)?;
    }
    let opt = optimizer(&cfg.optimization.algorithm)?;
    let mut notices = Vec::new();
    if cfg.compute_resource.as_deref() == Some("gpu") {
        notices.push("GPU is not supported; running on CPU.".to_string());
    }
    if let Some(gate) = cfg.target_states.single_gate() {
        notices.push(format!(
            "The fidelity is the {gate} gate's average gate fidelity over the computational basis; the initial \
             states only set what the dynamics plots show."
        ));
    }
    let mut history = History {
        inner: sink,
        fidelities: Vec::new(),
    };
    let mut ctl = RunControl {
        max_iter: cfg.optimization.max_iter,
        target_fidelity: cfg.optimization.targ_fid,
        sink: &mut history,
    };
    let result = opt.minimize(&model, &x0, &mut ctl)?;
    let amp = model.waveforms(&result.x)?.amp;
    let peak_amplitude: Vec<f64> = (0..amp.cols)
        .map(|q| (0..amp.rows).fold(0.0f64, |m, t| m.max(amp.get(t, q))))
        .collect();
    notices.extend(amplitude_notice(&peak_amplitude));
    Ok(RunResult {
        seed,
        algorithm: opt.name().to_string(),
        solution: result.x,
        fidelity: result.eval.fidelity,
        penalty: result.eval.penalty,
        fidelity_history: history.fidelities,
        iterations: result.iterations,
        evaluations: result.evaluations,
        exit: result.exit,
        elapsed_s: started.elapsed().as_secs_f64(),
        notices,
        peak_amplitude,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::examples;
    use crate::optim::NoProgress;

    #[test]
    fn a_run_reports_the_peak_amplitude_of_its_pulse() {
        let mut cfg = Config::from_json(examples()[0].1).unwrap();
        cfg.seed = Some(5);
        cfg.optimization.max_iter = 3;
        let r = run(&cfg, &mut NoProgress).unwrap();
        let model = CostModel::new(Problem::build_with_seed(&cfg, r.seed).unwrap());
        let amp = model.waveforms(&r.solution).unwrap().amp;
        let want = (0..amp.rows).fold(0.0f64, |m, t| m.max(amp.get(t, 0)));
        assert_eq!(r.peak_amplitude, vec![want]);
    }

    /// The amplitude limit is penalised, not enforced, so a pulse that exceeds it says so.
    #[test]
    fn exceeding_the_maximum_rabi_frequency_is_reported() {
        assert_eq!(amplitude_notice(&[0.5, 1.0]), None);
        let notice = amplitude_notice(&[0.5, 1.07]).expect("a notice");
        assert!(notice.contains("qubit 2") && notice.contains("1.07"), "{notice}");
    }
}
