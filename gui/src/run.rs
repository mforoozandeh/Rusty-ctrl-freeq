//! Running an optimisation and getting the numbers back out.
//!
//! The optimiser is a blocking call, so the work happens off the interface's thread and reaches it as a stream of
//! [`RunMessage`]s.  How that stream is carried differs by platform - a thread natively, a Web Worker in the browser
//! - but [`run_to_sink`] is the same code on both sides.

use ctrl_freeq::analysis::Analysis;
use ctrl_freeq::config::Config;
use ctrl_freeq::optim::{IterationReport, ProgressSink};
use ctrl_freeq::run::RunResult;
use ctrl_freeq::setup::resolve_seed;
use serde::{Deserialize, Serialize};

/// Everything a finished run produced, with the configuration it ran - including the seed it used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Results {
    /// The configuration, with its seed filled in.
    pub config: Config,
    /// The optimisation.
    pub run: RunResult,
    /// Plot data.
    pub analysis: Analysis,
}

/// Everything the interface can be told by a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RunMessage {
    /// Transport detail: the Web Worker has installed its message handler and can be sent a configuration.  A
    /// worker still loading its WebAssembly drops anything posted to it, so the page waits for this.
    Ready,
    /// One iteration's numbers.
    Progress(IterationReport),
    /// The run finished; its results.
    Finished(Box<Results>),
    /// The run could not start, or failed.
    Failed(String),
}

/// Somewhere for [`run_to_sink`] to put messages.  Natively a channel; in the browser, the page.
pub trait MessageSink {
    /// Hand over one message.
    fn send(&mut self, message: RunMessage);
    /// Whether the interface has asked the run to stop.
    fn cancelled(&self) -> bool {
        false
    }
}

/// Bridge from the optimiser's progress reports to a [`MessageSink`].
struct Bridge<'a>(&'a mut dyn MessageSink);

impl ProgressSink for Bridge<'_> {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.0.send(RunMessage::Progress(report.clone()));
    }
    fn should_cancel(&self) -> bool {
        self.0.cancelled()
    }
}

/// Run `config` and analyse the result, reporting into `sink`.
///
/// Always ends with exactly one [`RunMessage::Finished`] or [`RunMessage::Failed`].  A configuration without a seed
/// is given one first, so the results always say how to repeat the run.
pub fn run_to_sink(config: &Config, sink: &mut dyn MessageSink) {
    let mut config = config.clone();
    if config.seed.is_none() {
        config.seed = Some(resolve_seed(&config));
    }
    let outcome = ctrl_freeq::run(&config, &mut Bridge(sink)).and_then(|run| {
        let analysis = ctrl_freeq::analyse(&config, &run)?;
        Ok(Results {
            config: config.clone(),
            run,
            analysis,
        })
    });
    match outcome {
        Ok(results) => sink.send(RunMessage::Finished(Box::new(results))),
        Err(e) => sink.send(RunMessage::Failed(e.to_string())),
    }
}
