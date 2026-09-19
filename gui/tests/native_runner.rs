//! The native run pipeline end to end: configuration in, progress and one final message out.

use std::time::{Duration, Instant};

use ctrl_freeq::config::Config;
use ctrl_freeq::optim::Exit;
use ctrl_freeq_gui::presets::presets;
use ctrl_freeq_gui::run::{Results, RunMessage};
use ctrl_freeq_gui::runner::Runner;

fn small() -> Config {
    let mut c = presets()
        .into_iter()
        .find(|(n, _)| n == "single qubit parameters multiple initial targ")
        .unwrap()
        .1;
    c.parameters.point_in_pulse = vec![40];
    c.optimization.algorithm = "l-bfgs".into();
    c
}

/// Drain `runner` until its final message, or fail after a minute.
fn finish(mut runner: Runner, mut on_progress: impl FnMut(&mut Runner, usize)) -> (usize, RunMessage) {
    let started = Instant::now();
    let mut progress = 0;
    loop {
        for m in runner.drain() {
            match m {
                RunMessage::Progress(_) => {
                    progress += 1;
                    on_progress(&mut runner, progress);
                }
                RunMessage::Ready => {}
                last => return (progress, last),
            }
        }
        assert!(started.elapsed() < Duration::from_secs(60), "the run did not finish");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_run_reports_progress_then_finishes_with_analysis() {
    let (progress, last) = finish(Runner::start(small()), |_, _| {});
    let RunMessage::Finished(r) = last else {
        panic!("expected Finished, got {last:?}");
    };
    assert!(progress > 0);
    assert!(r.config.seed.is_some(), "the seed that was used is recorded");
    assert_eq!(r.analysis.pulses.len(), 1);
    assert_eq!(r.analysis.dynamics.len(), 3);
    assert!(r.run.fidelity > 0.9, "fidelity {}", r.run.fidelity);
}

#[test]
fn a_cancelled_run_still_delivers_its_best_pulse() {
    let mut c = small();
    c.optimization.targ_fid = 1.0;
    c.optimization.max_iter = 100_000;
    c.optimization.algorithm = "cobyla".into();
    let (_, last) = finish(Runner::start(c), |runner, n| {
        if n == 5 {
            runner.cancel();
        }
    });
    let RunMessage::Finished(r) = last else {
        panic!("expected Finished, got {last:?}");
    };
    assert_eq!(r.run.exit, Exit::Cancelled);
    assert!(!r.analysis.pulses.is_empty());
}

#[test]
fn an_invalid_configuration_fails_cleanly() {
    let mut c = small();
    c.parameters.n_para = vec![3];
    let (_, last) = finish(Runner::start(c), |_, _| {});
    assert!(
        matches!(last, RunMessage::Failed(ref e) if e.contains("n_para")),
        "{last:?}"
    );
}

#[test]
fn every_preset_validates() {
    for (name, c) in presets() {
        assert!(c.validate().is_empty(), "{name}: {:?}", c.validate());
    }
}

#[test]
fn results_survive_json_as_the_worker_sends_them() {
    let (_, last) = finish(Runner::start(small()), |_, _| {});
    let json = serde_json::to_string(&last).unwrap();
    let back: RunMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(back, last);
    let RunMessage::Finished(r) = back else { unreachable!() };
    let _: &Results = &r;
}
