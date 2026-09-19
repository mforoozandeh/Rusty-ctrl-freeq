//! Getting results out: CSV and JSON, and handing the user a file - a save dialog natively, a download in the
//! browser.

use std::fmt::Write as _;

use ctrl_freeq::analysis::{Analysis, PulseTrace};
use ctrl_freeq::config::Config;
use ctrl_freeq::run::RunResult;
use serde::Serialize;

use crate::run::Results;

/// The pulses as CSV: time in ns, then for each qubit its in-phase and quadrature components, amplitude and phase.
/// The components are fractions of the qubit's maximum Rabi frequency; the phase is in radians.
pub fn waveform_csv(analysis: &Analysis) -> String {
    let mut out = String::from("t_ns");
    for p in &analysis.pulses {
        let q = p.qubit + 1;
        let _ = write!(out, ",q{q}_cx,q{q}_cy,q{q}_amp,q{q}_phase");
    }
    out.push('\n');
    for (t, time) in analysis.times_ns.iter().enumerate() {
        let _ = write!(out, "{time:.6}");
        for p in &analysis.pulses {
            let _ = write!(
                out,
                ",{:.12},{:.12},{:.12},{:.12}",
                p.cx[t], p.cy[t], p.amp[t], p.phase[t]
            );
        }
        out.push('\n');
    }
    out
}

/// What the results file holds: the run and its pulses, without the bulky dynamics.
#[derive(Serialize)]
struct ResultsFile<'a> {
    config: &'a Config,
    run: &'a RunResult,
    times_ns: &'a [f64],
    pulses: &'a [PulseTrace],
}

/// The results as pretty-printed JSON.
pub fn results_json(results: &Results) -> String {
    let file = ResultsFile {
        config: &results.config,
        run: &results.run,
        times_ns: &results.analysis.times_ns,
        pulses: &results.analysis.pulses,
    };
    serde_json::to_string_pretty(&file).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
}

/// Hand `contents` to the user as a file called `filename`.
#[cfg(not(target_arch = "wasm32"))]
pub fn offer_file(filename: &str, contents: &str) -> Result<(), String> {
    match rfd::FileDialog::new().set_file_name(filename).save_file() {
        Some(path) => std::fs::write(&path, contents).map_err(|e| format!("could not write {}: {e}", path.display())),
        None => Ok(()),
    }
}

/// Hand `contents` to the user as a download called `filename`.
#[cfg(target_arch = "wasm32")]
pub fn offer_file(filename: &str, contents: &str) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let go = || -> Option<()> {
        let document = web_sys::window()?.document()?;
        let parts = js_sys::Array::new();
        parts.push(&wasm_bindgen::JsValue::from_str(contents));
        let options = web_sys::BlobPropertyBag::new();
        options.set_type("text/plain;charset=utf-8");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).ok()?;
        let url = web_sys::Url::create_object_url_with_blob(&blob).ok()?;
        let anchor: web_sys::HtmlAnchorElement = document.create_element("a").ok()?.dyn_into().ok()?;
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.click();
        let _ = web_sys::Url::revoke_object_url(&url);
        Some(())
    };
    go().ok_or_else(|| format!("could not offer {filename} as a download"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_qubit_analysis(points: usize) -> Analysis {
        let trace = |qubit| PulseTrace {
            qubit,
            cx: vec![0.5; points],
            cy: vec![-0.25; points],
            amp: vec![0.56; points],
            phase: vec![-0.46; points],
        };
        Analysis {
            times_ns: (0..points).map(|t| t as f64 * 2.0).collect(),
            pulses: vec![trace(0), trace(1)],
            dynamics: vec![],
            profiles: vec![],
        }
    }

    #[test]
    fn the_csv_has_a_header_and_one_row_per_time_point() {
        let csv = waveform_csv(&two_qubit_analysis(7));
        let lines: Vec<&str> = csv.trim_end().lines().collect();
        assert_eq!(lines.len(), 8);
        assert_eq!(lines[0], "t_ns,q1_cx,q1_cy,q1_amp,q1_phase,q2_cx,q2_cy,q2_amp,q2_phase");
        assert!(lines[1..].iter().all(|l| l.split(',').count() == 9));
        assert!(lines[2].starts_with("2.000000,0.5"));
    }
}
