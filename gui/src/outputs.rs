//! The central panel: convergence while a run is going, then the pulses, the dynamics and the excitation profile.

use ctrl_freeq::optim::IterationReport;
use eframe::egui::{self, Color32, RichText, Ui};
use egui_plot::{FilledArea, Legend, Line, Plot, PlotPoints};

use crate::run::Results;

/// Which tab is showing, and the choices made in it.
#[derive(Debug, Clone, Default)]
pub struct View {
    tab: Tab,
    log_infidelity: bool,
    initial_state: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Tab {
    #[default]
    Convergence,
    Pulses,
    Dynamics,
    Profile,
}

const X_COLOUR: Color32 = Color32::from_rgb(66, 133, 244);
const Y_COLOUR: Color32 = Color32::from_rgb(234, 134, 38);
const Z_COLOUR: Color32 = Color32::from_rgb(52, 168, 83);
const XYZ: [(&str, Color32); 3] = [("<X>", X_COLOUR), ("<Y>", Y_COLOUR), ("<Z>", Z_COLOUR)];
const PLOT_HEIGHT: f32 = 190.0;

/// Show the results area.
pub fn show(ui: &mut Ui, view: &mut View, history: &[IterationReport], results: Option<&Results>) {
    ui.horizontal(|ui| {
        ui.selectable_value(&mut view.tab, Tab::Convergence, "Convergence");
        ui.add_enabled_ui(results.is_some(), |ui| {
            ui.selectable_value(&mut view.tab, Tab::Pulses, "Pulses");
            ui.selectable_value(&mut view.tab, Tab::Dynamics, "Dynamics");
            ui.selectable_value(&mut view.tab, Tab::Profile, "Excitation profile");
        });
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| match (view.tab, results) {
        (Tab::Convergence, _) => convergence(ui, view, history),
        (Tab::Pulses, Some(r)) => pulses(ui, r),
        (Tab::Dynamics, Some(r)) => dynamics(ui, view, r),
        (Tab::Profile, Some(r)) => profile(ui, view, r),
        _ => waiting(ui, "Run an optimisation to see results here."),
    });
}

fn waiting(ui: &mut Ui, text: &str) {
    ui.add_space(40.0);
    ui.vertical_centered(|ui| ui.label(RichText::new(text).weak()));
}

fn convergence(ui: &mut Ui, view: &mut View, history: &[IterationReport]) {
    if history.is_empty() {
        waiting(
            ui,
            "Press Run: fidelity per iteration appears here as the optimiser works.",
        );
        return;
    }
    ui.checkbox(&mut view.log_infidelity, "Show log₁₀(1 − fidelity)");
    let points: PlotPoints = history
        .iter()
        .map(|r| {
            let y = if view.log_infidelity {
                (1.0 - r.fidelity).max(1e-16).log10()
            } else {
                r.fidelity
            };
            [r.iteration as f64, y]
        })
        .collect();
    // The best so far reads better than the raw trace for derivative-free methods, whose every evaluation counts.
    let mut best = f64::NEG_INFINITY;
    let best_points: PlotPoints = history
        .iter()
        .map(|r| {
            best = best.max(r.fidelity);
            let y = if view.log_infidelity {
                (1.0 - best).max(1e-16).log10()
            } else {
                best
            };
            [r.iteration as f64, y]
        })
        .collect();
    let label = if view.log_infidelity {
        "log₁₀(1 − F)"
    } else {
        "fidelity"
    };
    Plot::new("convergence")
        .height(ui.available_height().max(300.0) - 10.0)
        .x_axis_label("iteration")
        .y_axis_label(label)
        .legend(Legend::default())
        .show(ui, |p| {
            p.line(
                Line::new("each iteration", points)
                    .width(1.0)
                    .color(X_COLOUR.gamma_multiply(0.6)),
            );
            p.line(Line::new("best so far", best_points).width(2.0).color(Z_COLOUR));
        });
}

fn series(xs: &[f64], ys: &[f64]) -> PlotPoints<'static> {
    xs.iter().zip(ys).map(|(&x, &y)| [x, y]).collect()
}

fn pulses(ui: &mut Ui, r: &Results) {
    let a = &r.analysis;
    ui.label(RichText::new("Amplitudes are fractions of each qubit's maximum Rabi frequency.").weak());
    for p in &a.pulses {
        ui.add_space(6.0);
        ui.label(RichText::new(format!("Qubit {}", p.qubit + 1)).strong());
        ui.columns(3, |cols| {
            Plot::new(("iq", p.qubit))
                .height(PLOT_HEIGHT)
                .legend(Legend::default())
                .x_axis_label("t (ns)")
                .link_axis("pulses", [true, false])
                .show(&mut cols[0], |plot| {
                    plot.line(Line::new("I (cx)", series(&a.times_ns, &p.cx)).color(X_COLOUR));
                    plot.line(Line::new("Q (cy)", series(&a.times_ns, &p.cy)).color(Y_COLOUR));
                });
            Plot::new(("amp", p.qubit))
                .height(PLOT_HEIGHT)
                .legend(Legend::default())
                .x_axis_label("t (ns)")
                .link_axis("pulses", [true, false])
                .show(&mut cols[1], |plot| {
                    plot.line(Line::new("amplitude", series(&a.times_ns, &p.amp)).color(Z_COLOUR));
                });
            Plot::new(("phase", p.qubit))
                .height(PLOT_HEIGHT)
                .legend(Legend::default())
                .x_axis_label("t (ns)")
                .link_axis("pulses", [true, false])
                .show(&mut cols[2], |plot| {
                    plot.line(Line::new("phase (rad)", series(&a.times_ns, &p.phase)));
                });
        });
    }
}

/// A picker for the initial state, if there is more than one.
fn initial_state_picker(ui: &mut Ui, view: &mut View, r: &Results) {
    let n = r.config.initial_states.len();
    view.initial_state = view.initial_state.min(n.saturating_sub(1));
    if n > 1 {
        ui.horizontal(|ui| {
            ui.label("Initial state");
            for (i, s) in r.config.initial_states.iter().enumerate() {
                ui.selectable_value(&mut view.initial_state, i, format!("{} ({})", i + 1, s.join(", ")));
            }
        });
    }
}

fn dynamics(ui: &mut Ui, view: &mut View, r: &Results) {
    initial_state_picker(ui, view, r);
    let a = &r.analysis;
    let Some(d) = a.dynamics.get(view.initial_state) else {
        return;
    };
    let t = &a.state_times_ns;
    ui.label(RichText::new("Observables under the mean drift; the band spans the batch snapshots.").weak());
    for (q, obs) in d.observables.iter().enumerate() {
        Plot::new(("obs", q))
            .height(PLOT_HEIGHT)
            .legend(Legend::default())
            .x_axis_label("t (ns)")
            .y_axis_label(format!("qubit {}", q + 1))
            .include_y(-1.05)
            .include_y(1.05)
            .link_axis("dynamics", [true, false])
            .show(ui, |plot| {
                for (k, (name, colour)) in XYZ.iter().enumerate() {
                    // Named like its line, so the legend shows one entry that toggles both.
                    let band = FilledArea::new(*name, t, &d.observables_min[q][k], &d.observables_max[q][k])
                        .fill_color(colour.gamma_multiply(0.18));
                    plot.add(band);
                    plot.line(Line::new(*name, series(t, &obs[k])).color(*colour).width(2.0));
                }
            });
    }
    ui.add_space(8.0);
    ui.label(RichText::new("State components: mean drift in bold, batch snapshots faint.").weak());
    let columns = if d.labels.len() > 4 { 2 } else { 1 };
    let chunks: Vec<Vec<usize>> = (0..d.labels.len())
        .collect::<Vec<_>>()
        .chunks(columns)
        .map(|c| c.to_vec())
        .collect();
    for chunk in chunks {
        ui.columns(columns, |cols| {
            for (col, &c) in cols.iter_mut().zip(&chunk) {
                Plot::new(("component", c))
                    .height(PLOT_HEIGHT * 0.8)
                    .legend(Legend::default())
                    .x_axis_label("t (ns)")
                    .y_axis_label(d.labels[c].as_str())
                    .link_axis("dynamics", [true, false])
                    .show(col, |plot| {
                        for snapshot in &d.snapshots {
                            let re: Vec<f64> = snapshot[c].iter().map(|z| z[0]).collect();
                            let im: Vec<f64> = snapshot[c].iter().map(|z| z[1]).collect();
                            plot.line(
                                Line::new("", series(t, &re))
                                    .color(X_COLOUR.gamma_multiply(0.15))
                                    .allow_hover(false),
                            );
                            plot.line(
                                Line::new("", series(t, &im))
                                    .color(Y_COLOUR.gamma_multiply(0.15))
                                    .allow_hover(false),
                            );
                        }
                        let re: Vec<f64> = d.mean[c].iter().map(|z| z[0]).collect();
                        let im: Vec<f64> = d.mean[c].iter().map(|z| z[1]).collect();
                        plot.line(Line::new("Re", series(t, &re)).color(X_COLOUR).width(2.0));
                        plot.line(Line::new("Im", series(t, &im)).color(Y_COLOUR).width(2.0));
                    });
            }
        });
    }
}

fn profile(ui: &mut Ui, view: &mut View, r: &Results) {
    initial_state_picker(ui, view, r);
    let Some(p) = r.analysis.profiles.get(view.initial_state) else {
        return;
    };
    ui.label(RichText::new("Final <X>, <Y>, <Z> as the offsets sweep 1.5 sweep widths, all qubits together.").weak());
    for (q, xyz) in p.xyz.iter().enumerate() {
        let mhz: Vec<f64> = p.offsets_hz[q].iter().map(|v| v / 1e6).collect();
        Plot::new(("profile", q))
            .height(PLOT_HEIGHT)
            .legend(Legend::default())
            .x_axis_label(format!("qubit {} offset (MHz)", q + 1))
            .include_y(-1.05)
            .include_y(1.05)
            .show(ui, |plot| {
                for (k, (name, colour)) in XYZ.iter().enumerate() {
                    plot.line(Line::new(*name, series(&mhz, &xyz[k])).color(*colour).width(2.0));
                }
            });
    }
}
