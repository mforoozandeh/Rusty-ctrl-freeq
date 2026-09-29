//! The central panel: convergence while a run is going, then the pulses, the dynamics, the excitation profile and,
//! for a run with a Rabi spread, the Rabi map.

use ctrl_freeq::analysis::Dynamics;
use ctrl_freeq::optim::IterationReport;
use eframe::egui::{self, Color32, ColorImage, RichText, TextureHandle, TextureOptions, Ui};
use egui_plot::{FilledArea, Legend, Line, Plot, PlotImage, PlotPoint, PlotPoints};

use crate::run::Results;

/// Which tab is showing, and the choices made in it.
pub struct View {
    tab: Tab,
    log_infidelity: bool,
    initial_state: usize,
    /// Which of `⟨X⟩, ⟨Y⟩, ⟨Z⟩` the Rabi maps show.
    map_axis: usize,
    /// The Rabi maps as images, one per qubit, with the initial state and axis they show.
    map_textures: Option<((usize, usize), Vec<TextureHandle>)>,
}

impl Default for View {
    fn default() -> Self {
        View {
            tab: Tab::default(),
            log_infidelity: false,
            initial_state: 0,
            map_axis: 2,
            map_textures: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Tab {
    #[default]
    Convergence,
    Pulses,
    Dynamics,
    Profile,
    RabiMap,
}

const X_COLOUR: Color32 = Color32::from_rgb(66, 133, 244);
const Y_COLOUR: Color32 = Color32::from_rgb(234, 134, 38);
const Z_COLOUR: Color32 = Color32::from_rgb(52, 168, 83);
const XYZ: [(&str, Color32); 3] = [("<X>", X_COLOUR), ("<Y>", Y_COLOUR), ("<Z>", Z_COLOUR)];
const PLOT_HEIGHT: f32 = 190.0;
const GUIDE_COLOUR: Color32 = Color32::from_gray(40);

/// Show the results area.
pub fn show(ui: &mut Ui, view: &mut View, history: &[IterationReport], results: Option<&Results>) {
    ui.horizontal(|ui| {
        ui.selectable_value(&mut view.tab, Tab::Convergence, "Convergence");
        ui.add_enabled_ui(results.is_some(), |ui| {
            ui.selectable_value(&mut view.tab, Tab::Pulses, "Pulses");
            ui.selectable_value(&mut view.tab, Tab::Dynamics, "Dynamics");
            ui.selectable_value(&mut view.tab, Tab::Profile, "Excitation profile");
            if results.is_some_and(|r| !r.analysis.rabi_maps.is_empty()) {
                ui.selectable_value(&mut view.tab, Tab::RabiMap, "Rabi map");
            }
        });
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| match (view.tab, results) {
        (Tab::Convergence, _) => convergence(ui, view, history),
        (Tab::Pulses, Some(r)) => pulses(ui, r),
        (Tab::Dynamics, Some(r)) => dynamics(ui, view, r),
        (Tab::Profile, Some(r)) => profile(ui, view, r),
        (Tab::RabiMap, Some(r)) => rabi_map(ui, view, r),
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

/// Whether a leakage plot has anything to show: two-level models sit at zero throughout, while the mean drift can
/// leak where the sampled snapshots do not, so both it and the snapshots count.
fn leaks(d: &Dynamics) -> bool {
    d.leakage.iter().chain(&d.leakage_max).any(|&l| l > 1e-9)
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
    if leaks(d) {
        ui.add_space(8.0);
        ui.label(RichText::new("Population outside the computational subspace.").weak());
        Plot::new("leakage")
            .height(PLOT_HEIGHT)
            .legend(Legend::default())
            .x_axis_label("t (ns)")
            .y_axis_label("leakage")
            .include_y(0.0)
            .link_axis("dynamics", [true, false])
            .show(ui, |plot| {
                let band = FilledArea::new("leakage", t, &d.leakage_min, &d.leakage_max)
                    .fill_color(Y_COLOUR.gamma_multiply(0.18));
                plot.add(band);
                plot.line(Line::new("leakage", series(t, &d.leakage)).color(Y_COLOUR).width(2.0));
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

fn rabi_map(ui: &mut Ui, view: &mut View, r: &Results) {
    initial_state_picker(ui, view, r);
    let Some(m) = r.analysis.rabi_maps.get(view.initial_state) else {
        waiting(ui, "The Rabi map needs a Rabi spread and more than one Rabi snapshot.");
        return;
    };
    ui.horizontal(|ui| {
        for (k, (name, _)) in XYZ.iter().enumerate() {
            ui.selectable_value(&mut view.map_axis, k, *name);
        }
    });
    let shown = (view.initial_state, view.map_axis);
    if view.map_textures.as_ref().is_none_or(|(drawn, _)| *drawn != shown) {
        let textures = m
            .xyz
            .iter()
            .enumerate()
            .map(|(q, xyz)| {
                let image = colour_map(&xyz[view.map_axis], m.offsets_hz[q].len());
                ui.ctx()
                    .load_texture(format!("rabi-map-{q}"), image, TextureOptions::LINEAR)
            })
            .collect();
        view.map_textures = Some((shown, textures));
    }
    let Some((_, textures)) = &view.map_textures else {
        return;
    };
    ui.label(
        RichText::new(format!(
            "Final {} as the offsets and the Rabi frequencies sweep, all qubits together: blue −1, white 0, red +1.  \
             The box is the sweep width and the Rabi frequencies drawn for the optimisation.",
            XYZ[view.map_axis].0
        ))
        .weak(),
    );
    let (Some(&y0), Some(&y1)) = (m.scales.first(), m.scales.last()) else {
        return;
    };
    for (q, texture) in textures.iter().enumerate() {
        let mhz: Vec<f64> = m.offsets_hz[q].iter().map(|v| v / 1e6).collect();
        let (Some(&x0), Some(&x1)) = (mhz.first(), mhz.last()) else {
            continue;
        };
        // The offsets span one and a half sweep widths.
        let (centre, half_sweep) = ((x0 + x1) / 2.0, (x1 - x0) / 3.0);
        let edges = [centre - half_sweep, centre + half_sweep];
        Plot::new(("rabi-map", q))
            .height(PLOT_HEIGHT * 1.5)
            .legend(Legend::default())
            .x_axis_label(format!("qubit {} offset (MHz)", q + 1))
            .y_axis_label("Ω / Ω max")
            .show(ui, |plot| {
                plot.image(PlotImage::new(
                    "",
                    texture.id(),
                    PlotPoint::new(centre, (y0 + y1) / 2.0),
                    egui::vec2((x1 - x0) as f32, (y1 - y0) as f32),
                ));
                for edge in edges {
                    plot.line(Line::new("sweep width", series(&[edge, edge], &[y0, y1])).color(GUIDE_COLOUR));
                }
                for rabi in m.drawn[q] {
                    plot.line(Line::new("Rabi drawn", series(&edges, &[rabi, rabi])).color(GUIDE_COLOUR));
                }
            });
    }
}

/// `values` in `[-1, 1]`, `cols` to a row and the rows from the bottom up, on a blue-white-red scale.
fn colour_map(values: &[f64], cols: usize) -> ColorImage {
    let rows: Vec<&[f64]> = values.chunks(cols.max(1)).collect();
    let pixels = rows
        .iter()
        .rev()
        .flat_map(|row| row.iter().map(|&v| diverging(v)))
        .collect();
    ColorImage::new([cols, rows.len()], pixels)
}

fn diverging(v: f64) -> Color32 {
    const NEGATIVE: [f64; 3] = [59.0, 76.0, 192.0];
    const MIDDLE: [f64; 3] = [235.0, 235.0, 235.0];
    const POSITIVE: [f64; 3] = [180.0, 4.0, 38.0];
    let t = v.clamp(-1.0, 1.0);
    let (end, u) = if t < 0.0 { (NEGATIVE, -t) } else { (POSITIVE, t) };
    let channel = |k: usize| (MIDDLE[k] + (end[k] - MIDDLE[k]) * u).round() as u8;
    Color32::from_rgb(channel(0), channel(1), channel(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamics_with(leakage: Vec<f64>, leakage_max: Vec<f64>) -> Dynamics {
        Dynamics {
            initial_state: 0,
            labels: vec![],
            mean: vec![],
            snapshots: vec![],
            observables: vec![],
            observables_min: vec![],
            observables_max: vec![],
            leakage_min: vec![0.0; leakage.len()],
            leakage,
            leakage_max,
        }
    }

    /// The mean drift can leak where every sampled snapshot stays put; the plot still belongs on screen.
    #[test]
    fn leakage_under_the_mean_drift_alone_is_plotted() {
        assert!(leaks(&dynamics_with(vec![0.0, 0.003_6], vec![0.0, 5.11e-11])));
        assert!(leaks(&dynamics_with(vec![0.0, 0.0], vec![0.0, 0.002])));
        assert!(!leaks(&dynamics_with(vec![0.0, 0.0], vec![0.0, 0.0])));
    }

    #[test]
    fn the_colour_scale_runs_blue_white_red_and_clamps() {
        assert_eq!(diverging(-1.0), Color32::from_rgb(59, 76, 192));
        assert_eq!(diverging(0.0), Color32::from_rgb(235, 235, 235));
        assert_eq!(diverging(1.0), Color32::from_rgb(180, 4, 38));
        assert_eq!(diverging(-3.0), diverging(-1.0));
        assert_eq!(diverging(1.0 + 1e-12), diverging(1.0));
    }

    /// Map rows run from the lowest Rabi frequency up; an image's first row is its top.
    #[test]
    fn the_highest_rabi_frequency_is_the_top_row() {
        let image = colour_map(&[-1.0, -1.0, -1.0, 1.0, 1.0, 0.0], 3);
        assert_eq!(image.size, [3, 2]);
        assert_eq!(image.pixels[..3], [diverging(1.0), diverging(1.0), diverging(0.0)]);
        assert!(image.pixels[3..].iter().all(|&p| p == diverging(-1.0)));
    }
}
