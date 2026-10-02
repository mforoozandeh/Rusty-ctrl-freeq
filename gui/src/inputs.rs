//! The left panel: every field of the configuration, grouped as the Python GUI groups them.
//!
//! Values are stored, shown and typed in SI units, as in the JSON file: Hz and s, written as [`numbers::format`]
//! writes them, so that a nanosecond pulse and a millisecond one are edited alike.

use ctrl_freeq::basis::{basis, basis_names, envelope_names};
use ctrl_freeq::config::{Config, Coverage, Parameters, ROTATION_AXES, STATE_AXES, Space, Targets, WaveformMode};
use ctrl_freeq::hamiltonian::model_names;
use ctrl_freeq::optim::optimizer_names;
use ctrl_freeq::setup::gate_names;
use eframe::egui::{self, DragValue, RichText, Ui};

use crate::edit::{self, TargetMethod, coupling_types};
use crate::numbers;
use crate::platform::Platform;

/// Most qubits the interface offers: the state space grows as 2ⁿ (3ⁿ for transmons with leakage).
const MAX_QUBITS: usize = 4;
/// In the browser everything runs on one thread, so the ceiling is lower.
const MAX_WEB_QUBITS: usize = 3;

/// Show the editor for `cfg`.
pub fn show(ui: &mut Ui, cfg: &mut Config, platform: Platform) {
    let problems = cfg.validate();
    if !problems.is_empty() {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(
                RichText::new("Fix before running:")
                    .strong()
                    .color(ui.visuals().error_fg_color),
            );
            for p in &problems {
                ui.label(RichText::new(format!("• {p}")).color(ui.visuals().error_fg_color));
            }
        });
        ui.add_space(4.0);
    }
    system(ui, cfg, platform);
    pulse(ui, cfg);
    qubits(ui, cfg);
    if cfg.n_qubits() > 1 {
        coupling(ui, cfg);
    }
    targets(ui, cfg);
    optimisation(ui, cfg, platform);
}

/// A drag-or-type number field.  What is typed is stored on Enter or when the field is left, not at every key: on
/// the way to 200 the field would otherwise hold 2 and then 20, and each would be checked and acted on.  Escape
/// abandons the edit.
fn number<'a>(value: &'a mut impl egui::emath::Numeric) -> DragValue<'a> {
    DragValue::new(value).update_while_editing(false)
}

/// A drag-or-type field for a quantity in SI `unit`s, shown and read as [`numbers::format`] writes it, so `4e7` and
/// `2e-7` can be typed.
///
/// Dragging moves the value by half a percent of itself per point, so a nanosecond and a millisecond pulse drag
/// alike; `at_zero` is the step while the value is zero.
fn si(ui: &mut Ui, value: &mut f64, unit: &str, at_zero: f64) -> egui::Response {
    let speed = if *value == 0.0 { at_zero } else { value.abs() * 0.005 };
    ui.add(
        number(value)
            .speed(speed)
            .suffix(format!(" {unit}"))
            .custom_formatter(|v, _| numbers::format(v)),
    )
}

/// A frequency in Hz; from zero it drags a kilohertz at a time.
fn hz(ui: &mut Ui, value: &mut f64) -> egui::Response {
    si(ui, value, "Hz", 1e3)
}

/// A time in seconds; from zero it drags a nanosecond at a time.
fn seconds(ui: &mut Ui, value: &mut f64) -> egui::Response {
    si(ui, value, "s", 1e-9)
}

fn combo<T: Clone + PartialEq>(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &mut T,
    options: &[(T, &str)],
) {
    let current = options.iter().find(|(v, _)| v == value).map_or("?", |(_, l)| *l);
    egui::ComboBox::from_id_salt(id)
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (v, label) in options {
                ui.selectable_value(value, v.clone(), *label);
            }
        });
}

fn string_combo(ui: &mut Ui, id: impl std::hash::Hash + std::fmt::Debug, value: &mut String, options: &[&str]) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(value.as_str())
        .show_ui(ui, |ui| {
            for o in options {
                ui.selectable_value(value, o.to_string(), *o);
            }
        });
}

/// The most coefficients qubit `q` can have: the QR that orthonormalises its basis needs at least as many points in
/// the pulse (rows) as basis functions (columns).
fn max_n_para(p: &Parameters, q: usize) -> usize {
    let Ok(b) = basis(&p.wf_type[q]) else {
        return usize::MAX;
    };
    let (mode, points) = (p.wf_mode[q], p.point_in_pulse[q]);
    // No basis has fewer than one function per two coefficients, so the search ends before its bound.
    (1..=2 * points + 4)
        .filter(|&n| b.check_n_para(n, mode).is_ok())
        .take_while(|&n| b.columns(n, mode) <= points)
        .last()
        .unwrap_or(1)
}

fn section(ui: &mut Ui, title: &str, open: bool, body: impl FnOnce(&mut Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong())
        .default_open(open)
        .show(ui, body);
}

fn system(ui: &mut Ui, cfg: &mut Config, platform: Platform) {
    section(ui, "System", true, |ui| {
        egui::Grid::new("system")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("Platform");
                let mut model = cfg.hamiltonian_type.clone();
                egui::ComboBox::from_id_salt("model")
                    .selected_text(model_label(model.as_deref()))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut model, None, model_label(None));
                        for m in model_names() {
                            ui.selectable_value(&mut model, Some(m.to_string()), model_label(Some(m)));
                        }
                    });
                if model != cfg.hamiltonian_type {
                    edit::set_model(cfg, model.as_deref());
                }
                ui.end_row();

                ui.label("Qubits");
                ui.horizontal(|ui| {
                    let n = cfg.n_qubits();
                    let max = if platform.is_web { MAX_WEB_QUBITS } else { MAX_QUBITS };
                    if ui.add_enabled(n > 1, egui::Button::new("−")).clicked() {
                        edit::set_qubit_count(cfg, n - 1);
                    }
                    ui.label(n.to_string());
                    if ui.add_enabled(n < max, egui::Button::new("+")).clicked() {
                        edit::set_qubit_count(cfg, n + 1);
                    }
                });
                ui.end_row();

                let dissipative = cfg.is_dissipative();
                ui.label("Space");
                ui.add_enabled_ui(!dissipative, |ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut cfg.optimization.space, Space::Hilbert, "Hilbert (states)");
                        ui.radio_value(&mut cfg.optimization.space, Space::Liouville, "Liouville (densities)");
                    });
                });
                ui.end_row();

                ui.label("Relaxation");
                let mut on = dissipative;
                let allowed = cfg.hamiltonian_type.as_deref() != Some("duffing_transmon");
                let r = ui.add_enabled(allowed, egui::Checkbox::new(&mut on, "T1 and T2 (Lindblad)"));
                let r = if allowed {
                    r
                } else {
                    r.on_disabled_hover_text("Not available for three-level transmons")
                };
                if r.changed() {
                    edit::set_dissipative(cfg, on);
                }
                ui.end_row();
            });
    });
}

fn model_label(model: Option<&str>) -> &'static str {
    match model {
        None => "Spin chain (original)",
        Some("spin_chain") => "Spin chain",
        Some("superconducting") => "Transmons, two levels",
        Some("duffing_transmon") => "Transmons, three levels",
        Some(_) => "Unknown",
    }
}

fn pulse(ui: &mut Ui, cfg: &mut Config) {
    section(ui, "Pulse", true, |ui| {
        egui::Grid::new("pulse")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                let mut duration = cfg.parameters.pulse_duration.first().copied().unwrap_or(2e-7);
                let mut points = cfg.parameters.point_in_pulse.first().copied().unwrap_or(100);
                ui.label("Duration");
                let a = seconds(ui, &mut duration).changed();
                ui.end_row();
                ui.label("Time steps");
                let b = ui.add(number(&mut points).range(2..=2000)).changed();
                ui.end_row();
                if a || b {
                    edit::set_pulse(cfg, duration.max(1e-12), points);
                }
            });
    });
}

fn qubits(ui: &mut Ui, cfg: &mut Config) {
    let n = cfg.n_qubits();
    let model = cfg.hamiltonian_type.clone();
    let dissipative = cfg.is_dissipative();
    section(ui, "Qubits", true, |ui| {
        let p = &mut cfg.parameters;
        let max_coefficients: Vec<usize> = (0..n).map(|q| max_n_para(p, q)).collect();
        let delta = p.delta.get_or_insert_with(|| vec![0.0; n]);
        let selective = p
            .coverage
            .iter()
            .any(|c| matches!(c, Coverage::Selective | Coverage::BandSelective));
        let band = p.coverage.contains(&Coverage::BandSelective);
        egui::ScrollArea::horizontal().id_salt("qubit-grid").show(ui, |ui| {
            egui::Grid::new("qubits")
                .num_columns(n + 1)
                .striped(true)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label("");
                    for q in 0..n {
                        ui.label(RichText::new(format!("q{}", q + 1)).strong());
                    }
                    ui.end_row();

                    row(ui, "Offset Δ", n, |ui, q| drop(hz(ui, &mut delta[q])));
                    let sigma = p.sigma_delta.get_or_insert_with(|| vec![0.0; n]);
                    row(ui, "Offset spread σΔ", n, |ui, q| drop(hz(ui, &mut sigma[q])));
                    row(ui, "Rabi frequency", n, |ui, q| drop(hz(ui, &mut p.omega_r_max[q])));
                    row(ui, "Rabi spread", n, |ui, q| drop(hz(ui, &mut p.sigma_omega_r_max[q])));
                    row(ui, "Basis", n, |ui, q| {
                        string_combo(ui, ("basis", q), &mut p.wf_type[q], basis_names())
                    });
                    row(ui, "Waveform", n, |ui, q| {
                        let options: Vec<(WaveformMode, &str)> =
                            WaveformMode::ALL.iter().map(|m| (*m, m.name())).collect();
                        combo(ui, ("mode", q), &mut p.wf_mode[q], &options);
                    });
                    row(ui, "Coefficients", n, |ui, q| {
                        // A value already past the limit, after fewer points, is left for validation to report.
                        let field = number(&mut p.n_para[q]).range(1..=max_coefficients[q]);
                        drop(ui.add(field.clamp_existing_to_range(false)))
                    });
                    row(ui, "Envelope", n, |ui, q| {
                        string_combo(ui, ("env", q), &mut p.amplitude_envelope[q], envelope_names())
                    });
                    row(ui, "Envelope order", n, |ui, q| {
                        drop(ui.add(number(&mut p.amplitude_order[q]).range(1..=32)))
                    });
                    row(ui, "Coverage", n, |ui, q| {
                        let options: Vec<(Coverage, &str)> = Coverage::ALL.iter().map(|c| (*c, c.name())).collect();
                        combo(ui, ("coverage", q), &mut p.coverage[q], &options);
                    });
                    row(ui, "Sweep width", n, |ui, q| drop(hz(ui, &mut p.sw[q])));
                    row(ui, "Carrier offset", n, |ui, q| drop(hz(ui, &mut p.pulse_offset[q])));
                    if selective {
                        row(ui, "Band width", n, |ui, q| drop(hz(ui, &mut p.pulse_bandwidth[q])));
                        row(ui, "Fraction outside band", n, |ui, q| {
                            drop(ui.add(number(&mut p.ratio_factor[q]).range(0.0..=1.0).speed(0.01)))
                        });
                    }
                    if band {
                        row(ui, "Profile order", n, |ui, q| {
                            drop(ui.add(number(&mut p.profile_order[q]).range(1..=32)))
                        });
                    }
                    if dissipative {
                        let t1 = p.t1.get_or_insert_with(|| vec![1e-3; n]);
                        row(ui, "T1", n, |ui, q| drop(seconds(ui, &mut t1[q])));
                        let t2 = p.t2.get_or_insert_with(|| vec![5e-4; n]);
                        row(ui, "T2", n, |ui, q| drop(seconds(ui, &mut t2[q])));
                    }
                    if matches!(model.as_deref(), Some("superconducting" | "duffing_transmon")) {
                        let alpha = p.anharmonicities.get_or_insert_with(|| vec![-330e6; n]);
                        row(ui, "Anharmonicity", n, |ui, q| drop(hz(ui, &mut alpha[q])));
                    }
                    if model.as_deref() == Some("superconducting") {
                        ui.label("AC Stark shift");
                        let mut on = p.stark_shift_coeffs.is_some();
                        if ui.checkbox(&mut on, "").changed() {
                            p.stark_shift_coeffs = on.then(|| vec![0.0; n]);
                        }
                        ui.end_row();
                        if let Some(s) = &mut p.stark_shift_coeffs {
                            row(ui, "Stark coefficient", n, |ui, q| {
                                drop(
                                    ui.add(
                                        number(&mut s[q])
                                            .speed(1e-11)
                                            .custom_formatter(|v, _| numbers::format(v)),
                                    ),
                                )
                            });
                        }
                    }
                });
        });
    });
}

/// One labelled grid row with a cell per qubit.
fn row(ui: &mut Ui, label: &str, n: usize, mut cell: impl FnMut(&mut Ui, usize)) {
    ui.label(label);
    for q in 0..n {
        cell(ui, q);
    }
    ui.end_row();
}

fn coupling(ui: &mut Ui, cfg: &mut Config) {
    let n = cfg.n_qubits();
    let model = cfg.hamiltonian_type.clone();
    section(ui, "Coupling", true, |ui| {
        let p = &mut cfg.parameters;
        ui.horizontal(|ui| {
            ui.label("Type");
            let mut c = p
                .coupling_type
                .clone()
                .unwrap_or_else(|| coupling_types(model.as_deref())[0].into());
            string_combo(ui, "coupling-type", &mut c, coupling_types(model.as_deref()));
            p.coupling_type = Some(c);
            ui.separator();
            ui.label("Spread σJ");
            let s = p.sigma_j.get_or_insert(0.0);
            hz(ui, s);
        });
        ui.label("Couplings J (upper triangle)");
        let j = p.j.get_or_insert_with(|| vec![vec![0.0; n]; n]);
        matrix(ui, "j", j, n);
        if model.as_deref() == Some("superconducting") {
            let mut on = p.zz_crosstalk.is_some();
            if ui
                .checkbox(&mut on, "Calibrated static ZZ (otherwise from the anharmonicities)")
                .changed()
            {
                p.zz_crosstalk = on.then(|| vec![vec![0.0; n]; n]);
            }
            if let Some(zz) = &mut p.zz_crosstalk {
                matrix(ui, "zz", zz, n);
            }
        }
    });
}

/// Edit the upper triangle of an `n × n` matrix in Hz.  A value given in the lower triangle is moved up.
#[allow(clippy::needless_range_loop)] // entries (r, c) and (c, r) are read together
fn matrix(ui: &mut Ui, id: &str, m: &mut [Vec<f64>], n: usize) {
    egui::Grid::new(id)
        .num_columns(n + 1)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("");
            for c in 0..n {
                ui.label(format!("q{}", c + 1));
            }
            ui.end_row();
            for r in 0..n {
                ui.label(format!("q{}", r + 1));
                for c in 0..n {
                    if c > r {
                        if m[r][c] == 0.0 && m[c][r] != 0.0 {
                            m[r][c] = m[c][r];
                            m[c][r] = 0.0;
                        }
                        hz(ui, &mut m[r][c]);
                    } else {
                        ui.label("");
                    }
                }
                ui.end_row();
            }
        });
}

fn targets(ui: &mut Ui, cfg: &mut Config) {
    let n = cfg.n_qubits();
    section(ui, "Targets", true, |ui| {
        ui.horizontal(|ui| {
            let mut method = TargetMethod::of(&cfg.target_states);
            for m in [TargetMethod::Axis, TargetMethod::Gate, TargetMethod::Rotation] {
                let enabled = m != TargetMethod::Gate || !gate_names(n).is_empty();
                ui.add_enabled_ui(enabled, |ui| ui.radio_value(&mut method, m, m.label()));
            }
            if method != TargetMethod::of(&cfg.target_states) {
                edit::set_target_method(cfg, method);
            }
        });
        let mut remove = None;
        egui::Grid::new("targets")
            .num_columns(4)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                for i in 0..cfg.initial_states.len() {
                    ui.label(format!("{}.", i + 1));
                    ui.horizontal(|ui| {
                        for (q, axis) in cfg.initial_states[i].iter_mut().enumerate() {
                            string_combo(ui, ("init", i, q), axis, &STATE_AXES);
                        }
                    });
                    ui.label("to");
                    ui.horizontal(|ui| match &mut cfg.target_states {
                        Targets::Axis(axes) => {
                            for (q, axis) in axes[i].iter_mut().enumerate() {
                                string_combo(ui, ("targ", i, q), axis, &STATE_AXES);
                            }
                        }
                        Targets::Gate(gates) => string_combo(ui, ("gate", i), &mut gates[i], gate_names(n)),
                        Targets::PhiBeta { phi, beta } => {
                            for (q, (axis, angle)) in phi[i].iter_mut().zip(beta[i].iter_mut()).enumerate() {
                                string_combo(ui, ("phi", i, q), axis, &ROTATION_AXES);
                                ui.add(number(angle).speed(1.0).suffix("°"));
                            }
                        }
                    });
                    if ui
                        .add_enabled(cfg.initial_states.len() > 1, egui::Button::new("✖").small())
                        .clicked()
                    {
                        remove = Some(i);
                    }
                    ui.end_row();
                }
            });
        if let Some(i) = remove {
            edit::remove_initial_state(cfg, i);
        }
        if ui.button("Add initial state").clicked() {
            edit::add_initial_state(cfg);
        }
    });
}

fn optimisation(ui: &mut Ui, cfg: &mut Config, platform: Platform) {
    section(ui, "Optimisation", true, |ui| {
        egui::Grid::new("optimisation")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                let o = &mut cfg.optimization;
                ui.label("Algorithm");
                string_combo(ui, "algorithm", &mut o.algorithm, optimizer_names());
                ui.end_row();
                ui.label("Iteration limit");
                ui.add(number(&mut o.max_iter).range(1..=100_000))
                    .on_hover_text("Function evaluations for the derivative-free optimisers");
                ui.end_row();
                ui.label("Target fidelity");
                ui.add(number(&mut o.targ_fid).range(0.0..=1.0).speed(0.0005).max_decimals(6));
                ui.end_row();
                ui.label("Offset snapshots");
                ui.add(number(&mut o.h0_snapshots).range(1..=2000));
                ui.end_row();
                ui.label("Rabi snapshots");
                ui.add(number(&mut o.omega_r_snapshots).range(1..=200));
                ui.end_row();

                ui.label("Seed");
                ui.horizontal(|ui| {
                    let mut fixed = cfg.seed.is_some();
                    if ui.checkbox(&mut fixed, "fixed").changed() {
                        cfg.seed = fixed.then_some(1);
                    }
                    if let Some(seed) = &mut cfg.seed {
                        ui.add(number(seed));
                    } else {
                        ui.label(RichText::new("new each run").weak());
                    }
                });
                ui.end_row();

                if !platform.is_web {
                    ui.label("Threads");
                    ui.horizontal(|ui| {
                        let mut limited = cfg.cpu_cores.is_some();
                        if ui.checkbox(&mut limited, "limit").changed() {
                            cfg.cpu_cores = limited.then_some(platform.cores);
                        }
                        match &mut cfg.cpu_cores {
                            Some(c) => drop(ui.add(egui::Slider::new(c, 1..=platform.cores.max(1)))),
                            None => drop(ui.label(RichText::new(format!("all {}", platform.cores)).weak())),
                        }
                    });
                    ui.end_row();
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames of an 800 × 600 window holding one number field for `value`.
    struct Window {
        ctx: egui::Context,
        frame: u32,
    }

    impl Window {
        fn new() -> Self {
            Window {
                ctx: egui::Context::default(),
                frame: 0,
            }
        }

        /// One frame with `events`; returns where the field is.
        fn frame(&mut self, value: &mut usize, events: Vec<egui::Event>) -> egui::Rect {
            self.frame += 1;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))),
                time: Some(0.1 * f64::from(self.frame)),
                events,
                ..Default::default()
            };
            let mut rect = egui::Rect::NOTHING;
            let mut output = self.ctx.run_ui(input, |ui| rect = ui.add(number(value)).rect);
            // Nothing paints here, so nothing takes the font atlas.
            output.textures_delta.clear();
            rect
        }
    }

    fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }

    /// Typing 200 goes through 2 and 20 on the way; neither is stored, so neither is checked or acted on.
    #[test]
    fn a_typed_number_is_stored_when_it_is_finished() {
        let mut value = 100;
        let mut w = Window::new();
        let at = w.frame(&mut value, vec![]).center();
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        w.frame(&mut value, vec![egui::Event::PointerMoved(at), button(true)]);
        w.frame(&mut value, vec![button(false)]);
        w.frame(&mut value, vec![key(egui::Key::A, egui::Modifiers::COMMAND)]);
        w.frame(&mut value, vec![egui::Event::Text("2".into())]);
        assert_eq!(value, 100, "half-typed");
        w.frame(&mut value, vec![egui::Event::Text("00".into())]);
        assert_eq!(value, 100, "typed, not yet entered");
        w.frame(&mut value, vec![key(egui::Key::Enter, egui::Modifiers::NONE)]);
        w.frame(&mut value, vec![]);
        assert_eq!(value, 200);
    }

    /// Clicking elsewhere - another field, or Run - finishes the number as Enter does.
    #[test]
    fn leaving_a_field_stores_what_was_typed() {
        let mut value = 100;
        let mut w = Window::new();
        let at = w.frame(&mut value, vec![]).center();
        let click = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        w.frame(&mut value, vec![egui::Event::PointerMoved(at), click(at, true)]);
        w.frame(&mut value, vec![click(at, false)]);
        w.frame(&mut value, vec![key(egui::Key::A, egui::Modifiers::COMMAND)]);
        w.frame(&mut value, vec![egui::Event::Text("200".into())]);
        let away = egui::pos2(600.0, 400.0);
        w.frame(&mut value, vec![egui::Event::PointerMoved(away), click(away, true)]);
        w.frame(&mut value, vec![click(away, false)]);
        assert_eq!(value, 200);
    }

    #[test]
    fn the_coefficient_limit_is_the_most_basis_functions_the_points_hold() {
        let mut c = ctrl_freeq::hamiltonian::default_config("spin_chain", 1).unwrap();
        for name in basis_names() {
            let b = basis(name).unwrap();
            for mode in WaveformMode::ALL {
                for points in [2, 3, 10, 101, 1000] {
                    let p = &mut c.parameters;
                    (p.wf_type[0], p.wf_mode[0], p.point_in_pulse[0]) = (name.to_string(), mode, points);
                    let max = max_n_para(p, 0);
                    let next = (max + 1..).find(|&n| b.check_n_para(n, mode).is_ok()).unwrap();
                    let at = format!("{name}, {}, {points} points", mode.name());
                    assert!(b.columns(max, mode) <= points, "{at}: {max} does not fit");
                    assert!(b.columns(next, mode) > points, "{at}: {next} fits too");
                    c.parameters.n_para[0] = max;
                    assert!(c.validate().is_empty(), "{at}: {:?}", c.validate());
                }
            }
        }
    }
}
