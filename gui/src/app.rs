//! The application: the top bar, the run loop, and the two panels.

use ctrl_freeq::config::Config;
use ctrl_freeq::optim::IterationReport;
use eframe::egui;

use crate::export;
use crate::inputs;
use crate::outputs::{self, View};
use crate::platform::{self, FilePicker, Platform};
use crate::presets;
use crate::run::{Results, RunMessage};
use crate::runner::Runner;

/// Where a run has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Running,
    Done,
    Failed,
}

/// The whole application.
pub struct App {
    config: Config,
    presets: Vec<(String, Config)>,
    platform: Platform,
    picker: FilePicker,
    runner: Option<Runner>,
    history: Vec<IterationReport>,
    results: Option<Box<Results>>,
    status: Status,
    message: Option<String>,
    view: View,
}

const CONFIG_KEY: &str = "config";

impl App {
    /// A fresh application, showing the configuration from the last session if there was one.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Paint the first frame straight away rather than on the first input event.
        cc.egui_ctx.request_repaint();
        let config = cc
            .storage
            .and_then(|s| s.get_string(CONFIG_KEY))
            .and_then(|json| Config::from_json(&json).ok())
            .unwrap_or_else(presets::initial);
        App {
            config,
            presets: presets::presets(),
            platform: platform::detect(),
            picker: FilePicker::default(),
            runner: None,
            history: Vec::new(),
            results: None,
            status: Status::Idle,
            message: None,
            view: View::default(),
        }
    }

    fn start(&mut self) {
        self.history.clear();
        self.results = None;
        self.message = None;
        self.view = View::default();
        self.status = Status::Running;
        self.runner = Some(Runner::start(self.config.clone()));
    }

    fn cancel(&mut self) {
        if let Some(runner) = &mut self.runner {
            runner.cancel();
        }
        if !Runner::CANCEL_KEEPS_RESULT {
            self.runner = None;
            self.status = Status::Done;
            self.message = Some("Cancelled. The convergence so far is kept; the partial pulse is not.".into());
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(loaded) = self.picker.take() {
            match loaded.and_then(|json| Config::from_json(&json).map_err(|e| e.to_string())) {
                Ok(c) => {
                    self.config = c;
                    self.message = Some("Configuration loaded.".into());
                }
                Err(e) => self.message = Some(format!("Not loaded: {e}")),
            }
        }
        let Some(runner) = &mut self.runner else {
            return;
        };
        let mut finished = false;
        for message in runner.drain() {
            match message {
                RunMessage::Ready => {}
                RunMessage::Progress(p) => self.history.push(p),
                RunMessage::Finished(results) => {
                    let mut text = format!(
                        "{}: fidelity {:.6} after {} iterations in {:.1} s (seed {}).",
                        capitalise(&results.run.exit.message()),
                        results.run.fidelity,
                        results.run.iterations,
                        results.run.elapsed_s,
                        results.run.seed,
                    );
                    for notice in &results.run.notices {
                        text.push(' ');
                        text.push_str(notice);
                    }
                    self.message = Some(text);
                    self.results = Some(results);
                    self.status = Status::Done;
                    finished = true;
                }
                RunMessage::Failed(e) => {
                    self.message = Some(format!("The run failed: {e}"));
                    self.status = Status::Failed;
                    finished = true;
                }
            }
        }
        if finished {
            self.runner = None;
        } else {
            // Keep frames coming while work is in flight, so the plots move without the mouse.
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    /// Hand the user a file, reporting a failure in the status line.
    fn offer(&mut self, filename: &str, contents: &str) {
        if let Err(e) = export::offer_file(filename, contents) {
            self.message = Some(e);
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let running = self.status == Status::Running;
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.heading("Rusty-ctrl-freeq");
            ui.separator();
            ui.add_enabled_ui(!running, |ui| {
                egui::ComboBox::from_id_salt("presets")
                    .selected_text("Presets")
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        for (name, preset) in &self.presets {
                            if ui.selectable_label(false, name).clicked() {
                                self.config = preset.clone();
                                self.message = Some(format!("Loaded the preset \"{name}\"."));
                            }
                        }
                    });
                if ui
                    .button("Open…")
                    .on_hover_text("Load a configuration JSON file")
                    .clicked()
                {
                    self.picker.open();
                }
            });
            if ui
                .button("Save…")
                .on_hover_text("Save the configuration as JSON")
                .clicked()
            {
                let json = self.config.to_json();
                self.offer("ctrl-freeq-config.json", &json);
            }
            ui.separator();
            let problems = self.config.validate();
            if running {
                if ui.button("Cancel").clicked() {
                    self.cancel();
                }
                ui.spinner();
            } else {
                let run = ui.add_enabled(problems.is_empty(), egui::Button::new("▶ Run"));
                if run.clicked() {
                    self.start();
                }
            }
            ui.separator();
            let has_results = self.results.is_some();
            if ui
                .add_enabled(has_results, egui::Button::new("Results JSON…"))
                .clicked()
                && let Some(r) = &self.results
            {
                let json = export::results_json(r);
                self.offer("ctrl-freeq-results.json", &json);
            }
            if ui
                .add_enabled(has_results, egui::Button::new("Waveforms CSV…"))
                .clicked()
                && let Some(r) = &self.results
            {
                let csv = export::waveform_csv(&r.analysis);
                self.offer("ctrl-freeq-waveforms.csv", &csv);
            }
        });
        ui.horizontal_wrapped(|ui| {
            if let Some(last) = self.history.last() {
                ui.label(format!(
                    "Iteration {} · fidelity {:.6} · penalty {:.2e} · {:.1} s",
                    last.iteration, last.fidelity, last.penalty, last.elapsed_s
                ));
                ui.separator();
            }
            if let Some(m) = &self.message {
                let colour = match self.status {
                    Status::Failed => ui.visuals().error_fg_color,
                    _ => ui.visuals().text_color(),
                };
                ui.colored_label(colour, m);
            }
        });
        ui.add_space(2.0);
    }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string(CONFIG_KEY, self.config.to_json());
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll(ui.ctx());
        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui));
        let running = self.status == Status::Running;
        let platform = self.platform;
        egui::Panel::left("inputs")
            .resizable(true)
            .default_size(440.0)
            .size_range(340.0..=760.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_enabled_ui(!running, |ui| inputs::show(ui, &mut self.config, platform));
                });
            });
        egui::CentralPanel::default().show(ui, |ui| {
            outputs::show(ui, &mut self.view, &self.history, self.results.as_deref());
        });
    }
}
