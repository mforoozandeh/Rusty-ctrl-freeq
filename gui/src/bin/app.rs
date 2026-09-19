//! The interface binary: a window natively, a canvas in the browser.

use ctrl_freeq_gui::app::App;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 880.0])
            .with_min_inner_size([960.0, 620.0])
            .with_title("Rusty-ctrl-freeq"),
        ..Default::default()
    };
    // The application name keys the stored session; changing it would lose every saved configuration.
    eframe::run_native("ctrl-freeq", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();
    eframe::WebLogger::init(log::LevelFilter::Info).ok();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().expect("no window").document().expect("no document");
        let canvas = document
            .get_element_by_id("ctrl_freeq_canvas")
            .expect("the page needs a canvas with id ctrl_freeq_canvas")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("ctrl_freeq_canvas is not a canvas");
        let result = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(App::new(cc)))),
            )
            .await;
        // Take the loading text down whichever way this went.
        if let Some(loading) = document.get_element_by_id("loading") {
            match result {
                Ok(_) => loading.remove(),
                Err(e) => loading.set_inner_html(&format!("<p>Rusty-ctrl-freeq failed to start.</p><p>{e:?}</p>")),
            }
        }
    });
}
