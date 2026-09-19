//! What this build runs on, and opening a file: a dialog natively, a file input in the browser.

/// Facts about the host that change what the interface offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
    /// Running in a browser rather than as a desktop application.
    pub is_web: bool,
    /// Threads a run is spread over by default.  One in the browser, whose WebAssembly build has no threads.
    pub cores: usize,
}

/// The platform this build runs on.
pub fn detect() -> Platform {
    Platform {
        is_web: cfg!(target_arch = "wasm32"),
        cores: ctrl_freeq::run::available_threads(),
    }
}

/// Opening a text file chosen by the user.  Natively the dialog blocks and the text is ready at once; in the
/// browser it arrives a few frames later, so the interface polls.
#[derive(Default)]
pub struct FilePicker {
    #[cfg(target_arch = "wasm32")]
    pending: std::rc::Rc<std::cell::RefCell<Option<Result<String, String>>>>,
    #[cfg(not(target_arch = "wasm32"))]
    pending: Option<Result<String, String>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FilePicker {
    /// Ask the user for a JSON file.
    pub fn open(&mut self) {
        if let Some(path) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
            self.pending = Some(std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display())));
        }
    }

    /// The chosen file's text, once.
    pub fn take(&mut self) -> Option<Result<String, String>> {
        self.pending.take()
    }
}

#[cfg(target_arch = "wasm32")]
impl FilePicker {
    /// Ask the user for a JSON file.
    pub fn open(&mut self) {
        use wasm_bindgen::JsCast;
        use wasm_bindgen::prelude::*;

        let pending = std::rc::Rc::clone(&self.pending);
        let go = move || -> Option<()> {
            let document = web_sys::window()?.document()?;
            let input: web_sys::HtmlInputElement = document.create_element("input").ok()?.dyn_into().ok()?;
            input.set_type("file");
            input.set_accept(".json,application/json");
            let reader_input = input.clone();
            let on_change = Closure::<dyn FnMut(web_sys::Event)>::new(move |_: web_sys::Event| {
                let Some(file) = reader_input.files().and_then(|f| f.get(0)) else {
                    return;
                };
                let Ok(reader) = web_sys::FileReader::new() else {
                    *pending.borrow_mut() = Some(Err("this browser cannot read files".into()));
                    return;
                };
                let done_reader = reader.clone();
                let done_pending = std::rc::Rc::clone(&pending);
                let on_load = Closure::<dyn FnMut(web_sys::Event)>::new(move |_: web_sys::Event| {
                    let text = done_reader.result().ok().and_then(|v| v.as_string());
                    *done_pending.borrow_mut() = Some(text.ok_or_else(|| "the file could not be read".to_string()));
                });
                reader.set_onload(Some(on_load.as_ref().unchecked_ref()));
                on_load.forget();
                if reader.read_as_text(&file).is_err() {
                    *pending.borrow_mut() = Some(Err("the file could not be read".into()));
                }
            });
            input.set_onchange(Some(on_change.as_ref().unchecked_ref()));
            on_change.forget();
            input.click();
            Some(())
        };
        if go().is_none() {
            *self.pending.borrow_mut() = Some(Err("this browser would not open a file chooser".into()));
        }
    }

    /// The chosen file's text, once.
    pub fn take(&mut self) -> Option<Result<String, String>> {
        self.pending.borrow_mut().take()
    }
}
