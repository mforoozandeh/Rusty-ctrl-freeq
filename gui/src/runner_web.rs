//! Web runner: the optimisation in a dedicated Web Worker.
//!
//! A plain worker is used rather than WebAssembly threads on purpose.  Those need `SharedArrayBuffer`, which needs
//! cross-origin-isolation headers not every host can set, and a nightly standard library.  A worker gets its own
//! WebAssembly instance, talks by `postMessage`, and works on any static host.
//!
//! The cost is cancellation: a worker inside a synchronous optimisation cannot read an incoming message, so Cancel
//! terminates it.  The convergence recorded so far survives; the partial pulse does not, and the interface says so.

use std::cell::RefCell;
use std::rc::Rc;

use ctrl_freeq::config::Config;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use crate::run::RunMessage;

/// A run in progress.
pub struct Runner {
    worker: Option<web_sys::Worker>,
    inbox: Rc<RefCell<Vec<RunMessage>>>,
    // Kept alive as long as the worker is: dropping them would detach the handlers.
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_error: Closure<dyn FnMut(web_sys::Event)>,
}

impl Runner {
    /// Start `config` in a fresh worker.
    pub fn start(config: Config) -> Self {
        let inbox: Rc<RefCell<Vec<RunMessage>>> = Rc::new(RefCell::new(Vec::new()));
        // The worker's WebAssembly must load before it can listen, and a message posted earlier is dropped, so the
        // configuration waits here until the worker says it is ready.
        let pending: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(Some(config.to_json())));

        let worker = spawn_worker();

        let for_message = Rc::clone(&inbox);
        let pending_for_message = Rc::clone(&pending);
        let worker_for_message = worker.clone();
        let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
            let Some(text) = event.data().as_string() else {
                return;
            };
            match serde_json::from_str::<RunMessage>(&text) {
                Ok(RunMessage::Ready) => {
                    let Some(json) = pending_for_message.borrow_mut().take() else {
                        return;
                    };
                    let posted = worker_for_message
                        .as_ref()
                        .is_some_and(|w| w.post_message(&JsValue::from_str(&json)).is_ok());
                    if !posted {
                        for_message.borrow_mut().push(RunMessage::Failed(
                            "could not hand the configuration to the worker".into(),
                        ));
                    }
                }
                Ok(message) => for_message.borrow_mut().push(message),
                Err(e) => for_message
                    .borrow_mut()
                    .push(RunMessage::Failed(format!("unreadable worker message: {e}"))),
            }
        });

        let for_error = Rc::clone(&inbox);
        let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |_: web_sys::Event| {
            for_error.borrow_mut().push(RunMessage::Failed(
                "the optimisation worker stopped unexpectedly".into(),
            ));
        });

        match &worker {
            Some(w) => {
                w.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
                w.set_onerror(Some(on_error.as_ref().unchecked_ref()));
            }
            None => inbox
                .borrow_mut()
                .push(RunMessage::Failed("this browser would not start a Web Worker".into())),
        }

        Runner {
            worker,
            inbox,
            _on_message: on_message,
            _on_error: on_error,
        }
    }

    /// Every message that has arrived since the last call.
    pub fn drain(&mut self) -> Vec<RunMessage> {
        std::mem::take(&mut *self.inbox.borrow_mut())
    }

    /// Stop the run.  The worker is blocked in the optimiser and cannot hear a polite request, so it is terminated.
    pub fn cancel(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
    }

    /// Whether a cancelled run still delivers its best pulse on this platform: not in the browser.
    pub const CANCEL_KEEPS_RESULT: bool = false;
}

impl Drop for Runner {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
    }
}

/// Start the worker from the loader script Trunk emits next to the page.  It must be a classic worker: Trunk's
/// loader shim calls `importScripts`, which module workers lack.
fn spawn_worker() -> Option<web_sys::Worker> {
    let window = web_sys::window()?;
    let configured = js_sys::Reflect::get(&window, &JsValue::from_str("ctrl_freeq_worker_url"))
        .ok()
        .and_then(|v| v.as_string());
    let url = configured.unwrap_or_else(|| "./ctrl-freeq-worker_loader.js".to_string());
    web_sys::Worker::new(&url).ok()
}
