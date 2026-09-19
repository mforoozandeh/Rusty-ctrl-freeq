//! The optimisation worker.
//!
//! Trunk builds this as a second WebAssembly binary and the page starts it as a Web Worker.  It receives a JSON
//! configuration by `postMessage`, runs and analyses it, and posts a JSON `RunMessage` back for every iteration and
//! once more at the end.
//!
//! It never cancels itself: the page terminates the worker instead, because a worker blocked in the optimiser
//! cannot read its message queue.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("ctrl-freeq-worker is the WebAssembly worker for the browser build; it does nothing natively.");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use ctrl_freeq_gui::run::{self, RunMessage};
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;

    console_error_panic_hook::set_once();

    let scope: web_sys::DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let handler_scope = scope.clone();

    let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
        let mut sink = PostSink {
            scope: handler_scope.clone(),
        };
        let Some(text) = event.data().as_string() else {
            run::MessageSink::send(
                &mut sink,
                RunMessage::Failed("the worker was sent something other than a configuration".into()),
            );
            return;
        };
        match ctrl_freeq::config::Config::from_json(&text) {
            Ok(config) => run::run_to_sink(&config, &mut sink),
            Err(e) => run::MessageSink::send(&mut sink, RunMessage::Failed(format!("unreadable configuration: {e}"))),
        }
    });
    scope.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    // The closure has to outlive `main`, which returns at once.
    on_message.forget();

    // Only now is it safe to be sent work: anything posted before this point had no handler and was lost.
    if let Ok(json) = serde_json::to_string(&RunMessage::Ready) {
        let _ = scope.post_message(&JsValue::from_str(&json));
    }
}

/// A `MessageSink` that posts each message to the page.
#[cfg(target_arch = "wasm32")]
struct PostSink {
    scope: web_sys::DedicatedWorkerGlobalScope,
}

#[cfg(target_arch = "wasm32")]
impl ctrl_freeq_gui::run::MessageSink for PostSink {
    fn send(&mut self, message: ctrl_freeq_gui::run::RunMessage) {
        if let Ok(json) = serde_json::to_string(&message) {
            let _ = self.scope.post_message(&wasm_bindgen::JsValue::from_str(&json));
        }
    }
}
