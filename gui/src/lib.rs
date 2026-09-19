//! Rusty-ctrl-freeq: the ctrl-freeq pulse optimiser in a window, or in a browser tab.
//!
//! `cargo run -p ctrl-freeq-gui --release` gives the desktop application; `trunk serve --config gui/Trunk.toml`
//! gives the web one.  Same source both ways: the only platform-specific code is in [`platform`], the two `runner`
//! modules and the file helper in [`export`].
//!
//! The crate is a library with two binaries on top: `ctrl-freeq-gui` is the interface, and `ctrl-freeq-worker` is
//! the Web Worker the browser build runs the optimisation in.

pub mod app;
pub mod edit;
pub mod export;
pub mod inputs;
pub mod outputs;
pub mod platform;
pub mod presets;
pub mod run;

#[cfg_attr(target_arch = "wasm32", path = "runner_web.rs")]
#[cfg_attr(not(target_arch = "wasm32"), path = "runner_native.rs")]
pub mod runner;
