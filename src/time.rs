//! Clock shim.
//!
//! `std::time::Instant` and `std::time::SystemTime` panic on `wasm32-unknown-unknown`, which has no clock of its
//! own.  `web-time` provides drop-in replacements backed by `performance.now()` and `Date.now()`, so every timing
//! call in the crate goes through this module rather than `std::time` directly.

#[cfg(target_arch = "wasm32")]
pub use web_time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::{Instant, SystemTime, UNIX_EPOCH};
