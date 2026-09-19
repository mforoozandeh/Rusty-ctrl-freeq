//! Amplitude envelopes, which taper the waveform to (nearly) zero at the pulse edges.

use crate::error::{Error, Result};

/// The envelope with JSON name `name` and order `order`, evaluated at `x ∈ [−1, 1]`.
///
/// * `gn`: super-Gaussian of order `order`, its width chosen so its value at `x = ±1` is the same for every order.
/// * `hs`: hyperbolic secant of `5.3·x^order`.
/// * `quad`: `1 − x² + ε`.
///
/// `gn` and `hs` are rescaled to span `[ε, 1]`, ε being the machine epsilon, so no sample is exactly zero.
pub fn envelope(name: &str, x: &[f64], order: u32) -> Result<Vec<f64>> {
    let n = order as i32;
    let raw: Vec<f64> = match name {
        "gn" => {
            let sigma: f64 = 0.25;
            let g_ext = (-(1.0 / (2.0 * sigma * sigma))).exp();
            let sigma_u = (1.0 / (2.0 * (-g_ext.ln()).powf(1.0 / order as f64))).sqrt();
            x.iter()
                .map(|&v| (-((v * v / (2.0 * sigma_u * sigma_u)).powi(n))).exp())
                .collect()
        }
        "hs" => {
            let beta = 10.6 / 2.0;
            x.iter()
                .map(|&v| {
                    let a = beta * v.powi(n);
                    2.0 / (a.exp() + (-a).exp())
                })
                .collect()
        }
        "quad" => return Ok(x.iter().map(|&v| 1.0 - v * v + f64::EPSILON).collect()),
        _ => {
            return Err(Error::Config(format!(
                "unknown amplitude envelope \"{name}\"; choose one of {}",
                envelope_names().join(", ")
            )));
        }
    };
    Ok(rescale(&raw))
}

/// Every envelope name, in the order the interface lists them.
pub fn envelope_names() -> &'static [&'static str] {
    &["gn", "hs", "quad"]
}

/// Rescale to `[ε, 1]`: `(e − min)/(max − min)·(1 − ε) + ε`.
fn rescale(e: &[f64]) -> Vec<f64> {
    let min = e.iter().copied().fold(f64::INFINITY, f64::min);
    let max = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let eps = f64::EPSILON;
    e.iter().map(|&v| (v - min) / (max - min) * (1.0 - eps) + eps).collect()
}
