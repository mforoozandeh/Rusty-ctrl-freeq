//! The Fourier basis, whose parameter counts follow their own rule.

use super::Basis;
use crate::config::WaveformMode;
use crate::linalg::RMat;
use crate::setup::Rng;

/// `[1, cos 2πkx, sin 2πkx for k = 1..n]`: `2n + 1` columns.
pub(super) struct Fourier;

impl Fourier {
    /// The harmonic count `n` the Python package derives from `n_para`.
    fn harmonics(n_para: usize, mode: WaveformMode) -> usize {
        match mode {
            WaveformMode::PolarPhase => n_para.saturating_sub(1) / 2,
            WaveformMode::Cart | WaveformMode::Polar => n_para.saturating_sub(2) / 4,
        }
    }
}

impl Basis for Fourier {
    fn name(&self) -> &'static str {
        "fou"
    }

    fn matrix(&self, x: &[f64], ncols: usize, _: &mut Rng) -> RMat<f64> {
        let tau = 2.0 * std::f64::consts::PI;
        RMat::from_fn(x.len(), ncols, |r, c| {
            if c == 0 {
                1.0
            } else {
                let k = c.div_ceil(2) as f64;
                if c % 2 == 1 {
                    (tau * k * x[r]).cos()
                } else {
                    (tau * k * x[r]).sin()
                }
            }
        })
    }

    fn columns(&self, n_para: usize, mode: WaveformMode) -> usize {
        2 * Self::harmonics(n_para, mode) + 1
    }

    fn parameter_count(&self, n_para: usize, mode: WaveformMode) -> usize {
        let n = Self::harmonics(n_para, mode);
        match mode {
            WaveformMode::PolarPhase => 2 * n + 2,
            WaveformMode::Cart | WaveformMode::Polar => 4 * n + 2,
        }
    }

    fn check_n_para(&self, n_para: usize, mode: WaveformMode) -> std::result::Result<(), String> {
        let least = match mode {
            WaveformMode::PolarPhase => 1,
            WaveformMode::Cart | WaveformMode::Polar => 2,
        };
        if n_para < least {
            return Err(format!(
                "n_para must be at least {least} for the Fourier basis in {} mode",
                mode.name()
            ));
        }
        Ok(())
    }

    fn initial_coefficients(&self, raw: &[f64], mode: WaveformMode) -> Vec<f64> {
        let n = Self::harmonics(raw.len(), mode);
        match mode {
            WaveformMode::PolarPhase => raw[..2 * n + 1].to_vec(),
            WaveformMode::Cart | WaveformMode::Polar => raw[..4 * n + 2].to_vec(),
        }
    }
}
