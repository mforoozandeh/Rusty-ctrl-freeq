//! Basis functions for the pulse waveforms.
//!
//! Each qubit's waveform is a combination of a few smooth basis functions sampled on the pulse's time grid, mapped
//! to `x ∈ [−1, 1]`.  The raw basis is multiplied by an amplitude envelope where the waveform mode calls for it and
//! orthonormalised by a QR decomposition, as in the Python package's `mat_with_amplitude_and_qr`.
//!
//! A new basis is one type implementing [`Basis`] plus one entry in [`basis`] and [`basis_names`].

mod envelope;
mod fourier;
mod polynomial;
mod qr;

pub use envelope::{envelope, envelope_names};
pub use qr::qr_q;

use rand::RngExt;

use crate::config::WaveformMode;
use crate::error::{Error, Result};
use crate::linalg::RMat;
use crate::setup::Rng;

/// A family of basis functions.
pub trait Basis: Send + Sync {
    /// The JSON name.
    fn name(&self) -> &'static str;

    /// The `x.len() × ncols` matrix whose column `i` is the `i`-th basis function at `x`.
    fn matrix(&self, x: &[f64], ncols: usize, rng: &mut Rng) -> RMat<f64>;

    /// Number of basis functions for `n_para` user coefficients in `mode`.
    fn columns(&self, n_para: usize, mode: WaveformMode) -> usize {
        match mode {
            WaveformMode::PolarPhase => n_para,
            WaveformMode::Cart | WaveformMode::Polar => n_para / 2,
        }
    }

    /// Number of optimiser parameters for `n_para` user coefficients in `mode` (the Python `update_n_para`).
    fn parameter_count(&self, n_para: usize, mode: WaveformMode) -> usize {
        match mode {
            WaveformMode::PolarPhase => n_para + 1,
            WaveformMode::Cart | WaveformMode::Polar => n_para,
        }
    }

    /// Whether `n_para` is usable in `mode`; the message says why not.
    fn check_n_para(&self, n_para: usize, mode: WaveformMode) -> std::result::Result<(), String> {
        match mode {
            WaveformMode::PolarPhase if n_para == 0 => Err("n_para must be at least 1".into()),
            WaveformMode::Cart | WaveformMode::Polar if n_para < 2 || n_para % 2 == 1 => Err(format!(
                "n_para must be even and at least 2 in {} mode (got {n_para})",
                mode.name()
            )),
            _ => Ok(()),
        }
    }

    /// The initial coefficients kept from `raw`, `n_para` uniform draws in `[−1, 1)`, before `polar_phase` appends
    /// its extra entry.
    fn initial_coefficients(&self, raw: &[f64], mode: WaveformMode) -> Vec<f64> {
        match mode {
            WaveformMode::PolarPhase => raw.to_vec(),
            WaveformMode::Cart | WaveformMode::Polar => raw[..2 * (raw.len() / 2)].to_vec(),
        }
    }
}

/// The basis with JSON name `name`.
pub fn basis(name: &str) -> Result<Box<dyn Basis>> {
    Ok(match name {
        "cheb" => Box::new(polynomial::Chebyshev),
        "leg" => Box::new(polynomial::Legendre),
        "gegen" => Box::new(polynomial::Gegenbauer { lambda: 0.5 }),
        "poly" => Box::new(polynomial::Monomial),
        "hermite" => Box::new(polynomial::Hermite),
        "chirp" => Box::new(polynomial::Chirp),
        "random" => Box::new(polynomial::Random),
        "fou" => Box::new(fourier::Fourier),
        _ => {
            return Err(Error::Config(format!(
                "unknown basis \"{name}\"; choose one of {}",
                basis_names().join(", ")
            )));
        }
    })
}

/// Every basis name, in the order the interface lists them.
pub fn basis_names() -> &'static [&'static str] {
    &["cheb", "leg", "fou", "poly", "hermite", "gegen", "chirp", "random"]
}

/// `n` points evenly spaced from `start` to `stop` inclusive, computed as numpy's `linspace` does.
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (n - 1) as f64;
            let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
            v[n - 1] = stop;
            v
        }
    }
}

/// One qubit's waveform basis: the two matrices its parameters multiply.
#[derive(Clone, Debug, PartialEq)]
pub struct QubitBasis {
    /// The waveform mode.
    pub mode: WaveformMode,
    /// `cart`: x and y quadratures; `polar`: amplitude and phase; `polar_phase`: the envelope (every column the
    /// same) and the phase basis.
    pub q: [RMat<f64>; 2],
    /// The amplitude envelope as an `n_pulse × 1` column; `polar_phase` scales it by its first parameter.
    pub envelope: RMat<f64>,
    /// Number of optimiser parameters for this qubit.
    pub n_params: usize,
}

/// Build one qubit's basis matrices: sample the basis on `n_pulse` points, apply the envelope and orthonormalise.
pub fn qubit_basis(
    basis: &dyn Basis,
    envelope: &[f64],
    n_para: usize,
    mode: WaveformMode,
    n_pulse: usize,
    rng: &mut Rng,
) -> Result<QubitBasis> {
    basis.check_n_para(n_para, mode).map_err(Error::Config)?;
    let ncols = basis.columns(n_para, mode);
    if ncols > n_pulse {
        return Err(Error::Config(format!(
            "{ncols} basis functions need at least {ncols} points in the pulse, not {n_pulse}"
        )));
    }
    if envelope.len() != n_pulse {
        return Err(Error::Dimension(
            "envelope length differs from the number of pulse points".into(),
        ));
    }
    let x = linspace(-1.0, 1.0, n_pulse);
    let b = basis.matrix(&x, ncols, rng);
    let enveloped = RMat::from_fn(n_pulse, ncols, |r, c| b.get(r, c) * envelope[r]);
    let orthonormal = |m: &RMat<f64>| orthonormal(m, basis.name());
    let q = match mode {
        WaveformMode::PolarPhase => [RMat::from_fn(n_pulse, ncols, |r, _| envelope[r]), orthonormal(&b)?],
        WaveformMode::Polar => [orthonormal(&enveloped)?, orthonormal(&b)?],
        WaveformMode::Cart => {
            let q = orthonormal(&enveloped)?;
            [q.clone(), q]
        }
    };
    Ok(QubitBasis {
        mode,
        q,
        envelope: RMat::column(envelope.to_vec()),
        n_params: basis.parameter_count(n_para, mode),
    })
}

/// The QR basis of `m`'s columns, refused unless they are linearly independent: for dependent columns QR fills the
/// missing directions with arbitrary vectors, waveforms the basis never had.  The rank is numpy's `matrix_rank`: the
/// singular values above `σ_max · max(rows, cols) · ε`.
fn orthonormal(m: &RMat<f64>, name: &str) -> Result<RMat<f64>> {
    let s = nalgebra::DMatrix::from_fn(m.rows, m.cols, |r, c| m.get(r, c)).singular_values();
    let tol = s.max() * m.rows.max(m.cols) as f64 * f64::EPSILON;
    let rank = s.iter().filter(|&&v| v > tol).count();
    if rank < m.cols {
        return Err(Error::Config(format!(
            "the {name} basis has only {rank} independent functions of the {} requested on {} points; use fewer \
             parameters or more points",
            m.cols, m.rows
        )));
    }
    Ok(qr_q(m))
}

/// `n_para` raw coefficients drawn uniformly from `[−1, 1)`, as numpy's `uniform(-1, 1, n_para)` does.
pub fn raw_coefficients(n_para: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n_para).map(|_| -1.0 + 2.0 * rng.random::<f64>()).collect()
}

/// The initial optimiser parameters for one qubit from its raw coefficients.
pub fn initial_params(basis: &dyn Basis, raw: &[f64], mode: WaveformMode) -> Vec<f64> {
    let mut x = basis.initial_coefficients(raw, mode);
    if mode == WaveformMode::PolarPhase {
        if let Some(&last) = x.last() {
            x.push(last);
        }
    }
    x
}

#[cfg(test)]
mod tests;
