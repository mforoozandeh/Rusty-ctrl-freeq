//! The cost as a graph on the autodiff tape.

use crate::autodiff::{Scalar, Tape, Value, Var};
use crate::config::WaveformMode;
use crate::error::Result;
use crate::hamiltonian::Source;
use crate::linalg::CMat;
use crate::setup::{BatchElement, Evolution, Problem};

/// The waveform nodes for every qubit.
pub(crate) struct WaveformVars {
    /// `n_pulse × n_qubits` amplitudes before modulation.
    pub amp: Var,
    /// `n_pulse × n_qubits` modulated in-phase quadratures.
    pub cx: Var,
    /// `n_pulse × n_qubits` modulated quadratures.
    pub cy: Var,
}

/// Record the waveforms `x` describes: per-qubit basis expansions, then the carrier modulation.
pub(crate) fn build_waveforms<'a, T: Scalar>(p: &'a Problem, tape: &mut Tape<'a, T>, x: Var) -> Result<WaveformVars> {
    let pi = std::f64::consts::PI;
    let mut amps = Vec::with_capacity(p.n_qubits);
    let mut cxs = Vec::with_capacity(p.n_qubits);
    let mut cys = Vec::with_capacity(p.n_qubits);
    let mut start = 0;
    for qb in &p.qubits {
        let params = tape.slice(x, start, qb.n_params)?;
        start += qb.n_params;
        let (amp, cx, cy) = match qb.mode {
            WaveformMode::Cart => {
                let l = qb.n_params / 2;
                let c1 = tape.slice(params, 0, l)?;
                let c2 = tape.slice(params, l, l)?;
                let cx = tape.matmul_const(&qb.q[0], c1)?;
                let cy = tape.matmul_const(&qb.q[1], c2)?;
                let x2 = tape.square(cx)?;
                let y2 = tape.square(cy)?;
                let r2 = tape.add(x2, y2)?;
                (tape.sqrt(r2)?, cx, cy)
            }
            WaveformMode::Polar | WaveformMode::PolarPhase => {
                let (amp, phase_coeffs) = if qb.mode == WaveformMode::Polar {
                    let l = qb.n_params / 2;
                    let c1 = tape.slice(params, 0, l)?;
                    (tape.matmul_const(&qb.q[0], c1)?, tape.slice(params, l, l)?)
                } else {
                    let a = tape.slice(params, 0, 1)?;
                    (
                        tape.matmul_const(&qb.envelope, a)?,
                        tape.slice(params, 1, qb.n_params - 1)?,
                    )
                };
                let scaled = tape.scale(phase_coeffs, pi)?;
                let phi = tape.matmul_const(&qb.q[1], scaled)?;
                let cos = tape.cos(phi)?;
                let sin = tape.sin(phi)?;
                (amp, tape.mul(amp, cos)?, tape.mul(amp, sin)?)
            }
        };
        amps.push(amp);
        cxs.push(cx);
        cys.push(cy);
    }
    let amp = tape.hstack(&amps)?;
    let cx = tape.hstack(&cxs)?;
    let cy = tape.hstack(&cys)?;
    let z = tape.complex(cx, cy)?;
    let w = tape.mul_const_c(z, &p.modulation)?;
    Ok(WaveformVars {
        amp,
        cx: tape.real(w)?,
        cy: tape.imag(w)?,
    })
}

/// Record the cost for parameters `x`: `(cost, fidelity, penalty)`.
pub(crate) fn build_cost<'a, T: Scalar>(p: &'a Problem, tape: &mut Tape<'a, T>, x: Var) -> Result<(Var, Var, Var)> {
    let w = build_waveforms(p, tape, x)?;
    let penalty = tape.amp_penalty(w.amp)?;
    let fidelity = tape.batch_mean(&[w.cx, w.cy], p.batch.len(), |sub, inputs, b| {
        element_fidelity(p, &p.batch[b], sub, inputs[0], inputs[1])
    })?;
    let cost = tape.sub(penalty, fidelity)?;
    Ok((cost, fidelity, penalty))
}

/// The control amplitudes `u[t, k]` for one batch element, as an `n_pulse × K` node.
fn controls<'a, T: Scalar>(p: &'a Problem, e: &BatchElement, tape: &mut Tape<'a, T>, cx: Var, cy: Var) -> Result<Var> {
    let mut cols = Vec::with_capacity(p.channels.len());
    for ch in &p.channels {
        let source = match ch.source {
            Source::Cx => tape.col(cx, ch.qubit)?,
            Source::Cy => tape.col(cy, ch.qubit)?,
            Source::Power => {
                let a = tape.col(cx, ch.qubit)?;
                let b = tape.col(cy, ch.qubit)?;
                let a2 = tape.square(a)?;
                let b2 = tape.square(b)?;
                tape.add(a2, b2)?
            }
        };
        let factor = ch.coeff * e.rabi[ch.qubit].powi(ch.rabi_power);
        cols.push(tape.scale(source, factor)?);
    }
    tape.hstack(&cols)
}

/// The fidelity of one batch element after the whole pulse.
fn element_fidelity<'a, T: Scalar>(
    p: &'a Problem,
    e: &'a BatchElement,
    tape: &mut Tape<'a, T>,
    cx: Var,
    cy: Var,
) -> Result<Var> {
    let u = controls(p, e, tape, cx, cy)?;
    let mut state = tape.constant(Value::C(CMat::<T>::lift(&e.initial)));
    for t in 0..p.n_pulse {
        let h = tape.lincomb_row(&e.h0, u, t, &p.control_ops)?;
        let step = tape.expm_mi_dt(h, p.dt)?;
        state = match &p.evolution {
            Evolution::Hilbert => tape.matmul(step, state)?,
            Evolution::Liouville => tape.sandwich(step, state)?,
            Evolution::Lindblad(ops) => {
                let rho = tape.sandwich(step, state)?;
                tape.lindblad_step(rho, ops)?
            }
        };
    }
    match &p.evolution {
        Evolution::Hilbert => tape.fidelity(&e.target, state),
        Evolution::Liouville | Evolution::Lindblad(_) => tape.re_trace_product(&e.target, state),
    }
}
