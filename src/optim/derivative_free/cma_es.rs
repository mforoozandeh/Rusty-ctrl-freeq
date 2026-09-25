//! CMA-ES, the covariance matrix adaptation evolution strategy.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use super::{bounds, clamp, finish};
use crate::error::Result;
use crate::optim::{Derivatives, Exit, Monitor, Objective, OptimResult, Optimizer, RunControl};
use crate::setup::Rng;

/// Width of the first generation, in parameter units.  The same scale BOBYQA gives its first trust region.
const SIGMA0: f64 = 0.5;

/// Converged once the distribution has shrunk to this along every axis: no generation can tell its members
/// apart any more.
const XTOL: f64 = 1e-11;

/// CMA-ES with the default population and the weights, learning rates and damping of Hansen's reference
/// implementation.  The distribution starts spherical with standard deviation 0.5 around the starting point,
/// and every sample is drawn, clamped into the box and evaluated, so one iteration is still one evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct CmaEs;

impl Optimizer for CmaEs {
    fn name(&self) -> &'static str {
        "cma-es"
    }

    fn derivatives(&self) -> Derivatives {
        Derivatives::None
    }

    fn minimize(&self, obj: &dyn Objective, x0: &[f64], ctl: &mut RunControl) -> Result<OptimResult> {
        finish(Monitor::new(ctl), |m| search(obj, x0, m))
    }
}

/// The strategy's fixed parameters: everything derived from the number of parameters alone.
struct Strategy {
    n: usize,
    /// Population size.  The best `weights.len()` of each generation are recombined.
    lambda: usize,
    /// Recombination weights, summing to 1, and the variance-effective size they imply.
    weights: Vec<f64>,
    mueff: f64,
    /// Learning rates for the two evolution paths and the two covariance updates, and the step-size damping.
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    /// Expected length of a standard normal vector, the yardstick the step size is adapted against.
    chin: f64,
}

impl Strategy {
    fn new(n: usize) -> Strategy {
        let nf = n as f64;
        let lambda = 4 + (3.0 * nf.ln()).floor() as usize;
        let mu = lambda / 2;
        let raw: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff)).min(1.0 - c1);
        let damps = 1.0 + 2.0 * (0.0f64).max(((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0) + cs;
        let chin = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        Strategy {
            n,
            lambda,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chin,
        }
    }
}

/// The covariance matrix in the form the sampler and the step-size rule need it: `c = b diag(d)^2 b^T`.
struct Shape {
    c: DMatrix<f64>,
    b: DMatrix<f64>,
    d: DVector<f64>,
    /// `c^(-1/2)`, which turns a step in parameter space into one in the sphere the path length is measured in.
    inv_sqrt_c: DMatrix<f64>,
}

impl Shape {
    fn identity(n: usize) -> Shape {
        Shape {
            c: DMatrix::identity(n, n),
            b: DMatrix::identity(n, n),
            d: DVector::from_element(n, 1.0),
            inv_sqrt_c: DMatrix::identity(n, n),
        }
    }

    /// Re-derive `b`, `d` and `c^(-1/2)` from `c`, forcing the symmetry that rounding erodes.
    fn decompose(&mut self) {
        self.c = (&self.c + self.c.transpose()) * 0.5;
        let eigen = SymmetricEigen::new(self.c.clone());
        // Rounding can push a tiny eigenvalue below zero; the matrix is a covariance, so floor it instead.
        self.d = eigen.eigenvalues.map(|v| v.max(f64::MIN_POSITIVE).sqrt());
        self.b = eigen.eigenvectors;
        let inv_d = DMatrix::from_diagonal(&self.d.map(|v| 1.0 / v));
        self.inv_sqrt_c = &self.b * inv_d * self.b.transpose();
    }
}

fn search(obj: &dyn Objective, x0: &[f64], m: &mut Monitor) -> Result<Exit> {
    let s = Strategy::new(x0.len().max(1));
    let (lower, upper) = bounds(x0);
    let mut rng = Rng::seed_from_u64(m.seed);
    let normal = Normal::new(0.0, 1.0).expect("a unit normal is well formed");

    let mut mean = DVector::from_column_slice(x0);
    let mut sigma = SIGMA0;
    let mut path_c = DVector::zeros(s.n);
    let mut path_s = DVector::zeros(s.n);
    let mut shape = Shape::identity(s.n);
    // The decomposition only has to keep up with the covariance update, so it is redone every so many
    // evaluations rather than every generation.
    let stale_after = (0.5 / ((s.c1 + s.cmu) * s.n as f64)).max(1.0);
    let mut decomposed_at = 0.0;

    loop {
        // One generation: sample, clamp into the box and evaluate.
        let mut population: Vec<(Vec<f64>, f64)> = Vec::with_capacity(s.lambda);
        for _ in 0..s.lambda {
            let z = DVector::from_fn(s.n, |_, _| normal.sample(&mut rng));
            let step = &shape.b * z.component_mul(&shape.d);
            let mut x: Vec<f64> = (&mean + sigma * step).iter().copied().collect();
            clamp(&mut x, &lower, &upper);
            let f = m.evaluate(obj, &x)?;
            population.push((x, f));
            if m.exit.is_some() {
                return Ok(Exit::Converged);
            }
        }
        population.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Recombination: the new mean is the weighted average of the best `mu` samples.
        let old_mean = mean.clone();
        mean = DVector::zeros(s.n);
        for (w, (x, _)) in s.weights.iter().zip(&population) {
            mean += *w * DVector::from_column_slice(x);
        }
        let displacement = (&mean - &old_mean) / sigma;

        // Step-size path: how far the mean has travelled, measured in the sphere `c^(-1/2)` maps to.
        path_s = (1.0 - s.cs) * &path_s + (s.cs * (2.0 - s.cs) * s.mueff).sqrt() * (&shape.inv_sqrt_c * &displacement);
        let evaluations = m.evaluations as f64;
        let expected = (1.0 - (1.0 - s.cs).powf(2.0 * evaluations / s.lambda as f64)).sqrt();
        // Stall the rank-one update while the path is unusually long, so a run of aligned steps does not make
        // the covariance grow along with the step size.
        let hsig = path_s.norm() / expected / s.chin < 1.4 + 2.0 / (s.n as f64 + 1.0);
        let hsig_f = if hsig { 1.0 } else { 0.0 };
        path_c = (1.0 - s.cc) * &path_c + hsig_f * (s.cc * (2.0 - s.cc) * s.mueff).sqrt() * &displacement;

        // Covariance: a rank-one term along the path, and a rank-`mu` term from the selected samples.
        let rank_one = &path_c * path_c.transpose() + (1.0 - hsig_f) * s.cc * (2.0 - s.cc) * &shape.c;
        let mut rank_mu = DMatrix::zeros(s.n, s.n);
        for (w, (x, _)) in s.weights.iter().zip(&population) {
            let y = (DVector::from_column_slice(x) - &old_mean) / sigma;
            rank_mu += *w * (&y * y.transpose());
        }
        shape.c = (1.0 - s.c1 - s.cmu) * &shape.c + s.c1 * rank_one + s.cmu * rank_mu;

        // Step size: grow it while the path is longer than a random walk would be, shrink it when shorter.
        sigma *= ((s.cs / s.damps) * (path_s.norm() / s.chin - 1.0)).exp();
        if !sigma.is_finite() || sigma <= 0.0 {
            return Ok(Exit::Converged);
        }

        if evaluations - decomposed_at > stale_after {
            decomposed_at = evaluations;
            shape.decompose();
        }
        // The distribution has collapsed, or the generation could not tell its members apart.
        if sigma * shape.d.max() < XTOL || population[0].1 == population[s.lambda - 1].1 {
            return Ok(Exit::Converged);
        }
    }
}
