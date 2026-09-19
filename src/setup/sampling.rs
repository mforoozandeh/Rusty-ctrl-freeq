//! Sampling the uncertain parameters: offsets per coverage mode, couplings and Rabi frequencies.
//!
//! The distributions and their structure follow the Python package; the sampled values differ because the random
//! number generators do.

use rand::RngExt;
use rand_distr::{Distribution, Normal};

use super::Rng;
use crate::config::Coverage;
use crate::linalg::RMat;

/// Offsets for one qubit, all in rad/s.
pub(crate) struct QubitOffsets {
    pub coverage: Coverage,
    pub delta: f64,
    pub sigma: f64,
    pub sw: f64,
    pub bandwidth: f64,
    pub ratio: f64,
    pub profile_order: u32,
}

fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.random::<f64>()
}

fn normal(rng: &mut Rng, mean: f64, sd: f64) -> f64 {
    if sd == 0.0 {
        return mean;
    }
    Normal::new(mean, sd).map(|d| d.sample(rng)).unwrap_or(mean)
}

/// The number of offsets placed outside the band: `M·ratio` rounded half-to-even, made even.
fn outside_count(m: usize, ratio: f64) -> usize {
    let mut n = (m as f64 * ratio).round_ties_even() as usize;
    if n % 2 == 1 {
        n += 1;
    }
    if n > m {
        n = m - m % 2;
    }
    n
}

/// Sample `m` offsets for one qubit.
pub(crate) fn sample_offsets(q: &QubitOffsets, m: usize, rng: &mut Rng) -> Vec<f64> {
    let (lo, hi) = (q.delta - q.sw / 2.0, q.delta + q.sw / 2.0);
    let (band_lo, band_hi) = (q.delta - q.bandwidth / 2.0, q.delta + q.bandwidth / 2.0);
    match q.coverage {
        Coverage::Broadband => (0..m).map(|_| uniform(rng, lo, hi)).collect(),
        Coverage::Single => (0..m).map(|_| normal(rng, q.delta, q.sigma)).collect(),
        Coverage::Selective | Coverage::BandSelective => {
            let outside = outside_count(m, q.ratio);
            let inside = m - outside;
            let left: Vec<f64> = (0..outside / 2).map(|_| uniform(rng, lo, band_lo)).collect();
            let right: Vec<f64> = (0..outside / 2).map(|_| uniform(rng, band_hi, hi)).collect();
            let middle: Vec<f64> = (0..inside)
                .map(|_| {
                    if q.coverage == Coverage::BandSelective {
                        uniform(rng, band_lo, band_hi)
                    } else {
                        normal(rng, q.delta, q.sigma)
                    }
                })
                .collect();
            left.into_iter().chain(middle).chain(right).collect()
        }
    }
}

/// The excitation profile at `offsets`: where, and how strongly, the target applies.
pub(crate) fn excitation_profile(q: &QubitOffsets, offsets: &[f64]) -> Vec<f64> {
    match q.coverage {
        Coverage::Broadband | Coverage::Single => vec![1.0; offsets.len()],
        Coverage::Selective => offsets
            .iter()
            .map(|&o| {
                if o >= q.delta - q.bandwidth / 2.0 && o <= q.delta + q.bandwidth / 2.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        Coverage::BandSelective => {
            let s = q.bandwidth / (2.0 * (2.0 * 2f64.ln()).sqrt());
            offsets
                .iter()
                .map(|&o| (-((o - q.delta) / s).powi(2 * q.profile_order as i32)).exp())
                .collect()
        }
    }
}

/// `m` coupling instances: every non-zero entry of the upper-triangular `j` drawn from `normal(j, sigma)`.
pub(crate) fn coupling_instances(j: &RMat<f64>, sigma: f64, m: usize, rng: &mut Rng) -> Vec<RMat<f64>> {
    (0..m)
        .map(|_| {
            let mut out = j.clone();
            for v in out.data.iter_mut() {
                if *v != 0.0 {
                    *v = normal(rng, *v, sigma);
                }
            }
            out
        })
        .collect()
}

/// Rabi-frequency instances: one `[Ω_q]` when every σ is zero, otherwise `count` draws of `normal(Ω_q, σ_q)`.
pub(crate) fn rabi_instances(omega: &[f64], sigma: &[f64], count: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    if sigma.iter().all(|&s| s == 0.0) {
        return vec![omega.to_vec()];
    }
    (0..count)
        .map(|_| omega.iter().zip(sigma).map(|(&o, &s)| normal(rng, o, s)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn offsets(coverage: Coverage) -> QubitOffsets {
        QubitOffsets {
            coverage,
            delta: 10.0,
            sigma: 0.5,
            sw: 8.0,
            bandwidth: 2.0,
            ratio: 0.5,
            profile_order: 2,
        }
    }

    #[test]
    fn band_selective_splits_the_samples_around_the_band() {
        let mut rng = Rng::seed_from_u64(3);
        let q = offsets(Coverage::BandSelective);
        let s = sample_offsets(&q, 100, &mut rng);
        assert_eq!(s.len(), 100);
        assert!(s[..25].iter().all(|&o| (6.0..9.0).contains(&o)));
        assert!(s[25..75].iter().all(|&o| (9.0..11.0).contains(&o)));
        assert!(s[75..].iter().all(|&o| (11.0..14.0).contains(&o)));
        let p = excitation_profile(&q, &[10.0, 11.0, 13.0]);
        assert_eq!(p[0], 1.0);
        assert!(p[1] < 1.0 && p[2] < p[1]);
    }

    #[test]
    fn outside_counts_round_half_to_even_and_are_even() {
        assert_eq!(outside_count(100, 0.5), 50);
        assert_eq!(outside_count(5, 0.5), 2); // 2.5 rounds to 2
        assert_eq!(outside_count(7, 0.5), 4); // 3.5 rounds to 4
        assert_eq!(outside_count(9, 0.3), 4); // 2.7 -> 3 -> 4
        assert_eq!(outside_count(1, 1.0), 0);
    }

    #[test]
    fn broadband_and_single_cover_their_ranges() {
        let mut rng = Rng::seed_from_u64(4);
        let b = sample_offsets(&offsets(Coverage::Broadband), 200, &mut rng);
        assert!(b.iter().all(|&o| (6.0..14.0).contains(&o)));
        let s = sample_offsets(&offsets(Coverage::Single), 2000, &mut rng);
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        assert!((mean - 10.0).abs() < 0.05);
        let sel = excitation_profile(&offsets(Coverage::Selective), &[9.5, 11.5]);
        assert_eq!(sel, vec![1.0, 0.0]);
    }

    #[test]
    fn coupling_noise_leaves_zeros_alone() {
        let mut rng = Rng::seed_from_u64(5);
        let mut j = RMat::<f64>::zeros(3, 3);
        j.set(0, 1, 4.0);
        let inst = coupling_instances(&j, 0.1, 5, &mut rng);
        for m in &inst {
            assert!(m.get(0, 2) == 0.0 && m.get(1, 2) == 0.0 && m.get(1, 0) == 0.0);
            assert!((m.get(0, 1) - 4.0).abs() < 1.0);
        }
    }

    #[test]
    fn rabi_without_spread_is_one_instance() {
        let mut rng = Rng::seed_from_u64(6);
        assert_eq!(
            rabi_instances(&[1.0, 2.0], &[0.0, 0.0], 5, &mut rng),
            vec![vec![1.0, 2.0]]
        );
        assert_eq!(rabi_instances(&[1.0], &[0.1], 5, &mut rng).len(), 5);
    }
}
