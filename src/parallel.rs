//! Spreading work over threads without changing its result.
//!
//! With the `parallel` feature, which is on by default, [`fold`] runs on [rayon](https://docs.rs/rayon)'s thread
//! pool: one thread per core unless `RAYON_NUM_THREADS` says otherwise, or the pool a caller has entered with
//! `rayon::ThreadPool::install`.  Without the feature, and always on `wasm32`, which has no threads, the same code
//! runs on the calling thread.
//!
//! This is the only module in the crate that knows about threads.  The batch evaluation in the autodiff tape and
//! the Hessian's columns go through it; optimisers, Hamiltonian models and bases never do.
//!
//! # Determinism
//!
//! Floating-point addition is not associative.  A sum split the way rayon's own `reduce` splits it - wherever an
//! idle thread happens to steal work - comes out differently in its last digits from run to run, and an optimiser
//! can grow those digits into a different path altogether.  [`fold`] instead cuts the work into pieces fixed by
//! its length and a piece count its caller chooses, and merges them up a fixed tree.  Threads decide where a piece
//! runs, never how the results combine, so one thread, twelve, or the sequential build give bit-for-bit the same
//! answer.

/// The piece count used throughout the crate.  Fixed, never derived from the thread count, so results do not
/// depend on the machine.
pub(crate) const PIECES: usize = 64;

/// Fold every index in `0..len` into one accumulator.
///
/// The range is cut into at most `pieces` contiguous pieces, all of one length but the last.  Each piece is one
/// task: it starts from `init()` and applies `step` to its indices in order.  Neighbouring pieces are then
/// combined, left to right, by `merge(&mut left, right)` up a balanced binary tree.  An empty range gives `init()`.
pub(crate) fn fold<T, I, S, M>(len: usize, pieces: usize, init: I, step: S, merge: M) -> T
where
    T: Send,
    I: Fn() -> T + Sync,
    S: Fn(&mut T, usize) + Sync,
    M: Fn(&mut T, T) + Sync,
{
    let piece_len = len.div_ceil(pieces.max(1)).max(1);
    let work = Work {
        len,
        piece_len,
        init,
        step,
        merge,
    };
    work.pieces(0, len.div_ceil(piece_len))
}

/// Evaluate `f` at every index in `0..len`, possibly in parallel, and return the results in index order.
pub(crate) fn map<T, F>(len: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    fold(
        len,
        PIECES,
        Vec::new,
        |acc, i| acc.push(f(i)),
        |left, right| left.extend(right),
    )
}

/// A thread pool to run evaluations in: rayon's global pool, or one of a chosen size.  Without the `parallel`
/// feature, and on `wasm32`, it runs everything on the calling thread.
#[derive(Default)]
pub(crate) struct Pool {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pool: Option<rayon::ThreadPool>,
}

impl Pool {
    /// A pool of `threads` threads.  `threads` is clamped to at least one.
    pub(crate) fn with_threads(threads: usize) -> crate::Result<Pool> {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build()
                .map_err(|e| crate::Error::NotSupported(format!("cannot start {threads} threads: {e}")))?;
            Ok(Pool { pool: Some(pool) })
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        {
            let _ = threads;
            Ok(Pool {})
        }
    }

    /// Run `f` with this pool as the one [`fold`] and [`map`] use.
    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        if let Some(pool) = &self.pool {
            return pool.install(f);
        }
        f()
    }
}

/// Number of threads evaluations run on by default.
pub fn available_threads() -> usize {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    {
        1
    }
}

struct Work<I, S, M> {
    len: usize,
    piece_len: usize,
    init: I,
    step: S,
    merge: M,
}

impl<T, I, S, M> Work<I, S, M>
where
    T: Send,
    I: Fn() -> T + Sync,
    S: Fn(&mut T, usize) + Sync,
    M: Fn(&mut T, T) + Sync,
{
    /// Pieces `lo..hi`, combined.
    fn pieces(&self, lo: usize, hi: usize) -> T {
        if hi - lo <= 1 {
            let mut acc = (self.init)();
            for i in lo * self.piece_len..(hi * self.piece_len).min(self.len) {
                (self.step)(&mut acc, i);
            }
            return acc;
        }
        let mid = lo + (hi - lo) / 2;
        let (mut left, right) = join(|| self.pieces(lo, mid), || self.pieces(mid, hi));
        (self.merge)(&mut left, right);
        left
    }
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use rayon::join;

#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
fn join<A, B>(a: impl FnOnce() -> A, b: impl FnOnce() -> B) -> (A, B) {
    (a(), b())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every index is visited exactly once, and the merges keep them in order, for lengths either side of a whole
    /// number of pieces.
    #[test]
    fn every_index_is_folded_once_in_order() {
        for pieces in [0, 1, 2, 7, 64] {
            for len in [0, 1, 2, 63, 64, 65, 100, 129, 1000] {
                let got = fold(len, pieces, Vec::new, |v, i| v.push(i), |a, b| a.extend(b));
                assert_eq!(got, (0..len).collect::<Vec<_>>(), "{len} in {pieces}");
            }
        }
    }

    /// No more accumulators are made than pieces were asked for.
    #[test]
    fn the_piece_count_is_an_upper_bound() {
        for (len, pieces) in [(10, 64), (100, 64), (100, 8), (1000, 3), (5, 0)] {
            let made = fold(len, pieces, || 1, |_, _| {}, |a, b| *a += b);
            assert!(made <= pieces.max(1) && made <= len, "{made} for {len} in {pieces}");
        }
    }

    /// `map` keeps index order.
    #[test]
    fn map_preserves_order() {
        for len in [0, 1, 5, 64, 65, 300] {
            assert_eq!(map(len, |i| i * i), (0..len).map(|i| i * i).collect::<Vec<_>>());
        }
    }

    /// A sum whose grouping shows in its last digits comes out the same on one thread as on several.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    #[test]
    fn the_thread_count_does_not_change_the_result() {
        let sum = || {
            fold(
                10_000,
                PIECES,
                || 0.0,
                |s, i| *s += (i as f64).sqrt().sin(),
                |a, b| *a += b,
            )
        };
        let on = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(sum)
        };
        let one = on(1);
        for threads in [2, 3, 8] {
            assert_eq!(one.to_bits(), on(threads).to_bits(), "{threads} threads");
        }
    }
}
