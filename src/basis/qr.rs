//! Orthonormalisation of basis columns.

use crate::linalg::RMat;

/// The `Q` factor of the reduced (thin) Householder QR decomposition of `m` (`rows ≥ cols`).
///
/// This follows LAPACK's `dgeqrf` and `dorgqr` - the routines numpy's `qr` calls - including `dlarfg`'s choice of
/// reflection sign, so the columns match numpy's `qr(m)[0]` in sign as well as in value.  A different sign
/// convention would flip basis functions and with them the meaning of the optimiser's coefficients.
pub fn qr_q(m: &RMat<f64>) -> RMat<f64> {
    let (rows, cols) = m.shape();
    let k = rows.min(cols);
    // Column-major working copy: a[c][r].
    let mut a: Vec<Vec<f64>> = (0..cols).map(|c| m.col(c)).collect();
    let mut tau = vec![0.0; k];

    for j in 0..k {
        let alpha = a[j][j];
        let xnorm = a[j][j + 1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if xnorm == 0.0 {
            tau[j] = 0.0;
            continue;
        }
        // Fortran SIGN(|x|, alpha): positive for alpha = +0.
        let norm = alpha.hypot(xnorm);
        let beta = if alpha >= 0.0 { -norm } else { norm };
        tau[j] = (beta - alpha) / beta;
        let scale = 1.0 / (alpha - beta);
        for v in a[j][j + 1..].iter_mut() {
            *v *= scale;
        }
        a[j][j] = beta;
        // Apply H_j = I − τ·v·vᵀ, v = [1, a[j][j+1..]], to the remaining columns.
        let (head, tail) = a.split_at_mut(j + 1);
        let v = &head[j];
        for col in tail.iter_mut() {
            let dot = col[j] + (j + 1..rows).map(|i| v[i] * col[i]).sum::<f64>();
            let f = tau[j] * dot;
            col[j] -= f;
            for i in j + 1..rows {
                col[i] -= f * v[i];
            }
        }
    }

    // Q = H_0·H_1·…·H_{k−1} applied to the first k columns of the identity, built from the last reflection back.
    let mut q: Vec<Vec<f64>> = (0..k)
        .map(|c| {
            let mut e = vec![0.0; rows];
            e[c] = 1.0;
            e
        })
        .collect();
    for j in (0..k).rev() {
        let v = &a[j];
        for col in q.iter_mut().skip(j) {
            let dot = col[j] + (j + 1..rows).map(|i| v[i] * col[i]).sum::<f64>();
            let f = tau[j] * dot;
            col[j] -= f;
            for i in j + 1..rows {
                col[i] -= f * v[i];
            }
        }
    }
    RMat::from_fn(rows, k, |r, c| q[c][r])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_reproduces_a_with_an_upper_triangular_r() {
        let m = RMat::from_vec(
            4,
            3,
            vec![1.0, 2.0, 0.5, -1.0, 0.3, 2.2, 0.7, -0.4, 1.1, 2.0, 1.0, -3.0],
        )
        .unwrap();
        let q = qr_q(&m);
        let r = q.transpose().matmul(&m).unwrap();
        for i in 0..3 {
            for j in 0..i {
                assert!(r.get(i, j).abs() < 1e-14, "R[{i},{j}]");
            }
        }
        let back = q.matmul(&r).unwrap();
        for (a, b) in back.data.iter().zip(&m.data) {
            assert!((a - b).abs() < 1e-14);
        }
        // LAPACK's sign choice: R's diagonal has the opposite sign of the pivot it replaced.
        assert!(r.get(0, 0) < 0.0);
    }
}
