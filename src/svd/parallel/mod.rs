use std::sync::{Arc, RwLock};
use anyhow::anyhow;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

pub struct SVD {
    u: Array2<f64>,
    s: Array1<f64>,
    vt: Array2<f64>,
}

impl SVD {
    pub fn new(matrix: ArrayView2<f64>) -> Self {
        let (m,n) = matrix.dim();
        let u = RwLock::new(matrix.to_owned());
        let mut s = Array1::zeros(n.min(m));
        let vt = RwLock::new(Array2::eye(n));


        SVD { 
            u: u.into_inner().unwrap(), 
            s, 
            vt: vt.into_inner().unwrap() 
        }
    }

    fn bidiagonalize(u: &RwLock<Array2<f64>>, vt: &RwLock<Array2<f64>>) -> anyhow::Result<()> {
        let (m, n) = {
            let u_read = u.read().unwrap();
            u_read.dim()
        };
        let min_dim = m.min(n);

        for i in 0..min_dim {
            // Compute Householder reflection for column
            let mut u_write = u.write().unwrap();
            let (h_col, beta_col) = Self::householder_reflection(u_write.slice_mut(s![i.., i]));
            
            // Apply Householder reflection to U
            Self::apply_householder(&mut u_write, &h_col, beta_col, (i, i), false);
            drop(u_write);

            if i < min_dim - 1 {
                // Compute Householder reflection for row
                let mut u_write = u.write().unwrap();
                let (h_row, beta_row) = Self::householder_reflection(u_write.slice_mut(s![i, i+1..]));
                
                // Apply Householder reflection to U and V^T
                Self::apply_householder(&mut u_write, &h_row, beta_row, (i, i+1), true);
                drop(u_write);
                
                let mut vt_write = vt.write().unwrap();
                Self::apply_householder(&mut vt_write, &h_row, beta_row, (0, i+1), false);
            }
        }

        Ok(())
    }

    fn householder_reflection(x: ArrayView2<f64>) -> (Array1<f64>, f64) {
        let mut v = x.to_owned().into_raw_vec();
        let n = v.len();
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let alpha = -v[0].signum() * norm;
        v[0] -= alpha;
        let beta = v[0].abs();
        if beta != 0.0 {
            for i in 0..n {
                v[i] /= beta;
            }
        }
        (Array1::from(v), beta * beta / 2.0)
    }

    fn apply_householder(
        matrix: &mut Array2<f64>,
        h: &Array1<f64>,
        beta: f64,
        (row_start, col_start): (usize, usize),
        transpose: bool,
    ) {
        let h = h.to_owned();
        let h_len = h.len();

        if transpose {
            let (m, _) = matrix.dim();
            let chunks: Vec<_> = (0..m).collect();
            chunks.par_chunks(m / rayon::current_num_threads().max(1)).for_each(|chunk| {
                for &i in chunk {
                    let dot = (0..h_len).map(|j| matrix[(i, j + col_start)] * h[j]).sum::<f64>();
                    for j in 0..h_len {
                        matrix[(i, j + col_start)] -= 2.0 * beta * dot * h[j];
                    }
                }
            });
        } else {
            let (_, n) = matrix.dim();
            let chunks: Vec<_> = (0..n).collect();
            chunks.par_chunks(n / rayon::current_num_threads().max(1)).for_each(|chunk| {
                for &j in chunk {
                    let dot = (0..h_len).map(|i| matrix[(i + row_start, j)] * h[i]).sum::<f64>();
                    for i in 0..h_len {
                        matrix[(i + row_start, j)] -= 2.0 * beta * dot * h[i];
                    }
                }
            });
        }
    }

    fn diagonalize_step(
        u: &RwLock<Array2<f64>>,
        vt: &RwLock<Array2<f64>>,
        min_dim: usize,
        epsilon: f64,
    ) -> anyhow::Result<f64> {
        let max_change = (0..min_dim-1)
            .into_par_iter()
            .map(|i| {
                let mut local_max_change = 0.0f64;
                for j in i+1..min_dim {
                    let (c, s) = {
                        let u_read = u.read().unwrap();
                        Self::compute_rotation(&u_read, i, j)
                    };
                    if s.abs() > epsilon {
                        let change = Self::apply_rotation(u, vt, i, j, c, s);
                        local_max_change = local_max_change.max(change);
                    }
                }
                local_max_change
            })
            .reduce_with(f64::max)
            .unwrap_or(0.0);

        Ok(max_change)
    }

    fn column_norm(matrix: &Array2<f64>, col: usize) -> f64 {
        matrix.column(col).map(|&x| x * x).sum().sqrt()
    }

    fn compute_rotation(u: &Array2<f64>, i: usize, j: usize) -> (f64, f64) {
        let a = u.column(i).dot(&u.column(i));
        let b = u.column(i).dot(&u.column(j));
        let d = u.column(j).dot(&u.column(j));
        let zeta = (d - a) / (2.0 * b);
        let t = if zeta > 0.0 {
            1.0 / (zeta + (zeta * zeta + 1.0).sqrt())
        } else {
            -1.0 / (-zeta + (zeta * zeta + 1.0).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        (c, s)
    }

    fn apply_rotation(u: &RwLock<Array2<f64>>, vt: &RwLock<Array2<f64>>, i: usize, j: usize, c: f64, s: f64) -> f64 {
        let mut u_write = u.write().unwrap();
        let mut vt_write = vt.write().unwrap();

        let mut max_change = 0.0f64;

        for k in 0..u_write.nrows() {
            let temp = u_write[(k, i)];
            let new_ui = c * temp - s * u_write[(k, i)];
            let new_uj = s * temp + c * u_write[(k, j)];

            max_change = max_change.max((new_ui - u_write[(k, i)]).abs());
            max_change = max_change.max((new_uj - u_write[(k, j)]).abs());
            u_write[(k, i)] = new_ui;
            u_write[(k, j)] = new_uj;
        }

        for k in 0..vt_write.ncols() {
            let temp = vt_write[(i, k)];
            let new_vi = c * temp - s * vt_write[(j, k)];
            let new_vj = s * temp + c * vt_write[(j, k)];
            max_change = max_change.max((new_vi - vt_write[(i, k)]).abs());
            max_change = max_change.max((new_vj - vt_write[(j, k)]).abs());
            vt_write[(i, k)] = new_vi;
            vt_write[(j, k)] = new_vj;
        }

        max_change

    }

    fn reorder_matrices(u: &mut Array2<f64>, s: &mut Array1<f64>, vt: &mut Array2<f64>, order: &[usize])  {
        let (m ,n) = u.dim();
        let min_dim = n.min(m);

        let u_temp = u.clone();
        let s_temp = s.clone();
        let vt_temp = vt.clone();

        for (i, &idx) in order.iter().enumerate() {
            if i < min_dim {
                s[i] = s_temp[idx];
                u.column_mut(i).assign(&u_temp.column(idx));
                vt.row_mut(i).assign(&vt_temp.row(idx));
            }
        }
    }

    pub fn u(&self) -> &Array2<f64> {
        &self.u
    }

    pub fn s(&self) -> &Array1<f64> {
        &self.s
    }

    pub fn vt(&self) -> &Array2<f64> {
        &self.vt
    }
}