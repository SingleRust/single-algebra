use ndarray::{Array1, Array2, ArrayView2};
use nshare::{ToNalgebra, ToNdarray2};
use rayon::ThreadPool;

pub struct SVD {
    u: Option<Array2<f64>>,
    s: Option<Array1<f64>>,
    vt: Option<Array2<f64>>,
}

impl SVD {
    pub fn new() -> Self {
        SVD {
            u: None,
            s: None,
            vt: None,
        }
    }

    pub fn compute(&mut self, x: ArrayView2<f64>, thread_pool: &ThreadPool) -> anyhow::Result<()> {
        let matrix = x.into_nalgebra().clone_owned();

        let svd = thread_pool
            .install(|| nalgebra_lapack::SVD::new(matrix))
            .unwrap();

        self.u = Some(svd.u.());
        self.s = Some(Array1::from(svd.singular_values.as_slice().to_vec()));
        self.vt = Some(svd.vt.into_ndarray2());

        Ok(())
    }

    pub fn u(&self) -> Option<&Array2<f64>> {
        self.u.as_ref()
    }

    pub fn s(&self) -> Option<&Array1<f64>> {
        self.s.as_ref()
    }

    pub fn vt(&self) -> Option<&Array2<f64>> {
        self.vt.as_ref()
    }

    // Reconstruct the original matrix
    pub fn reconstruct(&self) -> Option<Array2<f64>> {
        match (self.u(), self.s(), self.vt()) {
            (Some(u), Some(s), Some(vt)) => {
                let s_diag = Array2::from_diag(s);
                Some(u.dot(&s_diag).dot(vt))
            }
            _ => None,
        }
    }
}

impl Default for SVD {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rayon::ThreadPoolBuilder;

    use super::*;

    #[test]
    fn test_simple_svd() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let mut svd = SVD::new();
        let tb = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        svd.compute(a.view(), &tb).unwrap();
        let s = svd.s().unwrap();
        let vt = svd.vt().unwrap();
        let u = svd.u().unwrap();
        // Check dimensions
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 2]);

        // Check singular values (pre-computed)
        assert_abs_diff_eq!(s[0], 5.4649857, epsilon = 1e-6);
        assert_abs_diff_eq!(s[1], 0.3659662, epsilon = 1e-6);

        // Check reconstruction
        let s_diag = ndarray::Array2::from_diag(s);
        let reconstructed = u.dot(&s_diag).dot(vt);
        
        for i in 0..2 { 
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
