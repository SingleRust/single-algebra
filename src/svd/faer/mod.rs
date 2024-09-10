use faer_ext::*;
use ndarray::{Array1, Array2, ArrayView2};

pub struct SVD {
    u: Array2<f64>,
    s: Array1<f64>,
    vt: Array2<f64>,
}

impl SVD {
    pub fn new(array: &ArrayView2<f64>) -> Self {
        let faer_mat = array.into_faer();
        let svd = faer_mat.svd();
        let u = svd.u().into_ndarray().to_owned();
        let s: Array1<f64> = Array1::from_iter(svd.s_diagonal().iter().cloned());
        let vt = svd.v().into_ndarray().to_owned();

        SVD { u, s, vt }
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_simple_svd() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let svd = SVD::new(&a.view());
        let s = svd.s();
        let vt = svd.vt();
        let u = svd.u();
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
