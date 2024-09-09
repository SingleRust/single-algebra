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
