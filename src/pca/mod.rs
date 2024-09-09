use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use ndarray::parallel::prelude::*;
use std::sync::Arc;

// Trait for SVD implementations
pub trait SVDImplementation: Send + Sync {
    fn compute(&self, matrix: ArrayView2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>);
}

pub struct PCABuilder<S: SVDImplementation> {
    n_components: Option<usize>,
    center: bool,
    scale: bool,
    svd_implementation: Arc<S>,
}

impl<S: SVDImplementation> PCABuilder<S> {
    pub fn new(svd_implementation: S) -> Self {
        PCABuilder {
            n_components: None,
            center: true,
            scale: false,
            svd_implementation: Arc::new(svd_implementation),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> Pca<S> {
        Pca {
            n_components: self.n_components,
            center: self.center,
            scale: self.scale,
            svd_implementation: self.svd_implementation,
            components: None,
            mean: None,
            std_dev: None,
            explained_variance_ratio: None,
            total_variance: None,
            eigenvalues: None,
        }
    }
}

pub struct Pca<S: SVDImplementation> {
    n_components: Option<usize>,
    center: bool,
    scale: bool,
    svd_implementation: Arc<S>,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    std_dev: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    total_variance: Option<f64>,
    eigenvalues: Option<Array1<f64>>,
}

impl<S: SVDImplementation> Pca<S> {
    pub fn fit(&mut self, x: ArrayView2<f64>) -> anyhow::Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_components = self.n_components.unwrap_or(n_features);

        // Center the data
        let mean = if self.center {
            Some(x.mean_axis(Axis(0)).expect("Failed to compute mean"))
        } else {
            None
        };

        // Scale the data
        let std_dev = if self.scale {
            Some(x.std_axis(Axis(0), 0.0))
        } else {
            None
        };

        // Preprocess the data (center and scale)
        let x_preprocessed = self.preprocess(x, &mean, &std_dev);

        // Compute SVD using the provided implementation
        let (_u, s, vt) = self.svd_implementation.compute(x_preprocessed.view());

        // Extract principal components and eigenvalues
        let components = vt.slice(s![..n_components, ..]).to_owned();
        let eigenvalues = s.mapv(|x| x * x / (n_samples as f64 - 1.0));

        // Compute explained variance ratio
        let total_variance = eigenvalues.sum();
        let explained_variance_ratio = &eigenvalues / total_variance;

        // Store results
        self.components = Some(components);
        self.mean = mean;
        self.std_dev = std_dev;
        self.explained_variance_ratio = Some(
            explained_variance_ratio
                .slice(s![..n_components])
                .to_owned(),
        );
        self.total_variance = Some(total_variance);
        self.eigenvalues = Some(eigenvalues.slice(s![..n_components]).to_owned());

        Ok(())
    }

    fn preprocess(
        &self,
        x: ArrayView2<f64>,
        mean: &Option<Array1<f64>>,
        std_dev: &Option<Array1<f64>>,
    ) -> Array2<f64> {
        let mut x_preprocessed = x.to_owned();

        // Center the data
        if let Some(m) = mean {
            x_preprocessed
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row| {
                    row -= m;
                });
        }

        // Scale the data
        if let Some(s) = std_dev {
            x_preprocessed
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row| {
                    row /= s;
                });
        }

        x_preprocessed
    }

    pub fn transform(&self, x: ArrayView2<f64>) -> anyhow::Result<Array2<f64>> {
        if let Some(components) = &self.components {
            let x_preprocessed = self.preprocess(x, &self.mean, &self.std_dev);

            // Ensure that we're using ArrayView2 for the dot product
            let x_preprocessed_view = x_preprocessed.view();
            let components_view = components.view();

            // Perform the matrix multiplication
            Ok(x_preprocessed_view.dot(&components_view.t()))
        } else {
            Err(anyhow::anyhow!("PCA has not been fitted yet"))
        }
    }

    pub fn fit_transform(&mut self, x: ArrayView2<f64>) -> anyhow::Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    // Getter methods for the computed values (unchanged)
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    pub fn total_variance(&self) -> Option<f64> {
        self.total_variance
    }

    pub fn eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }
}

// Example implementation of the SVDImplementation trait
pub struct LapackSVD;

impl SVDImplementation for LapackSVD {
    fn compute(&self, matrix: ArrayView2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        // This is where you'd implement the LAPACK SVD computation
        // For now, we'll just return dummy values
        let (m, n) = matrix.dim();
        (
            Array2::zeros((m, m)),
            Array1::zeros(m.min(n)),
            Array2::zeros((n, n)),
        )
    }
}

pub struct FearSVD;

impl SVDImplementation for FearSVD {
    fn compute(&self, matrix: ArrayView2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let svd = crate::svd::faer::SVD::new(&matrix);

        (svd.u().clone(), svd.s().clone(), svd.vt().clone())
    }
}
