//! # Sparse Principal Component Analysis
//!
//! This module implements PCA for sparse CSR matrices with optional data centering.
//! Supports both Lanczos and randomized SVD algorithms for flexible performance/accuracy trade-offs.

use crate::dimred::pca::SVDMethod;
use crate::sparse::MatrixSum;
use anyhow::anyhow;
use nalgebra::RealField;
use nalgebra_sparse::CsrMatrix;
use ndarray::{Array1, Array2};
use nshare::{IntoNalgebra, IntoNdarray2};
use single_svdlib::lanczos::svd_las2;
use single_svdlib::randomized::randomized_svd;
use single_svdlib::SvdFloat;
use single_utilities::traits::{FloatOpsTS, NumericOpsTS};
use std::ops::Div;

/// Principal Component Analysis for sparse CSR matrices.
///
/// Performs dimensionality reduction by finding the directions of maximum variance
/// in the data. Efficiently handles sparse matrices while supporting optional
/// data centering and multiple SVD algorithms.
///
/// # Type Parameters
/// - `T`: Numeric type supporting SVD operations (typically `f32` or `f64`)
///
/// # Features
/// - Sparse matrix support via CSR format
/// - Optional data centering
/// - Lanczos or randomized SVD algorithms
/// - Explained variance analysis
/// - Feature importance calculation
pub struct SparsePCA<T>
where
    T: SvdFloat + FloatOpsTS + 'static + RealField + ndarray::ScalarOperand,
{
    n_components: usize,
    alpha: T,
    tolerance: T,
    random_seed: u32,
    components_: Option<Array2<T>>,
    explained_variance_: Option<Array1<T>>,
    mean_: Option<Array1<T>>,
    center: bool,
    verbose: bool,
    svdmethod: SVDMethod,
}

impl<T> SparsePCA<T>
where
    T: SvdFloat + FloatOpsTS + 'static + RealField + ndarray::ScalarOperand,
{
    /// Creates a new SparsePCA instance with specified parameters.
    ///
    /// # Parameters
    /// - `n_components`: Number of principal components to compute
    /// - `alpha`: Regularization parameter (currently unused)
    /// - `tollerance`: Convergence tolerance for SVD algorithms
    /// - `random_seed`: Seed for reproducible randomized operations
    /// - `center`: Whether to center the data (subtract column means)
    /// - `verbose`: Enable detailed progress output
    /// - `svdmethod`: SVD algorithm to use (Lanczos or Randomized)
    pub fn new(
        n_components: usize,
        alpha: T,
        tollerance: Option<T>,
        random_seed: Option<u32>,
        center: bool,
        verbose: bool,
        svdmethod: SVDMethod,
    ) -> Self {
        Self {
            n_components,
            alpha,
            tolerance: tollerance.unwrap_or(T::from(1e-6).unwrap()),
            random_seed: random_seed.unwrap_or(42),
            components_: None,
            explained_variance_: None,
            mean_: None,
            center,
            verbose,
            svdmethod,
        }
    }

    /// Fits the PCA model to the provided sparse matrix.
    ///
    /// Computes the principal components by performing SVD on the input matrix.
    /// Optionally centers the data and calculates explained variance statistics.
    ///
    /// # Parameters
    /// - `x`: Input sparse CSR matrix (samples × features)
    ///
    /// # Returns
    /// - `Ok(&mut self)`: Success, model is fitted and ready for transformation
    /// - `Err`: SVD computation failed or other error occurred
    ///
    /// # Side Effects
    /// - Stores principal components in `components_`
    /// - Stores explained variance in `explained_variance_`
    /// - Stores column means in `mean_` (if centering enabled)
    pub fn fit(&mut self, x: &CsrMatrix<T>) -> anyhow::Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.center {
            let col_sums: Vec<T> = x.sum_col()?;
            let n_t_samples = T::from(n_samples).unwrap();
            self.mean_ = Some(Array1::from(
                col_sums
                    .iter()
                    .map(|&sum| sum / n_t_samples)
                    .collect::<Vec<T>>(),
            ));
        } else {
            self.mean_ = Some(Array1::zeros(n_samples));
        }

        let mut total_var = T::zero();
        if self.center {
            let col_sums: Vec<T> = x.sum_col()?;
            let col_sq_sums: Vec<T> = x.sum_col_squared()?;
            let n_t_samples = T::from(n_samples).unwrap();
            let n_minus_1 = n_t_samples - T::one();

            for j in 0..n_features {
                let mean = col_sums[j] / n_t_samples;
                let var = (col_sq_sums[j] - mean * col_sums[j]) / n_minus_1;
                total_var += var;
            }
        }

        let mut res = match self.svdmethod {
            SVDMethod::Lanczos => {
                let optimal_iterations = n_samples.max(n_features);
                let svd_result = svd_las2(
                    x,
                    self.n_components,
                    optimal_iterations,
                    &[T::from(-1.0e-30).unwrap(), T::from(1.0e30).unwrap()],
                    T::from(10e-6).unwrap(),
                    self.random_seed,
                )
                .map_err(|e| anyhow!("SVD computation failed: {}", e))?;

                if self.verbose {
                    println!("SVD using Lanczos algorithm:");
                    println!(
                        "  Input shape: {} samples × {} features",
                        n_samples, n_features
                    );
                    println!("  Reduced to: {} components", self.n_components);
                    println!(
                        "  Compression ratio: {:.2}%",
                        (self.n_components as f64 / n_features as f64) * 100.0
                    );
                }

                svd_result
            }
            SVDMethod::Random {
                n_oversamples,
                n_power_iterations,
                normalizer,
            } => {
                if self.verbose {
                    println!("Computing randomized SVD...");
                }

                let svd_result = randomized_svd(
                    x,
                    self.n_components,
                    n_oversamples,
                    n_power_iterations,
                    normalizer,
                    self.center,
                    Some(self.random_seed as u64),
                    self.verbose,
                )
                .map_err(|e| anyhow!("Randomized SVD computation failed: {}", e))?;

                if self.verbose {
                    println!("SVD using randomized algorithm:");
                    println!(
                        "  Input shape: {} samples × {} features",
                        n_samples, n_features
                    );
                    println!("  Reduced to: {} components", self.n_components);
                    println!(
                        "  Compression ratio: {:.2}%",
                        (self.n_components as f64 / n_features as f64) * 100.0
                    );
                    println!("  Oversampling: {}", n_oversamples);
                    println!("  Power iterations: {}", n_power_iterations);
                }

                svd_result
            }
        };

        let mut u = res.u.into_nalgebra();
        let mut vt = res.vt.into_nalgebra();
        single_svdlib::randomized::svd_flip(Some(&mut u), Some(&mut vt), false)?;

        res.u = u.into_ndarray2().into_owned();
        res.vt = vt.into_ndarray2().into_owned();

        self.components_ = Some(res.vt);

        let mut explained_variance = res.s.powi(2).div(T::from(n_samples - 1).unwrap());
        let n_minus_1 = T::from(n_samples - 1).unwrap();

        for i in 0..self.n_components {
            explained_variance[i] = num_traits::Float::powi(res.s[i], 2) / n_minus_1;
        }
        self.explained_variance_ = Some(explained_variance);

        if !self.center {
            total_var = T::zero(); // Just to make sure
            for i in 0..res.s.len() {
                total_var += num_traits::Float::powi(res.s[i], 2) / n_minus_1;
            }
        }

        let min_dim = n_samples.min(n_features);
        if self.verbose && self.n_components < min_dim {
            let mut exp_var_sum = T::zero();
            match &self.explained_variance_ {
                None => {}
                Some(v) => {
                    for &i in v {
                        exp_var_sum += i;
                    }
                }
            };
            let noise_var =
                (total_var - exp_var_sum) / T::from(min_dim - self.n_components).unwrap();
            println!("Estimated noise variance: {}", noise_var);
        }

        Ok(self)
    }

    /// Transforms data to the reduced dimensional space.
    ///
    /// Projects the input matrix onto the principal components learned during fitting.
    /// The model must be fitted before calling this method.
    ///
    /// # Parameters
    /// - `x`: Input sparse CSR matrix to transform (samples × features)
    ///
    /// # Returns
    /// - `Ok(Array2<T>)`: Transformed data (samples × n_components)
    /// - `Err`: Model not fitted or transformation failed
    pub fn transform(&self, x: &CsrMatrix<T>) -> anyhow::Result<Array2<T>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| anyhow!("Must be fitted before transform!"))?;
        let mean = self
            .mean_
            .as_ref()
            .ok_or_else(|| anyhow!("Must be fitted before transform!"))?;

        let n_samples = x.nrows();
        let mut transformed = Array2::zeros((n_samples, self.n_components));

        for (row_idx, row) in x.row_iter().enumerate() {
            for k in 0..self.n_components {
                let mut score = T::zero();
                for &col_idx in x.col_indices() {
                    let val = row.get_entry(col_idx).unwrap().into_value();
                    let effective_val = if self.center {
                        val - mean[col_idx]
                    } else {
                        val
                    };
                    score += effective_val * components[[k, col_idx]];
                }
                transformed[[row_idx, k]] = score;
            }
        }

        Ok(transformed)
    }

    /// Calculates feature importances based on component loadings.
    ///
    /// Returns the squared loadings of each feature for each component,
    /// indicating how much each feature contributes to each principal component.
    ///
    /// # Returns
    /// - `Ok(Array2<T>)`: Feature importances (n_components × n_features)
    /// - `Err`: Model not fitted
    pub fn feature_importances(&self) -> anyhow::Result<Array2<T>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| anyhow!("Model must be fitted first!"))?;
        let importances = components.mapv(|x| x * x);
        Ok(importances)
    }

    /// Calculates the proportion of variance explained by each component.
    ///
    /// Returns the ratio of each component's explained variance to the total
    /// variance explained by all computed components.
    ///
    /// # Returns
    /// - `Ok(Array1<T>)`: Variance ratios for each component (sum ≤ 1.0)
    /// - `Err`: Model not fitted
    pub fn explained_variance_ratio(&self) -> anyhow::Result<Array1<T>> {
        let explained_variance = self
            .explained_variance_
            .as_ref()
            .ok_or_else(|| anyhow!("Model must be fitted first!"))?;

        let total_variance = explained_variance.sum();
        let ratios = explained_variance.mapv(|v| v / total_variance);

        Ok(ratios)
    }

    /// Calculates cumulative explained variance ratios.
    ///
    /// Returns the cumulative sum of explained variance ratios, useful for
    /// determining how many components are needed to explain a desired
    /// percentage of the total variance.
    ///
    /// # Returns
    /// - `Ok(Array1<T>)`: Cumulative variance ratios (monotonically increasing)
    /// - `Err`: Model not fitted
    pub fn cumulative_explained_variance_ratio(&self) -> anyhow::Result<Array1<T>> {
        let ratios = self.explained_variance_ratio()?;
        let mut cumulative = Array1::zeros(ratios.len());
        let mut sum = T::zero();
        for (i, &ratio) in ratios.iter().enumerate() {
            sum += ratio;
            cumulative[i] = sum;
        }

        Ok(cumulative)
    }

    /// Convenience method that fits the model and transforms the data in one step.
    ///
    /// Equivalent to calling `fit(x)` followed by `transform(x)`.
    ///
    /// # Parameters
    /// - `x`: Input sparse CSR matrix (samples × features)
    ///
    /// # Returns
    /// - `Ok(Array2<T>)`: Transformed data (samples × n_components)
    /// - `Err`: Fitting or transformation failed
    pub fn fit_transform(&mut self, x: &CsrMatrix<T>) -> anyhow::Result<Array2<T>> {
        self.fit(x)?;
        self.transform(x)
    }
}

/// Builder for configuring and creating SparsePCA instances.
///
/// Provides a fluent interface for setting PCA parameters with sensible defaults.
/// Use `SparsePCABuilder::new()` to start building and `build()` to create the final instance.
///
/// # Example Usage
/// ```ignore
/// let pca = SparsePCABuilder::new()
///     .n_components(50)
///     .center(true)
///     .verbose(true)
///     .svd_method(SVDMethod::Lanczos)
///     .build();
/// ```
pub struct SparsePCABuilder<T>
where
    T: SvdFloat + FloatOpsTS + 'static + RealField + ndarray::ScalarOperand,
{
    n_components: usize,
    alpha: T,
    tolerance: T,
    random_seed: Option<u32>,
    center: bool,
    verbose: bool,
    svdmethod: SVDMethod,
}

impl<T> Default for SparsePCABuilder<T>
where
    T: SvdFloat + FloatOpsTS + 'static + RealField + ndarray::ScalarOperand,
{
    fn default() -> Self {
        Self {
            n_components: 50,
            alpha: T::from(1.0).unwrap(),
            tolerance: T::from(1e-6).unwrap(),
            random_seed: Some(42),
            center: true,
            verbose: false,
            svdmethod: SVDMethod::default(),
        }
    }
}

impl<T> SparsePCABuilder<T>
where
    T: SvdFloat + FloatOpsTS + 'static + RealField + ndarray::ScalarOperand,
{
    /// Creates a new builder with default parameters.
    ///
    /// Default values:
    /// - `n_components`: 50
    /// - `alpha`: 1.0
    /// - `tolerance`: 1e-6
    /// - `random_seed`: 42
    /// - `center`: true
    /// - `verbose`: false
    /// - `svdmethod`: Lanczos
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of principal components to compute.
    ///
    /// Should be less than min(n_samples, n_features) for meaningful results.
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn random_seed(mut self, seed: u32) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Sets whether to center the data before PCA.
    ///
    /// When `true`, subtracts column means from the data. Generally recommended
    /// unless the data is already centered or centering is not desired.
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets the SVD method to use for computation.
    ///
    /// - `SVDMethod::Lanczos`: More accurate, deterministic
    /// - `SVDMethod::Random`: Faster, especially for large matrices
    pub fn svd_method(mut self, svdmethod: SVDMethod) -> Self {
        self.svdmethod = svdmethod;
        self
    }

    /// Builds the final SparsePCA instance with the configured parameters.
    pub fn build(self) -> SparsePCA<T> {
        SparsePCA {
            n_components: self.n_components,
            alpha: self.alpha,
            tolerance: self.tolerance,
            random_seed: self.random_seed.unwrap_or(42),
            components_: None,
            explained_variance_: None,
            mean_: None,
            center: self.center,
            verbose: self.verbose,
            svdmethod: self.svdmethod,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::dimred::pca::{SVDMethod, SparsePCABuilder};
    use nalgebra_sparse::CsrMatrix;
    use rayon::ThreadPoolBuilder;
    use single_svdlib::randomized::PowerIterationNormalizer;

    fn create_sparse_matrix(
        rows: usize,
        cols: usize,
        density: f64,
    ) -> nalgebra_sparse::coo::CooMatrix<f64> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use std::collections::HashSet;

        let mut coo = nalgebra_sparse::coo::CooMatrix::new(rows, cols);

        let mut rng = StdRng::seed_from_u64(42);

        let nnz = (rows as f64 * cols as f64 * density).round() as usize;

        let nnz = nnz.max(1);

        let mut positions = HashSet::new();

        while positions.len() < nnz {
            let i = rng.random_range(0..rows);
            let j = rng.random_range(0..cols);

            if positions.insert((i, j)) {
                let val = loop {
                    let v: f64 = rng.random_range(-10.0..10.0);
                    if v.abs() > 1e-10 {
                        // Ensure it's not too close to zero
                        break v;
                    }
                };

                coo.push(i, j, val);
            }
        }

        // Verify the density is as expected
        let actual_density = coo.nnz() as f64 / (rows as f64 * cols as f64);
        println!("Created sparse matrix: {} x {}", rows, cols);
        println!("  - Requested density: {:.6}", density);
        println!("  - Actual density: {:.6}", actual_density);
        println!("  - Sparsity: {:.4}%", (1.0 - actual_density) * 100.0);
        println!("  - Non-zeros: {}", coo.nnz());

        coo
    }

    #[test]
    fn test_random_matrix_sparse_svd_comp_random() {
        let random_matrix = create_sparse_matrix(10000000, 2500, 0.01);
        let random_matrix = CsrMatrix::from(&random_matrix);

        let mut sparse_pca = SparsePCABuilder::<f64>::new()
            .random_seed(42)
            .svd_method(SVDMethod::Random {
                n_oversamples: 10,
                n_power_iterations: 7,
                normalizer: PowerIterationNormalizer::QR,
            })
            .n_components(50)
            .verbose(true)
            .center(true)
            .tolerance(1e-4)
            .alpha(1.5)
            .build();

        let thread_pool = ThreadPoolBuilder::new().num_threads(64).build().unwrap();
        let res_fit = thread_pool.install(|| sparse_pca.fit(&random_matrix));

        assert!(res_fit.is_ok());
    }
}
