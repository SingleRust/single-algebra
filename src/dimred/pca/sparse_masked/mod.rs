//! # Masked Sparse Principal Component Analysis
//!
//! This module implements PCA for sparse CSR matrices with feature masking capabilities.
//! Allows selective inclusion/exclusion of features during PCA computation, useful for
//! analyzing subsets of features or handling missing data patterns.

use crate::dimred::pca::SVDMethod;
use crate::sparse::MatrixSum;
use anyhow::anyhow;
use nalgebra::RealField;
use nalgebra_sparse::CsrMatrix;
use ndarray::{Array1, Array2};
use nshare::{IntoNalgebra, IntoNdarray2};
use rayon::iter::ParallelIterator;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};
use single_svdlib::lanczos::masked::MaskedCSRMatrix;
use single_svdlib::{lanczos, randomized, SvdFloat};
use single_utilities::traits::FloatOpsTS;
use std::collections::HashMap;
use std::time::Instant;

/// Builder for configuring and creating MaskedSparsePCA instances.
///
/// Provides a fluent interface for setting masked PCA parameters with sensible defaults.
/// The key difference from regular PCA is the ability to specify a feature mask that
/// determines which columns (features) are included in the analysis.
///
/// # Example Usage
/// ```ignore
/// let mask = vec![true, false, true, true]; // Include features 0, 2, 3; exclude feature 1
/// let pca = MaskedSparsePCABuilder::new()
///     .n_components(10)
///     .mask(mask)
///     .center(true)
///     .verbose(true)
///     .build();
/// ```
pub struct MaskedSparsePCABuilder<T>
where
    T: SvdFloat + 'static + RealField + FloatOpsTS,
{
    n_components: usize,
    alpha: T,
    tolerance: T,
    random_seed: Option<u32>,
    center: bool,
    verbose: bool,
    mask: Vec<bool>,
    svdmethod: SVDMethod,
}

impl<T> Default for MaskedSparsePCABuilder<T>
where
    T: SvdFloat + 'static + RealField + FloatOpsTS,
{
    fn default() -> Self {
        Self {
            n_components: 50,
            alpha: T::from(1.0).unwrap(),
            tolerance: T::from(1e-6).unwrap(),
            random_seed: Some(42),
            center: true,
            verbose: false,
            mask: Vec::new(),
            svdmethod: SVDMethod::default(),
        }
    }
}

impl<T> MaskedSparsePCABuilder<T>
where
    T: SvdFloat + 'static + RealField + FloatOpsTS,
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
    /// - `mask`: empty (must be set before building)
    /// - `svdmethod`: Lanczos
    pub fn new() -> Self {
        Self::default()
    }

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

    /// Sets the feature mask for selective analysis.
    ///
    /// The mask vector must have the same length as the number of features
    /// in the input matrix. Features with `true` values are included in the
    /// PCA analysis, while `false` values are excluded.
    ///
    /// # Parameters
    /// - `mask`: Boolean vector indicating which features to include
    pub fn mask(mut self, mask: Vec<bool>) -> Self {
        self.mask = mask;
        self
    }

    pub fn random_seed(mut self, seed: u32) -> Self {
        self.random_seed = Some(seed);
        self
    }

    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets the SVD method for computation.
    ///
    /// - `SVDMethod::Lanczos`: More accurate, better for smaller problems
    /// - `SVDMethod::Random`: Faster, especially beneficial for large masked matrices
    pub fn svd_method(mut self, method: SVDMethod) -> Self {
        self.svdmethod = method;
        self
    }

    /// Builds the final MaskedSparsePCA instance with the configured parameters.
    ///
    /// # Panics
    /// The mask vector must be set before calling build(), otherwise the
    /// resulting PCA instance may not work correctly.
    pub fn build(self) -> MaskedSparsePCA<T> {
        MaskedSparsePCA {
            n_components: self.n_components,
            alpha: self.alpha,
            tolerance: self.tolerance,
            random_seed: self.random_seed.unwrap_or(42),
            components_: None,
            explained_variance_: None,
            mean_: None,
            mask: self.mask,
            center: self.center,
            verbose: self.verbose,
            svd_method: self.svdmethod,
        }
    }
}

/// Principal Component Analysis for sparse CSR matrices with feature masking.
///
/// Extends standard sparse PCA by allowing selective inclusion of features through
/// a boolean mask. This is useful for:
/// - Analyzing feature subsets
/// - Handling missing data patterns
/// - Excluding irrelevant or noisy features
/// - Performing PCA on specific feature groups
///
/// The mask vector determines which columns (features) are included in the SVD
/// computation. Only masked features contribute to the principal components.
///
/// # Type Parameters
/// - `T`: Numeric type supporting SVD operations (typically `f32` or `f64`)
///
/// # Performance
/// Uses parallel processing for transformation operations to handle large datasets efficiently.
pub struct MaskedSparsePCA<T>
where
    T: SvdFloat + 'static + RealField + FloatOpsTS,
{
    n_components: usize,
    alpha: T,
    tolerance: T,
    random_seed: u32,
    components_: Option<Array2<T>>,
    explained_variance_: Option<Array1<T>>,
    mean_: Option<Array1<T>>,
    mask: Vec<bool>,
    center: bool,
    verbose: bool,
    svd_method: SVDMethod,
}

impl<T> MaskedSparsePCA<T>
where
    T: SvdFloat + 'static + RealField + FloatOpsTS,
{
    /// Creates a new MaskedSparsePCA instance with specified parameters.
    ///
    /// # Parameters
    /// - `n_components`: Number of principal components to compute
    /// - `alpha`: Regularization parameter (currently unused)
    /// - `tollerance`: Convergence tolerance for SVD algorithms
    /// - `random_seed`: Seed for reproducible randomized operations
    /// - `mask`: Boolean vector indicating which features to include
    /// - `center`: Whether to center the data (subtract column means)
    /// - `verbose`: Enable detailed progress output
    /// - `svd_method`: SVD algorithm to use (Lanczos or Randomized)
    ///
    /// # Note
    /// Consider using `MaskedSparsePCABuilder` for a more convenient API.
    pub fn new(
        n_components: usize,
        alpha: T,
        tollerance: Option<T>,
        random_seed: Option<u32>,
        mask: Vec<bool>,
        center: bool,
        verbose: bool,
        svd_method: SVDMethod,
    ) -> Self {
        Self {
            n_components,
            alpha,
            tolerance: tollerance.unwrap_or(T::from(1e-6).unwrap()),
            random_seed: random_seed.unwrap_or(42),
            components_: None,
            explained_variance_: None,
            mean_: None,
            mask,
            center,
            verbose,
            svd_method,
        }
    }

    /// Fits the masked PCA model to the provided sparse matrix.
    ///
    /// Computes principal components using only the features specified by the mask.
    /// The mask vector length must match the number of matrix columns.
    ///
    /// # Parameters
    /// - `x`: Input sparse CSR matrix (samples × features)
    ///
    /// # Returns
    /// - `Ok(&mut self)`: Success, model is fitted and ready for transformation
    /// - `Err`: Mask length mismatch, SVD computation failed, or other error
    ///
    /// # Performance Notes
    /// - Automatically chooses optimal SVD iterations based on problem size
    /// - Provides detailed timing information when verbose mode is enabled
    /// - Uses masked CSR matrix for efficient computation on feature subsets
    pub fn fit(&mut self, x: &CsrMatrix<T>) -> anyhow::Result<&mut Self> {
        let n_samples = x.nrows();
        let start = Instant::now();
        if x.ncols() != self.mask.len() {
            return Err(anyhow!(
                "The mask vector length and the number of features (columns) have to be the same!"
            ));
        }

        let mut n_features = 0usize;
        let mut cols_to_use: Vec<usize> = Vec::new();
        for (ind, &val) in self.mask.iter().enumerate() {
            if val {
                n_features += 1;
                cols_to_use.push(ind);
            }
        }

        let n_t_samples = T::from(n_samples).unwrap();

        if self.center {
            if self.verbose {
                println!("PCA | SparseMasked | Initializing centering...")
            }
            let col_sums: Vec<T> = x.sum_col()?;
            let mean = Array1::from(
                col_sums
                    .iter()
                    .map(|&sum| sum / n_t_samples)
                    .collect::<Vec<T>>(),
            );
            self.mean_ = Some(mean);
            if self.verbose {
                println!("PCA | SparseMasked | Computed centering statistics, took: {:?} of total running time", start.elapsed());
            }
        } else {
            self.mean_ = Some(Array1::zeros(x.ncols()));
        }

        let mut total_var = T::zero();
        if self.center {
            if self.verbose {
                println!("PCA | SparseMasked | Calculating total variance statistics....")
            }
            let col_sums: Vec<T> = x.sum_col()?;
            let col_sq_sums: Vec<T> = x.sum_col_squared()?;
            let n_minus_1 = n_t_samples - T::one();

            for &j in &cols_to_use {
                let mean = col_sums[j] / n_t_samples;
                let var = (col_sq_sums[j] - mean * col_sums[j]) / n_minus_1;
                total_var += var;
            }
            if self.verbose {
                println!("PCA | SparseMasked | Computed total variance statistics, took: {:?} of total running time", start.elapsed());
            }
        }

        let masked_matrix = MaskedCSRMatrix::new(x, self.mask.clone());

        let mut res = match self.svd_method {
            SVDMethod::Lanczos => {
                if self.verbose {
                    println!("PCA | SparseMasked | Computing Lanczos SVD....")
                }

                let optimal_iterations = (n_samples.max(n_features) * 2).max(100);
                lanczos::svd_las2(
                    &masked_matrix,
                    self.n_components,
                    optimal_iterations,
                    &[T::from(-1.0e-30).unwrap(), T::from(1.0e30).unwrap()],
                    T::from(10e-6).unwrap(),
                    self.random_seed,
                )
                .map_err(|e| anyhow!("SVD computation failed: {}", e))?
            }
            SVDMethod::Random {
                n_oversamples,
                n_power_iterations,
                normalizer,
            } => {
                if self.verbose {
                    println!("PCA | SparseMasked | Computing Randomized SVD....")
                }

                randomized::randomized_svd(
                    &masked_matrix,
                    self.n_components,
                    n_oversamples,
                    n_power_iterations,
                    normalizer,
                    self.center,
                    Some(self.random_seed as u64),
                    self.verbose,
                )
                .map_err(|e| anyhow!("Randomized SVD computation failed: {}", e))?
            }
        };

        if self.verbose {
            println!(
                "PCA | SparseMasked | Computed SVD, took: {:?} of total running time",
                start.elapsed()
            );
        }

        let mut u = res.u.into_nalgebra();
        let mut vt = res.vt.into_nalgebra();
        randomized::svd_flip(Some(&mut u), Some(&mut vt), false)?;
        res.u = u.into_ndarray2().into_owned();
        res.vt = vt.into_ndarray2().into_owned();

        self.components_ = Some(res.vt);

        let n_minus_1 = T::from(n_samples - 1).unwrap();
        let mut explained_variance = Array1::zeros(self.n_components);

        println!(
            "{:?}, components: {:?}, nfeatures: {:?}",
            res.s.dim(),
            self.n_components,
            n_features
        );
        for i in 0..self.n_components {
            explained_variance[i] = num_traits::Float::powi(res.s[i], 2) / n_minus_1;
        }
        self.explained_variance_ = Some(explained_variance.clone());

        if !self.center {
            total_var = T::zero();
            for i in 0..res.s.len() {
                total_var += num_traits::Float::powi(res.s[i], 2) / n_minus_1;
            }
        }

        let min_dim = n_samples.min(n_features);
        if self.n_components < min_dim {
            let noise_var = (total_var - explained_variance.sum())
                / T::from(min_dim - self.n_components).unwrap();

            if self.verbose {
                println!("Estimated noise variance: {}", noise_var);
            }
        }

        if self.verbose {
            println!("PCA completed successfully:");
            println!(
                "  Input shape: {} samples × {} features (using {} features with mask)",
                n_samples,
                x.ncols(),
                n_features
            );
            println!("  Reduced to: {} components", self.n_components);
            println!(
                "  Total variance explained: {:.2}%",
                (explained_variance.sum() / total_var * T::from(100.0).unwrap())
                    .to_f64()
                    .unwrap_or(0.0)
            );
        }

        Ok(self)
    }

    /// Transforms data to the reduced dimensional space using masked features.
    ///
    /// Projects the input matrix onto the principal components learned during fitting.
    /// Only features specified in the mask contribute to the transformation.
    /// Uses parallel processing for efficient computation on large datasets.
    ///
    /// # Parameters
    /// - `x`: Input sparse CSR matrix to transform (samples × features)
    ///
    /// # Returns
    /// - `Ok(Array2<T>)`: Transformed data (samples × n_components)
    /// - `Err`: Model not fitted, mask length mismatch, or transformation failed
    ///
    /// # Performance
    /// - Automatically adapts chunk size based on dataset size
    /// - Uses SIMD-friendly unrolled loops for small component counts
    /// - Employs parallel iteration with Rayon for large datasets
    pub fn transform(&self, x: &CsrMatrix<T>) -> anyhow::Result<Array2<T>> {
        let n_samples = x.nrows();
        if x.ncols() != self.mask.len() {
            return Err(anyhow!(
                "The mask vector length and the number of features (columns) have to be the same!"
            ));
        }

        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| anyhow!("Must be fitted before transform!"))?;
        let mean = self
            .mean_
            .as_ref()
            .ok_or_else(|| anyhow!("Must be fitted before transform!"))?;

        let cols_to_use: Vec<usize> = self
            .mask
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val { Some(i) } else { None })
            .collect();

        let col_to_masked_idx: HashMap<usize, usize> = cols_to_use
            .iter()
            .enumerate()
            .map(|(idx, &col)| (col, idx))
            .collect();

        let chunk_size = if n_samples > 10000 {
            64
        } else if n_samples > 1000 {
            128
        } else {
            256
        };

        let mut transformed = Array2::zeros((n_samples, self.n_components));

        let results: Vec<(usize, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|&row_idx| {
                        let row = x.row(row_idx);
                        let mut scores = vec![T::zero(); self.n_components];

                        for (&col_idx, &val) in row.col_indices().iter().zip(row.values().iter()) {
                            if let Some(&masked_idx) = col_to_masked_idx.get(&col_idx) {
                                let effective_val = if self.center {
                                    val - mean[col_idx]
                                } else {
                                    val
                                };

                                if self.n_components <= 8 {
                                    if self.n_components > 0 {
                                        scores[0] += effective_val * components[[0, masked_idx]];
                                    }
                                    if self.n_components > 1 {
                                        scores[1] += effective_val * components[[1, masked_idx]];
                                    }
                                    if self.n_components > 2 {
                                        scores[2] += effective_val * components[[2, masked_idx]];
                                    }
                                    if self.n_components > 3 {
                                        scores[3] += effective_val * components[[3, masked_idx]];
                                    }
                                    if self.n_components > 4 {
                                        scores[4] += effective_val * components[[4, masked_idx]];
                                    }
                                    if self.n_components > 5 {
                                        scores[5] += effective_val * components[[5, masked_idx]];
                                    }
                                    if self.n_components > 6 {
                                        scores[6] += effective_val * components[[6, masked_idx]];
                                    }
                                    if self.n_components > 7 {
                                        scores[7] += effective_val * components[[7, masked_idx]];
                                    }
                                } else {
                                    for k in (0..self.n_components).step_by(4) {
                                        let end = (k + 4).min(self.n_components);
                                        for kk in k..end {
                                            scores[kk] +=
                                                effective_val * components[[kk, masked_idx]];
                                        }
                                    }
                                }
                            }
                        }

                        (row_idx, scores)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (row_idx, scores) in results {
            for k in 0..self.n_components {
                transformed[[row_idx, k]] = scores[k];
            }
        }

        Ok(transformed)
    }

    /// Calculates feature importances for the masked features.
    ///
    /// Returns the squared loadings of each masked feature for each component,
    /// indicating how much each included feature contributes to each principal component.
    /// The returned matrix has dimensions (n_components × n_masked_features).
    ///
    /// # Returns
    /// - `Ok(Array2<T>)`: Feature importances for masked features only
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
    /// variance explained by all computed components from the masked features.
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

    /// Calculates cumulative explained variance ratios for masked features.
    ///
    /// Returns the cumulative sum of explained variance ratios from the masked
    /// features, useful for determining how many components are needed to explain
    /// a desired percentage of the variance in the selected feature subset.
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
    /// Equivalent to calling `fit(x)` followed by `transform(x)` on the masked features.
    /// Useful for one-shot dimensionality reduction workflows.
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
