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
use std::collections::HashMap;
use single_utilities::traits::{FloatOpsTS};

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

    pub fn svd_method(mut self, method: SVDMethod) -> Self {
        self.svdmethod = method;
        self
    }

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

    pub fn fit(&mut self, x: &CsrMatrix<T>) -> anyhow::Result<&mut Self> {
        let n_samples = x.nrows();
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
            let col_sums: Vec<T> = x.sum_col()?;
            let mean = Array1::from(
                col_sums
                    .iter()
                    .map(|&sum| sum / n_t_samples)
                    .collect::<Vec<T>>(),
            );
            self.mean_ = Some(mean);
        } else {
            self.mean_ = Some(Array1::zeros(x.ncols()));
        }

        let mut total_var = T::zero();
        if self.center {
            let col_sums: Vec<T> = x.sum_col()?;
            let col_sq_sums: Vec<T> = x.sum_col_squared()?;
            let n_minus_1 = n_t_samples - T::one();

            for &j in &cols_to_use {
                let mean = col_sums[j] / n_t_samples;
                let var = (col_sq_sums[j] - mean * col_sums[j]) / n_minus_1;
                total_var = total_var + var;
            }
        }

        let masked_matrix = MaskedCSRMatrix::new(x, self.mask.clone());

        let mut res = match self.svd_method {
            SVDMethod::Lanczos => {
                if self.verbose {
                    println!("Computing SVD using Lanczos algorithm...");
                }

                let optimal_iterations = n_samples.max(n_features);
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
                    println!("Computing randomized SVD...");
                }

                randomized::randomized_svd(
                    &masked_matrix,
                    self.n_components,
                    n_oversamples,
                    n_power_iterations,
                    normalizer,
                    self.center,
                    Some(self.random_seed as u64),
                )
                .map_err(|e| anyhow!("Randomized SVD computation failed: {}", e))?
            }
        };

        let mut u = res.u.into_nalgebra();
        let mut vt = res.vt.into_nalgebra();
        randomized::svd_flip(Some(&mut u), Some(&mut vt), false)?;
        res.u = u.into_ndarray2().into_owned();
        res.vt = vt.into_ndarray2().into_owned();

        self.components_ = Some(res.vt);

        let n_minus_1 = T::from(n_samples - 1).unwrap();
        let mut explained_variance = Array1::zeros(self.n_components);

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

    pub fn feature_importances(&self) -> anyhow::Result<Array2<T>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| anyhow!("Model must be fitted first!"))?;
        let importances = components.mapv(|x| x * x);
        Ok(importances)
    }

    pub fn explained_variance_ratio(&self) -> anyhow::Result<Array1<T>> {
        let explained_variance = self
            .explained_variance_
            .as_ref()
            .ok_or_else(|| anyhow!("Model must be fitted first!"))?;
        let total_variance = explained_variance.sum();
        let ratios = explained_variance.mapv(|v| v / total_variance);
        Ok(ratios)
    }

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

    pub fn fit_transform(
        &mut self,
        x: &CsrMatrix<T>
    ) -> anyhow::Result<Array2<T>> {
        self.fit(x)?;
        self.transform(x)
    }
}
