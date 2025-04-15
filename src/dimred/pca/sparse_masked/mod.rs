use crate::dimred::pca::SVDMethod;
use crate::sparse::MatrixSum;
use crate::NumericOps;
use anyhow::anyhow;
use nalgebra_sparse::CsrMatrix;
use nalgebra::RealField;
use ndarray::{s, Array1, Array2};
use single_svdlib::randomized::randomized_svd;
use single_svdlib::{lanczos::masked::MaskedCSRMatrix, lanczos::svd_las2};

pub struct MaskedSparsePCABuilder<T>
where
    T: single_svdlib::SvdFloat,
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
    T: single_svdlib::SvdFloat,
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
    T: single_svdlib::SvdFloat,
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
    T: single_svdlib::SvdFloat,
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
    T: single_svdlib::SvdFloat + NumericOps + 'static + RealField,
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

    pub fn initialize_components(&self, x: &CsrMatrix<T>) -> anyhow::Result<Array2<T>> {
        let n_samples = x.nrows();
        if x.ncols() != self.mask.len() {
            return Err(anyhow!(
                "The mask vector length and the number of features (columns) have to be the same!"
            ));
        }
        let mut n_features = 0usize;
        for f in self.mask.clone() {
            if f {
                n_features += 1;
            }
        }

        let masked_matrix = MaskedCSRMatrix::new(x, self.mask.clone());
        match self.svd_method {
            SVDMethod::Lanczos => {
                let optimal_iterations = n_samples.max(n_features);
                let svd_masked = svd_las2(
                    &masked_matrix,
                    self.n_components,
                    optimal_iterations,
                    &[T::from(-1.0e-30).unwrap(), T::from(1.0e30).unwrap()],
                    T::from(10e-6).unwrap(),
                    self.random_seed,
                )
                .map_err(|e| anyhow!("SVD computation failed: {}", e))?;

                let components = svd_masked.vt.slice(s![..self.n_components, ..]).to_owned();

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

                Ok(components)
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
                    &masked_matrix,
                    self.n_components,
                    n_oversamples,
                    n_power_iterations,
                    normalizer,
                    Some(self.random_seed as u64),
                )
                .map_err(|e| anyhow!("Randomized SVD computation failed: {}", e))?;

                let components = svd_result.vt.slice(s![..self.n_components, ..]).to_owned();

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

                Ok(components)
            }
        }
    }

    pub fn fit(&mut self, x: &CsrMatrix<T>, max_iter: Option<usize>) -> anyhow::Result<&mut Self> {
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
            self.mean_ = Some(Array1::zeros(n_features));
        }

        let mut components = self.initialize_components(x)?;
        let max_iter = max_iter.unwrap_or(1000);

        let mut converged = false;
        let mut iter = 0;

        let mut u = Array2::zeros((n_samples, self.n_components));
        let mut v = Array2::zeros((n_features, self.n_components));

        let mean = self.mean_.as_ref().unwrap();

        while !converged && iter < max_iter {
            let prev_components = components.clone();

            for (row_idx, row) in x.row_iter().enumerate() {
                for k in 0..self.n_components {
                    let mut score = T::zero();
                    for (idx, &col_idx) in cols_to_use.iter().enumerate() {
                        let val = row.get_entry(col_idx).unwrap().into_value();
                        let effective_val = if self.center {
                            val - mean[col_idx]
                        } else {
                            val
                        };
                        score += effective_val * components[[k, idx]];
                    }
                    u[[row_idx, k]] = score;
                }
            }

            for j in 0..n_features {
                let col_idx = cols_to_use[j];
                for k in 0..self.n_components {
                    let mut loading = T::zero();

                    for (row_idx, row) in x.row_iter().enumerate() {
                        if let Some(val) = row.get_entry(col_idx) {
                            let effective_val = if self.center {
                                val.into_value() - mean[col_idx]
                            } else {
                                val.into_value()
                            };

                            loading += effective_val * u[[row_idx, k]];
                        }
                    }
                    let sign = num_traits::Float::signum(loading);
                    let magnitude = num_traits::Float::abs(loading);
                    v[[j, k]] = sign * num_traits::Float::max(T::zero(), (magnitude - self.alpha));
                    //T::zero().max(magnitude - self.alpha);
                }
            }

            for k in 0..self.n_components {
                let norm = num_traits::Float::sqrt(v.column(k).iter().map(|&x| x * x).sum::<T>());
                if norm > T::zero() {
                    for j in 0..n_features {
                        v[[j, k]] = v[[j, k]] / norm;
                        components[[k, j]] = v[[j, k]];
                    }
                }
            }

            let diff = num_traits::Float::sqrt(
                (&components - &prev_components)
                    .iter()
                    .map(|&x| x * x)
                    .sum::<T>(),
            );

            converged = diff < self.tolerance;
            iter += 1;

            if self.verbose {
                println!("Fitting completed after {} iterations", iter);
                if !converged {
                    println!("Warning: Maximum iterations reached before convergence");
                }
            }
        }

        self.components_ = Some(components);

        let mut explained_variance = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let variance = u.column(k).dot(&u.column(k));
            explained_variance[k] = variance / (n_t_samples - T::one());
        }
        self.explained_variance_ = Some(explained_variance);

        Ok(self)
    }

    pub fn transform(&self, x: &CsrMatrix<T>) -> anyhow::Result<Array2<T>> {
        let n_samples = x.nrows();
        if x.ncols() != self.mask.len() {
            return Err(anyhow!(
                "The mask vector length and the number of features (columns) have to be the same!"
            ));
        }

        let mut cols_to_use: Vec<usize> = Vec::new();
        for (ind, &val) in self.mask.iter().enumerate() {
            if val {
                cols_to_use.push(ind);
            }
        }

        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| anyhow!("Must be fitted before transform!"))?;
        let mean = self
            .mean_
            .as_ref()
            .ok_or_else(|| anyhow!("Must be fitted before transform!"))?;

        let mut transformed = Array2::zeros((n_samples, self.n_components));

        for (row_idx, row) in x.row_iter().enumerate() {
            for k in 0..self.n_components {
                let mut score = T::zero();
                for (cidx, &col_idx) in cols_to_use.iter().enumerate() {
                    let val = row
                        .get_entry(col_idx)
                        .map_or(T::zero(), |entry| entry.into_value());
                    let effective_val = if self.center {
                        val - mean[col_idx]
                    } else {
                        val
                    };
                    score += effective_val * components[[k, cidx]];
                }
                transformed[[row_idx, k]] = score;
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
        x: &CsrMatrix<T>,
        max_iter: Option<usize>,
    ) -> anyhow::Result<Array2<T>> {
        self.fit(x, max_iter)?;
        self.transform(x)
    }
}
