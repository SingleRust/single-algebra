use anyhow::anyhow;
use nalgebra_sparse::CsrMatrix;
use ndarray::{s, Array1, Array2};
use single_svdlib::svdLAS2;

use crate::sparse::MatrixSum;

/// This method only allows f64 as a datatype for the input matrix.
pub struct SparsePCA {
    n_components: usize,
    alpha: f64,
    tollerance: f64,

    random_seed: u32,

    components_: Option<Array2<f64>>,
    explained_variance_: Option<Array1<f64>>,
    mean_: Option<Array1<f64>>,
    center: bool,
    verbose: bool,
}

impl SparsePCA {
    pub fn new(
        n_components: usize,
        alpha: f64,
        tollerance: Option<f64>,
        random_seed: Option<u32>,
        center: bool,
        verbose: bool,
    ) -> Self {
        SparsePCA {
            n_components,
            alpha,
            tollerance: tollerance.unwrap_or(1e-6),
            random_seed: random_seed.unwrap_or(42),
            components_: None,
            explained_variance_: None,
            mean_: None,
            center,
            verbose,
        }
    }

    fn initialize_components(&self, x: &CsrMatrix<f64>) -> anyhow::Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.n_components >= n_features {
            return Err(anyhow!(
                "Number of components ({}) must be less than number of features ({})!",
                self.n_components,
                n_features
            ));
        }

        let optimal_iterations = n_samples.max(n_features);
        let svd = svdLAS2(
            x,
            self.n_components,
            optimal_iterations,
            &[-1.0e-30, 1.0e-30],
            1.0e-6,
            self.random_seed,
        )
        .map_err(|e| anyhow::anyhow!("SVD computation failed: {}", e))?;

        // Changed this line to use n_features instead of n_samples
        let components = svd.vt.slice(s![..self.n_components, ..]).to_owned();

        if self.verbose {
            println!("Dimensionality reduction summary:");
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

    pub fn fit(&mut self, x: &CsrMatrix<f64>) -> anyhow::Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.center {
            let col_sums: Vec<f64> = x.sum_col()?;
            let mean = Array1::from(
                col_sums
                    .iter()
                    .map(|&sum| sum / n_samples as f64)
                    .collect::<Vec<f64>>(),
            );
            self.mean_ = Some(mean);
        } else {
            self.mean_ = Some(Array1::zeros(n_features));
        }

        let mut components = self.initialize_components(x)?;

        let max_iter = 1000;
        let mut converged = false;
        let mut iter = 0;

        let mut u = Array2::zeros((n_samples, self.n_components));
        let mut v = Array2::zeros((n_features, self.n_components));

        let mean = self.mean_.as_ref().unwrap();

        while !converged && iter < max_iter {
            let prev_components = components.clone();

            for (row_idx, row) in x.row_iter().enumerate() {
                for k in 0..self.n_components {
                    let mut score = 0.0;
                    for &col_idx in x.col_indices() {
                        let val = row.get_entry(col_idx).unwrap().into_value();
                        let effective_val = if self.center {
                            val - mean[col_idx]
                        } else {
                            val
                        };
                        score += effective_val * components[[k, col_idx]];
                    }
                    u[[row_idx, k]] = score;
                }
            }

            for j in 0..n_features {
                for k in 0..self.n_components {
                    let mut loading = 0.0;

                    for (row_idx, row) in x.row_iter().enumerate() {
                        if let Some(val) = row.get_entry(j) {
                            let effective_val = if self.center {
                                val.into_value() - mean[j]
                            } else {
                                val.into_value()
                            };
                            loading += effective_val * u[[row_idx, k]];
                        }
                    }

                    let sign = loading.signum();
                    let magnitude = loading.abs();
                    v[[j, k]] = sign * (0.0f64).max(magnitude - self.alpha);
                }
            }

            for k in 0..self.n_components {
                let norm: f64 = v.column(k).iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for j in 0..n_features {
                        v[[j, k]] /= norm;
                        components[[k, j]] = v[[j, k]];
                    }
                }
            }

            let diff = (&components - &prev_components)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();

            converged = diff < self.tollerance;
            iter += 1;

            if self.verbose && iter % 100 == 0 {
                println!("Iteration {}: convergence criterion = {}", iter, diff);
            }
        }

        if self.verbose {
            println!("Fitting completed after {} iterations", iter);
            if !converged {
                println!("Warning: Maximum iterations reached before convergence");
            }
        }

        self.components_ = Some(components);

        let mut explained_variance = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            let variance = u.column(k).dot(&u.column(k));
            explained_variance[k] = variance / (n_samples as f64 - 1.0);
        }
        self.explained_variance_ = Some(explained_variance);

        Ok(self)
    }

    pub fn transform(&self, x: &CsrMatrix<f64>) -> anyhow::Result<Array2<f64>> {
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
                let mut score = 0.0;
                for &col_idx in x.col_indices() {
                    let val = row.get_entry(col_idx).unwrap().into_value();
                    let effective_val = if self.center {
                        val - mean[col_idx] // Using the unwrapped mean
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

    pub fn feature_importances(&self) -> anyhow::Result<Array2<f64>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| anyhow!("Model must be fitted first!"))?;

        let importances = components.mapv(|x| x * x);

        Ok(importances)
    }

    pub fn explained_variance_ratio(&self) -> anyhow::Result<Array1<f64>> {
        let explained_variance = self
            .explained_variance_
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model must be fitted first"))?;

        let total_variance = explained_variance.sum();
        let ratios = explained_variance.mapv(|v| v / total_variance);

        Ok(ratios)
    }

    pub fn commulative_explained_variance_ratio(&self) -> anyhow::Result<Array1<f64>> {
        let ratios = self.explained_variance_ratio()?;
        let mut cummulative = Array1::zeros(ratios.len());
        let mut sum = 0.0;

        for (i, &ratio) in ratios.iter().enumerate() {
            sum += ratio;
            cummulative[i] = sum;
        }

        Ok(cummulative)
    }

    pub fn fit_transform(&mut self, x: &CsrMatrix<f64>) -> anyhow::Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra_sparse::CooMatrix;

    fn create_test_matrix() -> CsrMatrix<f64> {
        // Create a simple test matrix with known patterns
        // Using a 5x4 matrix to ensure n_components (2) is less than min dimension
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let cols = vec![0, 1, 1, 2, 0, 2, 1, 3, 0, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let coo = CooMatrix::try_from_triplets(
            5, // 5 rows
            4, // 4 columns
            rows, cols, vals,
        )
        .unwrap();

        CsrMatrix::from(&coo)
    }

    #[test]
    fn test_initialization() {
        // Test with reasonable number of components
        let spca = SparsePCA::new(
            2, // n_components should be less than min(n_samples, n_features)
            0.1,
            Some(1e-6),
            Some(42),
            true,
            false,
        );

        assert_eq!(spca.n_components, 2);
        assert_eq!(spca.alpha, 0.1);
        assert_eq!(spca.tollerance, 1e-6);
        assert_eq!(spca.random_seed, 42);
        assert!(spca.components_.is_none());
        assert!(spca.explained_variance_.is_none());
        assert!(spca.mean_.is_none());

        // Test matrix dimensions
        let x = create_test_matrix();
        assert_eq!(x.nrows(), 5, "Test matrix should have 5 rows");
        assert_eq!(x.ncols(), 4, "Test matrix should have 4 columns");
        assert!(
            spca.n_components < x.ncols(),
            "n_components should be less than n_features"
        );
    }

    #[test]
    fn test_basic_fit() -> anyhow::Result<()> {
        let mut spca = SparsePCA::new(2, 0.1, None, None, true, false);
        let x = create_test_matrix();

        spca.fit(&x)?;

        // Check if components were created
        assert!(spca.components_.is_some());
        let components = spca.components_.as_ref().unwrap();

        // Check dimensions
        assert_eq!(components.shape(), &[2, 4]);

        // Check if mean was computed
        assert!(spca.mean_.is_some());
        let mean = spca.mean_.as_ref().unwrap();
        assert_eq!(mean.len(), 4);

        Ok(())
    }

    #[test]
    fn test_transform() -> anyhow::Result<()> {
        let mut spca = SparsePCA::new(2, 0.1, None, None, true, false);
        let x = create_test_matrix();

        let transformed = spca.fit_transform(&x)?;

        // Check dimensions of transformed data
        assert_eq!(transformed.shape(), &[5, 2]);

        // Transform should work on fitted model
        let transformed2 = spca.transform(&x)?;
        assert_eq!(transformed2.shape(), &[5, 2]);

        // Both transforms should give same results
        for i in 0..transformed.shape()[0] {
            for j in 0..transformed.shape()[1] {
                assert_relative_eq!(transformed[[i, j]], transformed2[[i, j]], epsilon = 1e-10);
            }
        }

        Ok(())
    }

    #[test]
    fn test_feature_importances() -> anyhow::Result<()> {
        let mut spca = SparsePCA::new(2, 0.1, None, None, true, false);
        let x = create_test_matrix();

        spca.fit(&x)?;
        let importances = spca.feature_importances()?;

        // Check dimensions
        assert_eq!(importances.shape(), &[2, 4]);

        // Check if importances are non-negative
        assert!(importances.iter().all(|&x| x >= 0.0));

        Ok(())
    }

    #[test]
    fn test_explained_variance_ratio() -> anyhow::Result<()> {
        let mut spca = SparsePCA::new(2, 0.1, None, None, true, false);
        let x = create_test_matrix();

        spca.fit(&x)?;

        // Test regular ratios
        let ratios = spca.explained_variance_ratio()?;
        assert_eq!(ratios.len(), 2);
        assert!((ratios.sum() - 1.0).abs() < 1e-10);

        // Test cumulative ratios
        let cum_ratios = spca.commulative_explained_variance_ratio()?;
        assert_eq!(cum_ratios.len(), 2);
        assert!(cum_ratios[0] <= cum_ratios[1]);
        assert!((cum_ratios[1] - 1.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_error_cases() {
        // Test error when n_components >= n_features
        let mut spca = SparsePCA::new(5, 0.1, None, None, true, false);
        let x = create_test_matrix();

        assert!(spca.fit(&x).is_err());

        // Test error when transforming before fitting
        let spca = SparsePCA::new(2, 0.1, None, None, true, false);
        assert!(spca.transform(&x).is_err());
        assert!(spca.feature_importances().is_err());
        assert!(spca.explained_variance_ratio().is_err());
    }

    #[test]
    fn test_center_vs_no_center() -> anyhow::Result<()> {
        let x = create_test_matrix();

        // With centering
        let mut spca_centered = SparsePCA::new(2, 0.1, None, None, true, false);
        let transformed_centered = spca_centered.fit_transform(&x)?;

        // Without centering
        let mut spca_uncentered = SparsePCA::new(2, 0.1, None, None, false, false);
        let transformed_uncentered = spca_uncentered.fit_transform(&x)?;

        // Results should be different
        let mut all_equal = true;
        for i in 0..transformed_centered.shape()[0] {
            for j in 0..transformed_centered.shape()[1] {
                if (transformed_centered[[i, j]] - transformed_uncentered[[i, j]]).abs() > 1e-10 {
                    all_equal = false;
                    break;
                }
            }
        }
        assert!(!all_equal, "Centered and uncentered results should differ");

        Ok(())
    }
}
