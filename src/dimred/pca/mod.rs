//! # Principal Component Analysis (PCA)
//!
//! This module provides Principal Component Analysis implementations for sparse matrices.
//! PCA is a dimensionality reduction technique that finds the principal components
//! (directions of maximum variance) in high-dimensional data.
//!
//! ## Available Implementations
//!
//! - [`SparsePCA`] - Standard PCA for sparse CSR matrices
//! - [`MaskedSparsePCA`] - PCA with feature masking for selective analysis
//!
//! ## SVD Methods
//!
//! Two SVD algorithms are supported via [`SVDMethod`]:
//! - **Lanczos**: Deterministic iterative method, good for general use
//! - **Randomized**: Faster approximation method with configurable parameters
//!
//! ## Key Features
//!
//! - **Sparse matrix support**: Efficient handling of sparse CSR matrices
//! - **Optional centering**: Data can be centered or used as-is
//! - **Explained variance**: Calculate variance ratios and cumulative variance
//! - **Feature masking**: Selectively include/exclude features in analysis
//! - **Builder patterns**: Convenient configuration with sensible defaults
//!
//! ## Usage Pattern
//!
//! 1. Create PCA instance using builder pattern
//! 2. Fit the model to training data
//! 3. Transform data to reduced dimensions
//! 4. Analyze explained variance and feature importance

mod sparse;

mod sparse_masked;

pub use sparse::SparsePCA;
pub use sparse::SparsePCABuilder;
pub use sparse_masked::MaskedSparsePCA;
pub use sparse_masked::MaskedSparsePCABuilder;
pub use single_svdlib::randomized::PowerIterationNormalizer;
pub use single_svdlib::SvdFloat;

/// SVD computation method for PCA.
///
/// Determines the algorithm used for singular value decomposition:
/// - `Lanczos`: Iterative method with guaranteed convergence
/// - `Random`: Faster randomized approximation with tunable accuracy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVDMethod {
    /// Lanczos algorithm for exact SVD computation
    Lanczos,
    /// Randomized SVD for faster approximation
    Random {
        /// Number of extra samples for oversampling (typically 5-20)
        n_oversamples: usize,
        /// Number of power iterations for accuracy (typically 2-7)
        n_power_iterations: usize,
        /// Normalization method for power iterations
        normalizer: PowerIterationNormalizer,
    },
}

impl Default for SVDMethod {
    fn default() -> Self {
        Self::Lanczos
    }
}

