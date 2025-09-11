# single-algebra 🧮

A high-performance linear algebra library optimized for sparse matrices and dimensionality reduction algorithms. Designed for machine learning, data analysis, and scientific computing applications where efficiency with sparse data is crucial.

## Features 🚀

- **Sparse Matrix Operations**: Efficient CSR/CSC matrix implementations with comprehensive operations
- **Advanced PCA**: Multiple PCA variants including standard and masked sparse PCA
- **Flexible SVD**: Support for both Lanczos and randomized SVD algorithms
- **Feature Masking**: Selective analysis of feature subsets for targeted dimensionality reduction
- **Parallel Processing**: Multi-threaded operations using Rayon for large datasets
- **Memory Efficient**: Optimized for large, sparse datasets that don't fit in memory
- **Type Generic**: Supports both `f32` and `f64` numeric types
- **Utilities**: Data preprocessing with normalization and logarithmic transformations

## Core Modules 📊

### Sparse Matrix Operations
- **CSR/CSC Formats**: Comprehensive sparse matrix support with efficient storage
- **Matrix Arithmetic**: Sum operations, column statistics, and element-wise operations
- **Memory Optimization**: Designed for large, high-dimensional sparse datasets

### Dimensionality Reduction ⬇️
- **Sparse PCA**: Principal Component Analysis optimized for sparse CSR matrices
- **Masked Sparse PCA**: PCA with feature masking for selective analysis
- **SVD Algorithms**: Choice between Lanczos (exact) and randomized (fast) SVD methods
- **Variance Analysis**: Explained variance ratios and cumulative variance calculations
- **Feature Importance**: Component loading analysis for feature interpretation

### Data Preprocessing �
- **Normalization**: Row and column normalization utilities
- **Log Transformations**: Log1P transformations for numerical stability
- **Centering**: Optional data centering for PCA and other algorithms

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-algebra = "0.8.6"
```

## Usage Examples

### Sparse PCA with Builder Pattern

```rust
use nalgebra_sparse::CsrMatrix;
use single_algebra::dimred::pca::{SparsePCABuilder, SVDMethod};
use single_algebra::dimred::pca::sparse::PowerIterationNormalizer;

// Create or load your sparse matrix (samples × features)
let sparse_matrix: CsrMatrix<f64> = create_your_sparse_matrix();

// Build PCA with customized parameters
let mut pca = SparsePCABuilder::new()
    .n_components(50)
    .center(true)
    .verbose(true)
    .svd_method(SVDMethod::Random {
        n_oversamples: 10,
        n_power_iterations: 7,
        normalizer: PowerIterationNormalizer::QR,
    })
    .build();

// Fit and transform data
let transformed = pca.fit_transform(&sparse_matrix).unwrap();

// Analyze results
let explained_variance_ratio = pca.explained_variance_ratio().unwrap();
let cumulative_variance = pca.cumulative_explained_variance_ratio().unwrap();
let feature_importance = pca.feature_importances().unwrap();
```

### Masked Sparse PCA for Feature Subset Analysis

```rust
use single_algebra::dimred::pca::{MaskedSparsePCABuilder, SVDMethod};

// Create a feature mask (true = include, false = exclude)
let feature_mask = vec![true, false, true, true, false, true]; // Include features 0, 2, 3, 5

// Build masked PCA
let mut masked_pca = MaskedSparsePCABuilder::new()
    .n_components(10)
    .mask(feature_mask)
    .center(true)
    .verbose(true)
    .svd_method(SVDMethod::Lanczos)
    .build();

// Perform PCA on selected features only
let transformed = masked_pca.fit_transform(&sparse_matrix).unwrap();
```

### Sparse Matrix Operations

```rust
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use single_algebra::sparse::MatrixSum;

// Create a sparse matrix
let mut coo = CooMatrix::new(1000, 5000);
// ... populate with data ...
let csr: CsrMatrix<f64> = (&coo).into();

// Efficient column operations
let col_sums: Vec<f64> = csr.sum_col().unwrap();
let col_squared_sums: Vec<f64> = csr.sum_col_squared().unwrap();
```

### Data Preprocessing

```rust
use single_algebra::{Normalize, Log1P};

// Apply preprocessing transformations
let normalized_data = your_data.normalize()?;
let log_transformed = your_data.log1p()?;
```

## Algorithm Selection Guide

### When to Use Each PCA Variant

- **SparsePCA**: For standard dimensionality reduction on sparse matrices
- **MaskedSparsePCA**: When you need to analyze specific feature subsets or handle missing data patterns

### SVD Method Selection

- **Lanczos**: More accurate, deterministic results. Best for smaller problems or when precision is critical
- **Randomized**: Faster computation, especially for large matrices. Configurable accuracy vs. speed trade-off

### Performance Optimization

- Use sparse matrices (CSR format) for datasets with >90% zero values
- Enable verbose mode to monitor performance and convergence
- For very large datasets, consider using randomized SVD with appropriate oversampling
- Parallel processing is automatically utilized for transformation operations

## Planned Features 🚧

- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for non-linear visualization
- **UMAP**: Uniform Manifold Approximation and Projection for manifold learning
- **Additional similarity measures**: More distance metrics and similarity functions
- **Batch processing**: Enhanced support for processing data in chunks

## Performance Focus

This library is specifically optimized for:
- **Large sparse datasets** (text analysis, genomics, recommendation systems)
- **Memory-constrained environments**
- **High-dimensional data** requiring dimensionality reduction
- **Scientific computing** workflows requiring numerical precision

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE.md file for details.

## Acknowledgments

- The LAPACK integration is built upon the `nalgebra-lapack` crate
- Some components are inspired by scikit-learn's implementations
- The Faer backend leverages the high-performance `faer` crate