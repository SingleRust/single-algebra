# single-algebra 🧮

A powerful linear algebra and machine learning utilities library for Rust, providing efficient matrix operations, dimensionality reduction, and statistical analysis tools.

## Features 🚀

- **Efficient Matrix Operations**: Support for both dense and sparse matrices (CSR/CSC formats)
- **Dimensionality Reduction**: PCA implementations for both dense and sparse matrices
- **SVD Implementations**: Multiple SVD backends including LAPACK and Faer
- **Statistical Analysis**: Comprehensive statistical operations with batch processing support
- **Similarity Measures**: Collection of distance/similarity metrics for high-dimensional data
- **Masking Support**: Selective data processing with boolean masks
- **Parallel Processing**: Efficient multi-threaded implementations using Rayon
- **Feature-Rich**: Configurable through feature flags for specific needs

## Matrix Operations 📊

- **SVD Decomposition**: Choose between parallel, LAPACK, or Faer implementations
- **Sparse Matrix Support**: Comprehensive operations for CSR and CSC sparse matrix formats
- **Masked Operations**: Selective data processing with boolean masks
- **Batch Processing**: Statistical operations grouped by batch identifiers
- **Normalization**: Row and column normalization with customizable targets

## Dimensionality Reduction ⬇️

- **PCA Framework**: Flexible implementation with customizable SVD backends
- **Dense Matrix PCA**: Optimized implementation for dense matrices
- **Sparse Matrix PCA**: Memory-efficient PCA for sparse matrices
- **Masked Sparse PCA**: Apply PCA on selected features only
- **Incremental Processing**: Support for large datasets that don't fit in memory

## Similarity Measures 📏

- **Cosine Similarity**: Measure similarity based on the cosine of the angle between vectors
- **Euclidean Similarity**: Similarity based on Euclidean distance
- **Pearson Similarity**: Measure linear correlation between vectors
- **Manhattan Similarity**: Similarity based on Manhattan distance
- **Jaccard Similarity**: Measure similarity as intersection over union

## Statistical Analysis 📈

- **Basic Statistics**: Mean, variance, sum, min/max operations
- **Batch Statistics**: Compute statistics grouped by batch identifiers
- **Matrix Variance**: Efficient variance calculations for matrices
- **Nonzero Counting**: Count non-zero elements in sparse matrices
- **Masked Statistics**: Compute statistics on selected rows/columns only

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-algebra = "0.5.0"
```

### Feature Flags

Enable optional features based on your needs:

```toml
[dependencies]
single-algebra = { version = "0.5.0", features = ["lapack", "faer"] }
```

Available features:
- `smartcore`: Enable integration with the SmartCore machine learning library
- `lapack`: Use the LAPACK backend for linear algebra operations
- `faer`: Use the Faer backend for linear algebra operations
- `simba`: Enable SIMD optimizations via simba
- `clustering`: Enable clustering algorithms (includes network and local_moving)
- `network`: Enable network-based algorithms
- `local_moving`: Enable local moving algorithm for community detection

## Usage Examples

### Basic PCA with LAPACK Backend

```rust
use ndarray::{Array2, ArrayView2};
use single_algebra::dimred::pca::dense::{PCABuilder, LapackSVD};

// Create a sample matrix
let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

// Build PCA with LAPACK backend
let mut pca = PCABuilder::new(LapackSVD)
    .n_components(2)
    .center(true)
    .scale(false)
    .build();

// Fit and transform data
pca.fit(data.view()).unwrap();
let transformed = pca.transform(data.view()).unwrap();

// Access results
let components = pca.components().unwrap();
let explained_variance = pca.explained_variance_ratio().unwrap();
```

### Sparse Matrix Operations

```rust
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use single_algebra::sparse::MatrixSum;

// Create a sparse matrix
let mut coo = CooMatrix::new(3, 3);
coo.push(0, 0, 1.0);
coo.push(1, 1, 2.0);
coo.push(2, 2, 3.0);
let csr: CsrMatrix<f64> = (&coo).into();

// Calculate column sums
let col_sums: Vec<f64> = csr.sum_col().unwrap();
```

### Batch Processing

```rust
use nalgebra_sparse::CsrMatrix;
use single_algebra::sparse::BatchMatrixMean;

// Sample data with batch identifiers
let matrix = create_sparse_matrix();
let batches = vec!["batch1", "batch1", "batch2", "batch2", "batch3"];

// Calculate mean per batch
let batch_means = matrix.mean_batch_col(&batches).unwrap();

// Access results for a specific batch
let batch1_means = batch_means.get("batch1").unwrap();
```

### Similarity Measures

```rust
use ndarray::Array1;
use single_algebra::similarity::{SimilarityMeasure, CosineSimilarity};

let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

let cosine = CosineSimilarity;
let similarity = cosine.calculate(a.view(), b.view());
```

## Performance Considerations

- For large matrices, consider using sparse representations (CSR/CSC)
- Enable the appropriate backend (`lapack` or `faer`) based on your needs
- Use masked operations when working with subsets of data
- Batch processing can significantly improve performance for grouped operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE.md file for details.

## Acknowledgments

- The LAPACK integration is built upon the `nalgebra-lapack` crate
- Some components are inspired by scikit-learn's implementations
- The Faer backend leverages the high-performance `faer` crate