# single-algebra 🧮

A high-performance linear algebra library optimized for sparse matrices and dimensionality reduction algorithms. Designed for machine learning, data analysis, and scientific computing applications where memory efficiency and performance with sparse data are critical.

## Features

- **Multi-Backend Sparse Support**: Comprehensive trait implementations for both `nalgebra-sparse` and `sprs` (CSR/CSC formats).
- **Advanced Statistics**: Performant calculation of non-zeros, sums, means, and variances.
- **Numerically Stable**: Uses modified two-pass variance algorithms to prevent catastrophic cancellation with high-magnitude data offsets.
- **Optimized Top-N**: $O(N)$ selection of top elements using `select_nth_unstable` logic, avoiding expensive full sorts.
- **Dimensionality Reduction**: High-performance implementations of PCA, Sparse PCA, Masked Sparse PCA, and TSNE.
- **Preprocessing Utilities**: In-place normalization and Log1P transformations.
- **Universal Parallelization**: Seamless integration with `Rayon` across all core operations, with automatic thresholding for large datasets.

## Backends

### nalgebra-sparse
Full support for `CsrMatrix` and `CscMatrix`. Optimized for integration with the broader `nalgebra` ecosystem.

### sprs
High-performance support for `CsMatI<M, I>`. Supports generic index types and provides format-aware execution paths for maximum efficiency.

## Quick Start

```rust
use single_algebra::sparse::{MatrixSum, MatrixVariance};
use sprs::CsMat;

let mat = CsMat::new_csc((3, 3), vec![0, 2, 4, 5], vec![0, 2, 0, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
let row_sums: Vec<f64> = mat.sum_row().unwrap();
// Generic u32 count, f64 result
let col_vars: Vec<f64> = mat.var_col::<u32, f64>().unwrap();
```

## Performance

This library is designed to handle sparse datasets where >90% of values are zero. 

- **Format-Aware Iteration**: Prioritizes row-iteration for CSR and column-iteration for CSC to maximize cache locality and maintain $O(NNZ)$ complexity.
- **Memory Efficiency**: Minimizes heap allocations by providing `_chunk` methods for in-place buffer processing.
- **Automatic Concurrency**: Core operations (Sum, NonZero, Variance, MinMax) automatically parallelize via `Rayon` when the matrix exceeds 200,000 non-zero elements.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-algebra = "1.1.0"
```

## License

This project is licensed under the terms found in the LICENSE.md file.
