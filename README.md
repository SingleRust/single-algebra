# single-algebra 🧮

A high-performance linear algebra library optimized for sparse matrices and dimensionality reduction algorithms. Designed for machine learning, data analysis, and scientific computing applications where memory efficiency and performance with sparse data are critical.

## Features

- **Multi-Backend Sparse Support**: Comprehensive trait implementations for both `nalgebra-sparse` and `sprs` (CSR/CSC formats).
- **Advanced Statistics**: Performant calculation of non-zeros, sums, means, and variances (including numerically stable 2-pass algorithms).
- **Dimensionality Reduction**: High-performance implementations of PCA, Sparse PCA, Masked Sparse PCA, and TSNE.
- **Preprocessing Utilities**: In-place normalization and Log1P transformations.
- **Multi-threaded Execution**: Seamless integration with `Rayon` for parallelizing large-scale matrix operations.
- **Numerically Stable**: Optimized for stability with high-dimensional data and large numerical offsets.

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
let col_vars: Vec<f64> = mat.var_col::<u32, f64>().unwrap();
```

## Performance

This library is designed to handle sparse datasets where >90% of values are zero. By leveraging format-aware iterators (e.g., prioritized row-iteration for CSR), we achieve $O(NNZ)$ performance for almost all statistical operations while minimizing heap allocations.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-algebra = "1.1.0"
```

## License

This project is licensed under the terms found in the LICENSE.md file.
