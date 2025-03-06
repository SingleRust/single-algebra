# single-algebra 🧮

The companion algebra library for single-rust, providing powerful matrix operations and machine learning utilities.

## Features 🚀

- Efficient operations on sparse and dense matrices
- Dimensionality reduction techniques
- Clustering algorithms including Louvain community detection
- Batch processing utilities with masking support
- Statistical analysis and inference
- More features planned!

## Matrix Operations 📊

- SVD decomposition with parallel and LAPACK implementations
- Matrix convenience functions for statistical operations
- Support for both CSR and CSC sparse matrix formats
- Masked operations for selective data processing
- Batch-wise statistics (mean, variance) with flexible batch identifiers

## Clustering 🔍

- Louvain community detection
- Similarity network construction
- K-nearest neighbors graph building
- Local moving algorithm for community refinement
- Leiden clustering implementation (work in progress)

## Dimensionality Reduction ⬇️

- Incremental PCA implementation
- Support for sparse matrices in dimensionality reduction
- SVD-based compression and analysis

## Statistical Analysis 📈

- Multiple testing correction methods
- Parametric and non-parametric hypothesis testing
- Effect size calculations
- Batch-wise statistical comparisons

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-algebra = "0.2.2-alpha.0"
```

## Batch Processing

The library now includes flexible batch processing capabilities with the `BatchIdentifier` trait, which supports common identifier types:

- String and string slices
- Integer types (i32, u32, usize)
- Custom types (by implementing the trait)

```rust
// Example of batch-wise statistics
let batches = vec!["batch1", "batch2", "batch3"];
let batch_means = matrix.mean_batch_col(&batches)?;
```

## Masked Operations

Selective processing is now available through masked operations:

```rust
// Only process selected columns
let mask = vec![true, false, true, true, false];
let masked_sums = matrix.sum_col_masked(&mask)?;
```

## Acknowledgments 🙏

The Louvain clustering implementation was adapted from [louvain-rs](https://github.com/graphext/louvain-rs/tree/master) written by Juan Morales (crispamares@gmail.com). The original implementation has been modified to better suit the needs of single-algebra.