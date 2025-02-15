# single-algebra 🧮

The companion algebra library for single-rust, providing powerful matrix operations and machine learning utilities.

## Features 🚀

- Efficient operations on sparse and dense matrices
- Dimensionality reduction techniques
- Clustering algorithms including Louvain community detection
- More features planned!

## Matrix Operations 📊

- SVD decomposition with parallel and LAPACK implementations
- Matrix convenience functions for statistical operations
- Support for both CSR and CSC sparse matrix formats

## Clustering 🔍

- Louvain community detection
- Similarity network construction
- K-nearest neighbors graph building
- Local moving algorithm for community refinement

## Dimensionality Reduction ⬇️

- Incremental PCA implementation
- Support for sparse matrices in dimensionality reduction

## Acknowledgments 🙏

The Louvain clustering implementation was adapted from [louvain-rs](https://github.com/graphext/louvain-rs/tree/master) written by Juan Morales (crispamares@gmail.com). The original implementation has been modified to better suit the needs of single-algebra.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-algebra = "0.2.0-alpha.0"
```