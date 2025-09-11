//! # Single Algebra
//!
//! A high-performance linear algebra library optimized for sparse matrices and 
//! dimensionality reduction algorithms. Designed for machine learning, data analysis,
//! and scientific computing applications where efficiency with sparse data is crucial.
//!
//! ## Core Modules
//!
//! ### Matrix Operations
//! - [`sparse`]: Sparse matrix implementations (CSR, CSC) with efficient operations
//! - [`dense`]: Dense matrix utilities and operations
//!
//! ### Dimensionality Reduction
//! - [`dimred`]: Principal Component Analysis (PCA) and planned manifold learning algorithms
//!
//! ### Utilities
//! - [`Normalize`]: Data normalization transformations
//! - [`Log1P`]: Logarithmic transformations for numerical stability
//!
//! ## Key Features
//!
//! - **Sparse matrix efficiency**: Optimized CSR/CSC formats for memory and computational efficiency
//! - **Dimensionality reduction**: PCA with both Lanczos and randomized SVD algorithms
//! - **Feature masking**: Selective analysis of feature subsets
//! - **Parallel processing**: Multi-threaded operations for large datasets
//! - **Type flexibility**: Generic implementations supporting `f32` and `f64`
//!
//! ## Typical Workflow
//!
//! 1. Load or create sparse matrices using the [`sparse`] module
//! 2. Apply preprocessing with [`Normalize`] or [`Log1P`] utilities
//! 3. Perform dimensionality reduction using [`dimred::pca`] algorithms
//! 4. Analyze results with variance explanations and feature importance
//!
//! ## Performance Focus
//!
//! This library is optimized for scenarios involving:
//! - Large, sparse datasets (e.g., text analysis, genomics, recommendation systems)
//! - Memory-constrained environments
//! - High-dimensional data requiring dimensionality reduction
//! - Scientific computing workflows requiring numerical precision

pub mod dense;
pub mod sparse;

pub mod dimred;

mod utils;

pub use utils::Normalize;
pub use utils::Log1P;

