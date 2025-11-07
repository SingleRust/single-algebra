//! # Dimensionality Reduction
//!
//! This module provides algorithms for reducing the dimensionality of high-dimensional data
//! while preserving important structural properties. These techniques are essential for
//! data visualization, noise reduction, and computational efficiency improvements.
//!
//! ## Currently Available
//! - **PCA** ([`pca`]): Principal Component Analysis for linear dimensionality reduction
//!
//! ## Planned Implementations
//! - **t-SNE**: t-Distributed Stochastic Neighbor Embedding for non-linear visualization
//! - **UMAP**: Uniform Manifold Approximation and Projection for manifold learning
//! - Additional manifold learning techniques
//!
//! ## Algorithm Selection Guide
//! - Use **PCA** for linear relationships, feature analysis, and when interpretability is important
//! - Use **t-SNE** (when available) for non-linear visualization of clusters and local structure
//! - Use **UMAP** (when available) for preserving both local and global structure in embeddings

pub mod pca;
pub mod tsne;