//! t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation.
//! 
//! This module uses the Barnes-Hut t-SNE algorithm for efficient dimensionality reduction
//! of high-dimensional data, primarily used for visualization.

use ndarray::{Array2, ArrayD, ArrayViewD};

pub struct TSNEConfig {
    pub output_dim: u8,
    pub perplexity: f32,
    pub epochs: usize,
    pub theta: f32,
}

impl Default for TSNEConfig {
    fn default() -> Self {
        Self {
            output_dim: 2,
            perplexity: 30.0,
            epochs: 1000,
            theta: 0.5,
        }
    }
}

impl TSNEConfig {
    pub fn new(output_dim: u8, perplexity: f32, epochs: usize, theta: f32) -> Self {
        Self {
            output_dim,
            perplexity,
            epochs,
            theta,
        }
    }
}

pub fn run_f32(
    x: ArrayViewD<f32>,
    config: TSNEConfig,
) -> anyhow::Result<ArrayD<f32>> {
    let n_obs = x.shape()[0];
    let n_dim = x.shape()[1];
    
    // Ensure we have a standard layout for chunking
    let x_standard = if x.is_standard_layout() {
        x
    } else {
        // This is a bit expensive but ensures correctness
        return Err(anyhow::anyhow!("Input must be in standard layout"));
    };

    let x_slice = x_standard.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get slice from array"))?;

    let x_chunked_slice: Vec<&[f32]> = x_slice.chunks(n_dim).collect();
    let tsne_result = bhtsne::tSNE::new(&x_chunked_slice)
        .embedding_dim(config.output_dim)
        .perplexity(config.perplexity)
        .epochs(config.epochs)
        .barnes_hut(config.theta, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .embedding();

    let result = Array2::from_shape_vec((n_obs, config.output_dim as usize), tsne_result)?;
    Ok(result.into_dyn())
}

pub fn run_f64(
    x: ArrayViewD<f64>,
    config: TSNEConfig,
) -> anyhow::Result<ArrayD<f64>> {
    let n_obs = x.shape()[0];
    let n_dim = x.shape()[1];
    
    let x_standard = if x.is_standard_layout() {
        x
    } else {
        return Err(anyhow::anyhow!("Input must be in standard layout"));
    };

    let x_slice = x_standard.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get slice from array"))?;

    let x_chunked_slice: Vec<&[f64]> = x_slice.chunks(n_dim).collect();
    let tsne_result = bhtsne::tSNE::new(&x_chunked_slice)
        .embedding_dim(config.output_dim)
        .perplexity(config.perplexity as f64)
        .epochs(config.epochs)
        .barnes_hut(config.theta as f64, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .embedding();

    let result = Array2::from_shape_vec((n_obs, config.output_dim as usize), tsne_result)?;
    Ok(result.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_tsne_f32_dimensions() {
        // Small dataset: 20 points in 5D
        // Perplexity should be smaller than N/3 usually, but bhtsne handles it
        let data = Array2::from_elem((20, 5), 1.0f32);
        let config = TSNEConfig {
            output_dim: 2,
            perplexity: 5.0,
            epochs: 10, // Fast test
            theta: 0.5,
        };

        let result = run_f32(data.view().into_dyn(), config).unwrap();
        assert_eq!(result.shape(), &[20, 2]);
    }

    #[test]
    fn test_tsne_f64_dimensions() {
        let data = Array2::from_elem((20, 5), 1.0f64);
        let config = TSNEConfig {
            output_dim: 3,
            perplexity: 5.0,
            epochs: 10,
            theta: 0.5,
        };

        let result = run_f64(data.view().into_dyn(), config).unwrap();
        assert_eq!(result.shape(), &[20, 3]);
    }

    #[test]
    fn test_tsne_separation() {
        // Create two very distant clusters
        // Cluster 1: near (0, 0, 0)
        // Cluster 2: near (100, 100, 100)
        let mut data = Array2::zeros((20, 3));
        for i in 0..10 {
            data[[i, 0]] = 0.0;
            data[[i, 1]] = 0.0;
            data[[i, 2]] = 0.0;
        }
        for i in 10..20 {
            data[[i, 0]] = 100.0;
            data[[i, 1]] = 100.0;
            data[[i, 2]] = 100.0;
        }

        let config = TSNEConfig {
            output_dim: 2,
            perplexity: 5.0,
            epochs: 200, // Enough to see some structure
            theta: 0.5,
        };

        let result = run_f64(data.view().into_dyn(), config).unwrap();
        
        // Calculate average distance between points in same cluster vs different clusters
        // Or simply check if cluster 1 points are closer to each other than to cluster 2 points
        let p1 = result.slice(ndarray::s![0, ..]);
        let p2 = result.slice(ndarray::s![1, ..]);
        let p11 = result.slice(ndarray::s![11, ..]);
        
        let dist_same = ((&p1 - &p2).mapv(|x| x*x).sum()).sqrt();
        let dist_diff = ((&p1 - &p11).mapv(|x| x*x).sum()).sqrt();
        
        // In t-SNE, clusters should eventually separate. 
        // With only 200 epochs it might not be perfect, but dist_diff should generally be > dist_same
        // This is a soft test.
        assert!(dist_diff > dist_same, "Clusters should be somewhat separated: same={}, diff={}", dist_same, dist_diff);
    }
}
