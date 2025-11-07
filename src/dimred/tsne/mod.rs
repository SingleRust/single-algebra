// Still working on that
// Probably going to implement, either using Linfa_tsne or another approach

use ndarray::{Array2, ArrayD, ArrayViewD};
use single_utilities::traits::FloatOpsTS;

pub struct TSNEConfig {
    output_dim: u8,
    perplexity: f32,
    epochs: usize,
    theta: f32,
}

pub fn run_f32<T: FloatOpsTS>(
    x: ArrayViewD<f32>,
    config: TSNEConfig,
) -> anyhow::Result<ArrayD<f32>> {
    let n_obs = x.shape()[0];
    let n_dim = x.shape()[1];
    let x_slice = x.as_slice().unwrap();

    let x_chunked_slice: Vec<&[f32]> = x_slice.chunks(n_dim).collect();
    let tsne_result = bhtsne::tSNE::new(&x_chunked_slice)
        .embedding_dim(config.output_dim)
        .perplexity(config.perplexity)
        .epochs(config.epochs)
        .barnes_hut(config.theta, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(&a, &b)| num_traits::Float::powi(a - b, 2))
                .sum::<f32>()
                .sqrt()
        })
        .embedding();

    let result = Array2::from_shape_vec((n_obs, config.output_dim as usize), tsne_result)?;
    Ok(result.into_dyn())
}

pub fn run_f64<T: FloatOpsTS>(
    x: ArrayViewD<f64>,
    config: TSNEConfig,
) -> anyhow::Result<ArrayD<f64>> {
    let n_obs = x.shape()[0];
    let n_dim = x.shape()[1];
    let x_slice = x.as_slice().unwrap();

    let x_chunked_slice: Vec<&[f64]> = x_slice.chunks(n_dim).collect();
    let tsne_result = bhtsne::tSNE::new(&x_chunked_slice)
        .embedding_dim(config.output_dim)
        .perplexity(config.perplexity as f64)
        .epochs(config.epochs)
        .barnes_hut(config.theta as f64, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(&a, &b)| num_traits::Float::powi(a - b, 2))
                .sum::<f64>()
                .sqrt()
        })
        .embedding();

    let result = Array2::from_shape_vec((n_obs, config.output_dim as usize), tsne_result)?;
    Ok(result.into_dyn())
}
