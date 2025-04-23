// https://github.com/graphext/louvain-rs/tree/master
// Copyright 2018 Juan Morales (crispamares@gmail.com)
// Repository: https://github.com/graphext/louvain-rs/tree/master
// Licensed under the MIT License.
use crate::network::Network;
use crate::similarity::SimilarityMeasure;
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::{Array2, ArrayD, ArrayViewD};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use petgraph::graph::UnGraph;
use std::collections::HashSet;
use kiddo::traits::DistanceMetric;

pub fn create_similarity_network<T, S>(
    data: &Array2<T>,
    similarity: &S,
    threshold: f64,
) -> Network<f64, f64>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
    S: SimilarityMeasure,
{
    let n_samples = data.nrows();
    let mut graph = UnGraph::with_capacity(n_samples, n_samples * (n_samples - 1) / 2);
    let mut node_indices = Vec::with_capacity(n_samples);

    // Add nodes
    for _ in 0..n_samples {
        node_indices.push(graph.add_node(1.0));
    }

    let mut seen = vec![HashSet::new(); n_samples];
    let mut node_weights = vec![0.0; n_samples];

    // Calculate similarities and add edges
    for i in 0..n_samples {
        let row_i = data.row(i);
        for j in (i + 1)..n_samples {
            let row_j = data.row(j);

            let sim = similarity.calculate(row_i, row_j);

            if sim > threshold && seen[i].insert(j as u32) {
                graph.add_edge(node_indices[i], node_indices[j], sim);
                node_weights[j] += 1.0;
                node_weights[i] += 1.0;
            }
        }
    }

    // Update node weights
    for &i in &node_indices {
        *graph.node_weight_mut(i).unwrap() = node_weights[i.index()];
    }

    Network::new_from_graph(graph)
}

pub fn build_knn_network_combined_matrix_arrayd<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<CsrMatrix<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Default + 'static,
    T: num_traits::float::FloatCore,
    T: std::fmt::Debug,
    T: std::ops::AddAssign,
    D: DistanceMetric<T, K>
{
    if data.ndim() != 2 {
        return Err(anyhow::anyhow!("The input array has to be two dimensional in order to be used to build a knn network."));
    }

    let shape = data.shape();
    let n_samples = shape[0] as u64;
    let n_features = shape[1] as u64;

    if (n_features as usize) < K {
        return Err(anyhow::anyhow!("The data must have at least K features in order to be used for building a knn network."))
    }

    let mut kdtree: KdTree<T, K> = KdTree::new();

    for i in 0..n_samples {
        let mut point_array = [T::zero(); K];
        for j in 0..K {
            point_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }
        kdtree.add(&point_array, i);
    }

    let mut all_distances = Vec::with_capacity((n_samples * k) as usize);
    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }
        let neighbors = kdtree.nearest_n::<D>(&query_array, (k+1) as usize);
        for neighbor in neighbors.iter().skip(1) {
            all_distances.push(neighbor.distance);
        }
    }

    all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_idx = all_distances.len() / 2;
    let global_sigma = if all_distances.is_empty() {
        T::from_f64(1.0).unwrap()
    } else {
        all_distances[median_idx] / T::from_f64(f64::ln(k as f64)).unwrap()
    };

    let mut triplets = Vec::with_capacity((n_samples * k) as usize);

    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }

        let neighbors = kdtree.nearest_n::<D>(&query_array, (k+1) as usize);

        for neighbor in neighbors.iter().skip(1) {
            if i <= neighbor.item {
                let weight = (-neighbor.distance / global_sigma).exp();
                let min_weight = T::from_f64(1e-6).unwrap();
                if weight > min_weight {
                    triplets.push((i as usize, neighbor.item as usize, weight));
                }
            }
        }
    }

    let coo = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        triplets.iter().map(|&(i, _, _)| i).collect(),
        triplets.iter().map(|&(_, j, _)| j).collect(),
        triplets.iter().map(|&(_, _, v)| v).collect(),
    )
        .map_err(|e| anyhow::anyhow!("Failed to create COO matrix: {}", e))?;

    Ok(CsrMatrix::from(&coo))
}


pub fn build_knn_network_combined_matrix<T, const K: usize>(
    data: &Array2<T>,
    k: u64,
) -> anyhow::Result<CsrMatrix<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Default + 'static,
    T: num_traits::float::FloatCore,
    T: std::fmt::Debug,
    T: std::ops::AddAssign,
{
    let n_samples = data.nrows() as u64;
    let n_features = data.ncols() as u64;

    if (n_features as usize) < K {
        return Err(anyhow::anyhow!("Data must have at least K features"));
    }

    let mut kdtree: KdTree<T, K> = KdTree::new();

    for i in 0..n_samples {
        let mut point_array = [T::zero(); K];
        for j in 0..K {
            point_array[j] = data[(i as usize, j)];
        }
        kdtree.add(&point_array, i);
    }

    let mut triplets = Vec::with_capacity((n_samples * k) as usize);

    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = data[(i as usize, j)];
        }

        let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&query_array, (k + 1) as usize);

        // Skip first result (self) and process remaining neighbors
        for neighbor in neighbors.iter().skip(1) {
            if i <= neighbor.item {
                let weight = (-neighbor.distance.sqrt()).exp();
                triplets.push((i as usize, neighbor.item as usize, weight));
            }
        }
    }

    let coo = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        triplets.iter().map(|&(i, _, _)| i).collect(),
        triplets.iter().map(|&(_, j, _)| j).collect(),
        triplets.iter().map(|&(_, _, v)| v).collect(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create COO matrix: {}", e))?;

    Ok(CsrMatrix::from(&coo))
}

/// returns first, connectivity matrix, secondly adjacency
pub fn build_knn_network_separate_matrix<T, const K: usize>(
    data: &Array2<T>,
    k: u64,
) -> anyhow::Result<(CsrMatrix<T>, CsrMatrix<T>)>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Default + 'static,
    T: num_traits::float::FloatCore,
    T: std::fmt::Debug,
    T: std::ops::AddAssign,
{
    let n_samples = data.nrows() as u64;
    let n_features = data.ncols() as u64;

    if (n_features as usize) < K {
        return Err(anyhow::anyhow!("Data must have at least K features"));
    }

    let mut kdtree: KdTree<T, K> = KdTree::new();

    for i in 0..n_samples {
        let mut point_array = [T::zero(); K];
        for j in 0..K {
            point_array[j] = data[(i as usize, j)];
        }
        kdtree.add(&point_array, i);
    }

    let capacity = (n_samples * k) as usize;
    let mut conn_triplets = Vec::with_capacity(capacity);
    let mut adj_triplets = Vec::with_capacity(capacity);

    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = data[(i as usize, j)];
        }

        let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&query_array, (k + 1) as usize);

        // Skip first result (self) and process remaining neighbors
        for neighbor in neighbors.iter().skip(1) {
            if i <= neighbor.item {
                let weight = (-T::from(neighbor.distance).unwrap().sqrt()).exp();

                // Connectivity matrix gets 1.0 for connected points
                conn_triplets.push((i as usize, neighbor.item as usize, T::one()));

                // Adjacency matrix gets the actual distances
                adj_triplets.push((i as usize, neighbor.item as usize, weight));
            }
        }
    }
    let connectivity = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        conn_triplets.iter().map(|&(i, _, _)| i).collect(),
        conn_triplets.iter().map(|&(_, j, _)| j).collect(),
        conn_triplets.iter().map(|&(_, _, v)| v).collect(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create connectivity matrix: {}", e))?;

    // Create adjacency matrix
    let adjacency = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        adj_triplets.iter().map(|&(i, _, _)| i).collect(),
        adj_triplets.iter().map(|&(_, j, _)| j).collect(),
        adj_triplets.iter().map(|&(_, _, v)| v).collect(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create adjacency matrix: {}", e))?;
    
    Ok((CsrMatrix::from(&connectivity), CsrMatrix::from(&adjacency)))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::{
        CosineSimilarity, EuclideanSimilarity, JaccardSimilarity, ManhattanSimilarity,
        PearsonSimilarity,
    };
    use ndarray::{arr2, array};
    use std::f64::EPSILON;

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 4.0, 6.0];
        let similarity = CosineSimilarity;

        let result = similarity.calculate(a.view(), b.view());
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_euclidean_similarity() {
        let a = array![0.0, 0.0];
        let b = array![1.0, 1.0];
        let similarity = EuclideanSimilarity::default();

        let result = similarity.calculate(a.view(), b.view());
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_pearson_similarity() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![2.0, 4.0, 6.0, 8.0];
        let similarity = PearsonSimilarity;

        let result = similarity.calculate(a.view(), b.view());
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_similarity() {
        let a = array![0.0, 0.0];
        let b = array![1.0, 1.0];
        let similarity = ManhattanSimilarity::default();

        let result = similarity.calculate(a.view(), b.view());
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = array![1.0, 0.0, 1.0, 1.0];
        let b = array![1.0, 1.0, 0.0, 1.0];
        let similarity = JaccardSimilarity::default();

        let result = similarity.calculate(a.view(), b.view());
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_knn_network() {
        let data = arr2(&[
            [1.0, 2.0, 9.0, 9.0], // Extra features will be ignored
            [1.1, 2.1, 8.0, 8.0],
            [5.0, 5.0, 7.0, 7.0],
            [5.1, 5.1, 6.0, 6.0],
        ]);

        // Using only first 2 features for KNN
        let result = build_knn_network_combined_matrix::<f64, 2>(&data, 2).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 4);

        let mut found_close = false;
        let mut found_far = false;

        for (row, col, &weight) in result.triplet_iter() {
            match (row, col) {
                (0, 1) | (1, 0) => {
                    assert!(weight > 0.5);
                    found_close = true;
                }
                (0, 2) | (0, 3) | (1, 2) | (1, 3) => {
                    assert!(weight < 0.5);
                    found_far = true;
                }
                _ => {}
            }
        }

        assert!(found_close);
        assert!(found_far);
    }

    #[test]
    fn test_more_features_than_k() {
        let data = arr2(&[
            [1.0, 2.0, 3.0, 4.0], // 4 features
            [1.1, 2.1, 5.0, 6.0],
        ]);

        // Using only first 2 features
        let result = build_knn_network_combined_matrix::<f64, 2>(&data, 1).unwrap();
        assert_eq!(result.nrows(), 2);
    }
}
