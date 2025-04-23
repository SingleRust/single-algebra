// Based on https://arxiv.org/html/2312.04876v2 ... still very much WIP
use ahash::AHasher;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::MulAssign;

use crate::network::clustering::NetworkGrouping;
use crate::network::Network;

pub struct ParallelLocalMoving<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign,
{
    resolution: T,
    cluster_weights: Vec<T>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
}

impl<T> ParallelLocalMoving<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign,
{
    pub fn new(resolution: T) -> Self {
        ParallelLocalMoving {
            resolution,
            cluster_weights: Vec::new(),
            nodes_per_cluster: Vec::new(),
            unused_clusters: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, size: usize) {
        self.cluster_weights.resize(size, T::zero());
        self.nodes_per_cluster.resize(size, 0);
        self.unused_clusters.resize(size, 0);
    }

    pub fn iterate<N, E, G>(&mut self, network: &Network<N, E>, clustering: &mut G) -> bool
    where
        N: Float + FromPrimitive + ToPrimitive + Send + Sync,
        E: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign + std::ops::AddAssign,
        G: NetworkGrouping + Send + Sync,
    {
        let node_count = network.nodes();
        self.ensure_capacity(node_count);

        // Initialize cluster weights and counts
        self.cluster_weights.fill(T::zero());
        self.nodes_per_cluster.fill(0);

        // Compute initial statistics serially
        for i in 0..node_count {
            let cluster = clustering.get_group(i);
            self.cluster_weights[cluster] = self.cluster_weights[cluster]
                + T::from_f64(network.weight(i).to_f64().unwrap()).unwrap();
            self.nodes_per_cluster[cluster] += 1;
        }

        // Find unused clusters
        let mut initial_num_unused_clusters = 0;
        for i in (0..node_count).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[initial_num_unused_clusters] = i;
                initial_num_unused_clusters += 1;
            }
        }

        let total_edge_weight =
            T::from_f64(network.get_total_edge_weight().to_f64().unwrap()).unwrap();
        let nodes: Vec<_> = (0..node_count).collect();

        // Calculate chunk size for parallel processing
        let chunk_size = ((node_count as f64) / (rayon::current_num_threads() as f64)) as usize;
        let chunk_size = std::cmp::max(256, chunk_size);

        let mut updates = vec![0usize; node_count];

        // Process nodes in parallel
        nodes
            .par_chunks(chunk_size)
            .zip(updates.par_chunks_mut(chunk_size))
            .for_each(|(nodes, updates)| {
                let mut neighboring_clusters = vec![0usize; node_count];
                let mut edge_weight_per_cluster = vec![T::zero(); node_count];

                // Process each node in the chunk
                for (node_idx, update) in nodes.iter().zip(updates.iter_mut()) {
                    let node = *node_idx;
                    let current_cluster = clustering.get_group(node);

                    // Handle current cluster status
                    let curr_cluster_nodes = self.nodes_per_cluster[current_cluster] - 1;
                    let curr_cluster_unused = curr_cluster_nodes == 0;

                    // Initialize neighboring clusters
                    if curr_cluster_unused {
                        neighboring_clusters[0] = current_cluster;
                    } else {
                        neighboring_clusters[0] =
                            self.unused_clusters[initial_num_unused_clusters - 1];
                    }
                    let mut num_neighboring_clusters = 1;

                    // Find neighboring clusters
                    for (target, weight) in network.neighbors(node) {
                        let neighbor_cluster = clustering.get_group(target);
                        let edge_weight = T::from_f64(weight.to_f64().unwrap()).unwrap();

                        if edge_weight_per_cluster[neighbor_cluster] == T::zero() {
                            neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                            num_neighboring_clusters += 1;
                        }
                        edge_weight_per_cluster[neighbor_cluster] =
                            edge_weight_per_cluster[neighbor_cluster] + edge_weight;
                    }

                    // Find best cluster
                    let mut best_cluster = if curr_cluster_unused {
                        current_cluster
                    } else {
                        self.unused_clusters[initial_num_unused_clusters - 1]
                    };
                    let mut max_qv_increment = T::neg_infinity();
                    let node_weight = T::from_f64(network.weight(node).to_f64().unwrap()).unwrap();

                    for &cluster in &neighboring_clusters[..num_neighboring_clusters] {
                        let cluster_weight = if cluster == current_cluster {
                            // Use most up-to-date information including node removal
                            self.cluster_weights[cluster] - node_weight
                        } else {
                            // Use previous iteration's information
                            self.cluster_weights[cluster]
                        };

                        let qv_increment = edge_weight_per_cluster[cluster]
                            - node_weight * cluster_weight * self.resolution
                                / (T::from_f64(2.0).unwrap() * total_edge_weight);

                        if qv_increment > max_qv_increment {
                            best_cluster = cluster;
                            max_qv_increment = qv_increment;
                        // Generalized minimum label heuristic
                        } else if qv_increment == max_qv_increment && cluster != current_cluster {
                            let mut h = AHasher::default();
                            cluster.hash(&mut h);
                            let l_hash = h.finish();

                            let mut h = AHasher::default();
                            best_cluster.hash(&mut h);
                            let best_cluster_hash = h.finish();

                            if l_hash < best_cluster_hash {
                                best_cluster = cluster;
                            }
                        }

                        edge_weight_per_cluster[cluster] = T::zero();
                    }

                    *update = best_cluster;
                }
            });

        // Apply updates and track changes serially
        let mut changed = false;
        for (i, new_cluster_label) in updates.into_iter().enumerate() {
            changed |= clustering.get_group(i) != new_cluster_label;
            clustering.set_group(i, new_cluster_label);
        }
        if changed {
            clustering.normalize_groups();
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::clustering::VectorGrouping;
    use petgraph::graph::UnGraph;

    fn create_test_network() -> Network<f64, f64> {
        let mut graph = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..6).map(|_| graph.add_node(1.0)).collect();

        // Create two distinct communities
        graph.add_edge(nodes[0], nodes[1], 2.0);
        graph.add_edge(nodes[1], nodes[2], 2.0);
        graph.add_edge(nodes[0], nodes[2], 2.0);

        graph.add_edge(nodes[3], nodes[4], 2.0);
        graph.add_edge(nodes[4], nodes[5], 2.0);
        graph.add_edge(nodes[3], nodes[5], 2.0);

        // Weak connection between communities
        graph.add_edge(nodes[2], nodes[3], 0.5);

        Network::new_from_graph(graph)
    }

    #[test]
    fn test_parallel_local_moving() {
        let network = create_test_network();
        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving: ParallelLocalMoving<f64> = ParallelLocalMoving::new(1.0);

        assert!(local_moving.iterate(&network, &mut clustering));
        assert_eq!(clustering.group_count(), 2);

        // Verify community structure
        let group_0 = clustering.get_group(0);
        assert_eq!(clustering.get_group(1), group_0);
        assert_eq!(clustering.get_group(2), group_0);

        let group_3 = clustering.get_group(3);
        assert_eq!(clustering.get_group(4), group_3);
        assert_eq!(clustering.get_group(5), group_3);

        assert_ne!(group_0, group_3);
    }

    #[test]
    fn test_single_community() {
        let mut graph = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..4).map(|_| graph.add_node(1.0)).collect();

        // Create strongly connected community
        graph.add_edge(nodes[0], nodes[1], 2.0);
        graph.add_edge(nodes[1], nodes[2], 2.0);
        graph.add_edge(nodes[2], nodes[3], 2.0);
        graph.add_edge(nodes[3], nodes[0], 2.0);

        let network = Network::new_from_graph(graph);
        let mut clustering = VectorGrouping::create_unified(network.nodes());
        let mut local_moving: ParallelLocalMoving<f64> = ParallelLocalMoving::new(1.0);

        assert!(!local_moving.iterate(&network, &mut clustering));
        assert_eq!(clustering.group_count(), 1);

        // All nodes should be in same community
        let group = clustering.get_group(0);
        for i in 1..network.nodes() {
            assert_eq!(clustering.get_group(i), group);
        }
    }
}
