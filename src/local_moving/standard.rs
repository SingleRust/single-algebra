// See: https://en.wikipedia.org/wiki/Louvain_method & https://github.com/graphext/louvain-rs/tree/master
// Copyright 2018 Juan Morales (crispamares@gmail.com)
// Repository: https://github.com/graphext/louvain-rs/tree/master
// Licensed under the MIT License.
use crate::network::clustering::NetworkGrouping;
use crate::network::Network;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use std::iter::Sum;
use std::ops::MulAssign;

#[derive(Default)]
pub struct StandardLocalMoving<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign + std::ops::AddAssign,
{
    resolution: T,
    cluster_weights: Vec<T>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
    node_order: Vec<usize>,
    edge_weight_per_cluster: Vec<T>,
    neighboring_clusters: Vec<usize>,
}

impl<T> StandardLocalMoving<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign + std::ops::AddAssign,
{
    pub fn new(resolution: T) -> Self {
        StandardLocalMoving {
            resolution,
            cluster_weights: Vec::new(),
            nodes_per_cluster: Vec::new(),
            unused_clusters: Vec::new(),
            node_order: Vec::new(),
            edge_weight_per_cluster: Vec::new(),
            neighboring_clusters: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, size: usize) {
        self.cluster_weights.resize(size, T::zero());
        self.nodes_per_cluster.resize(size, 0);
        self.unused_clusters.resize(size, 0);
        self.node_order.resize(size, 0);
        self.edge_weight_per_cluster.resize(size, T::zero());
        self.neighboring_clusters.resize(size, 0);
    }

    pub fn iterate<N, E, C, R>(
        &mut self,
        network: &Network<N, E>,
        clustering: &mut C,
        rng: &mut R,
    ) -> bool
    where
        N: Float + FromPrimitive + ToPrimitive + Send + Sync,
        E: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign + std::ops::AddAssign,
        C: NetworkGrouping,
        R: RngCore,
    {
        let mut update = false;
        let node_count = network.nodes();
        self.ensure_capacity(node_count);

        self.cluster_weights.fill(T::zero());
        self.nodes_per_cluster.fill(0);
        self.edge_weight_per_cluster.fill(T::zero());

        for i in 0..node_count {
            let cluster = clustering.get_group(i);
            self.cluster_weights[cluster] = self.cluster_weights[cluster]
                + T::from_f64(network.weight(i).to_f64().unwrap()).unwrap();
            self.nodes_per_cluster[cluster] += 1;
        }

        let mut num_unused_clusters = 0;
        for i in (0..node_count).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[num_unused_clusters] = i;
                num_unused_clusters += 1;
            }
        }
        //println!("Number of unused clusters: {:?}", num_unused_clusters);

        self.node_order.clear();
        self.node_order.extend(0..node_count);
        self.node_order.shuffle(rng);

        let total_edge_weight =
            T::from_f64(network.get_total_edge_weight().to_f64().unwrap()).unwrap();
        let mut num_unstable_nodes = node_count;
        let mut i = 0;

        //println!("Total edge weight: {:?}, num unstable nodes: {:?}, i = {:?}", total_edge_weight.to_f64().unwrap(), num_unstable_nodes, i);

        while num_unstable_nodes > 0 {
            //println!("ITERATION | Total edge weight: {:?}, num unstable nodes: {:?}, i = {:?}", total_edge_weight.to_f64().unwrap(), num_unstable_nodes, i);
            let node = self.node_order[i];
            let current_cluster = clustering.get_group(node);
            //println!("ITERATION | Node: {:?}, Current Cluster: {:?}", node, current_cluster);

            // Remove node from current cluster
            let node_weight = T::from_f64(network.weight(node).to_f64().unwrap()).unwrap();
            self.cluster_weights[current_cluster] =
                self.cluster_weights[current_cluster] - node_weight;
            self.nodes_per_cluster[current_cluster] -= 1;

            if self.nodes_per_cluster[current_cluster] == 0 {
                self.unused_clusters[num_unused_clusters] = current_cluster;
                num_unused_clusters += 1;
                //println!("ITERATION | Nodes per cluster == 0, num unused clusters {:?}", num_unused_clusters);
            }

            // Find neighboring clusters
            self.neighboring_clusters[0] = self.unused_clusters[num_unused_clusters - 1];
            let mut num_neighboring_clusters = 1;

            for (target, weight) in network.neighbors(node) {
                let neighbor_cluster = clustering.get_group(target);
                let edge_weight = T::from_f64(weight.to_f64().unwrap()).unwrap();
                //println!("ITERATION | FORLOOP | target: {:?}, neighbor cluster: {:?}, edge_weight: {:?}", target, neighbor_cluster, edge_weight.to_f64().unwrap());
                if self.edge_weight_per_cluster[neighbor_cluster] == T::zero() {
                    //println!("ITERATION | FORLOOP | Is T::zero, edge_weight_per_cluster");
                    self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                    num_neighboring_clusters += 1;
                }
                self.edge_weight_per_cluster[neighbor_cluster] =
                    self.edge_weight_per_cluster[neighbor_cluster] + edge_weight;
            }

            // Find best cluster
            let mut best_cluster = current_cluster;
            let mut max_quality_increment = self.edge_weight_per_cluster[current_cluster]
                - (node_weight * self.cluster_weights[current_cluster] * self.resolution)
                    / (T::from_f64(2.0).unwrap() * total_edge_weight);

            //println!("ITERATION | Best Cluster {:?} Max Quality Increment {:?}", best_cluster, max_quality_increment.to_f64().unwrap());

            for &cluster in &self.neighboring_clusters[..num_neighboring_clusters] {
                let quality_increment = self.edge_weight_per_cluster[cluster]
                    - (node_weight * self.cluster_weights[cluster] * self.resolution)
                        / (T::from_f64(2.0).unwrap() * total_edge_weight);
                //println!("ITERATION | Cluster {:?} Quality Increment {:?}", cluster, quality_increment.to_f64().unwrap());
                if quality_increment > max_quality_increment
                    || (quality_increment == max_quality_increment && cluster < best_cluster)
                {
                    best_cluster = cluster;
                    max_quality_increment = quality_increment;
                    //println!("ITERATION | Passes if for quality improvement")
                }
                self.edge_weight_per_cluster[cluster] = T::zero();
            }

            // Update cluster assignment
            self.cluster_weights[best_cluster] = self.cluster_weights[best_cluster] + node_weight;
            self.nodes_per_cluster[best_cluster] += 1;

            if best_cluster == self.unused_clusters[num_unused_clusters - 1] {
                num_unused_clusters -= 1;
            }

            num_unstable_nodes -= 1;

            if best_cluster != current_cluster {
                clustering.set_group(node, best_cluster);
                update = true;
            }

            i = (i + 1) % node_count;
            //println!("END: {:?} best cluster {:?} num unstable nodes {:?} num unused clusters {:?}", i, best_cluster, num_unstable_nodes, num_unused_clusters)
        }

        if update {
            clustering.normalize_groups();
        }

        update
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::clustering::VectorGrouping;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn create_test_network() -> Network<f64, f64> {
        // Create a simple network with 4 nodes and 5 edges
        // Node weights: [1.0, 1.0, 1.0, 1.0]
        // Edge structure:
        // 0 -- 1 (weight 2.0)
        // 1 -- 2 (weight 1.0)
        // 2 -- 3 (weight 2.0)
        // 0 -- 2 (weight 0.5)
        // 1 -- 3 (weight 0.5)
        let mut graph = UnGraph::new_undirected();

        // Add nodes
        for _ in 0..4 {
            graph.add_node(1.0);
        }

        // Add edges
        graph.add_edge(0.into(), 1.into(), 2.0);
        graph.add_edge(1.into(), 2.into(), 1.0);
        graph.add_edge(2.into(), 3.into(), 2.0);
        graph.add_edge(0.into(), 2.into(), 0.5);
        graph.add_edge(1.into(), 3.into(), 0.5);

        Network::new_from_graph(graph)
    }

    #[test]
    fn test_basic_clustering() {
        let network = create_test_network();
        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving = StandardLocalMoving::new(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        // Run one iteration
        let updated = local_moving.iterate(&network, &mut clustering, &mut rng);
        assert!(updated, "First iteration should update clustering");

        // Verify some basic properties
        assert!(
            clustering.group_count() <= network.nodes(),
            "Number of clusters should not exceed number of nodes"
        );

        // All nodes should have valid cluster assignments
        for i in 0..network.nodes() {
            assert!(
                clustering.get_group(i) < clustering.group_count(),
                "Node {} has invalid cluster assignment",
                i
            );
        }
    }

    #[test]
    fn test_resolution_parameter() {
        let network = create_test_network();
        let mut rng = StdRng::seed_from_u64(42);

        // Test with high resolution (should favor more clusters)
        let mut clustering_high = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving_high = StandardLocalMoving::new(2.0);
        local_moving_high.iterate(&network, &mut clustering_high, &mut rng);
        let high_clusters = clustering_high.group_count();

        // Test with low resolution (should favor fewer clusters)
        let mut clustering_low = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving_low = StandardLocalMoving::new(0.1);
        local_moving_low.iterate(&network, &mut clustering_low, &mut rng);
        let low_clusters = clustering_low.group_count();

        assert!(
            high_clusters >= low_clusters,
            "Higher resolution should result in same or more clusters"
        );
    }

    #[test]
    fn test_convergence() {
        let network = create_test_network();
        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving = StandardLocalMoving::new(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        // Run multiple iterations until convergence
        let mut iterations = 0;
        let max_iterations = 10;

        while local_moving.iterate(&network, &mut clustering, &mut rng) {
            iterations += 1;
            if iterations >= max_iterations {
                break;
            }
        }

        assert!(
            iterations < max_iterations,
            "Algorithm should converge within reasonable iterations"
        );
    }

    #[test]
    fn test_single_node_network() {
        let mut graph = UnGraph::new_undirected();
        graph.add_node(1.0);
        let network: Network<f64, f64> = Network::new_from_graph(graph);

        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving = StandardLocalMoving::new(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        let updated = local_moving.iterate(&network, &mut clustering, &mut rng);
        assert!(!updated, "Single node network should not update clustering");
        assert_eq!(
            clustering.group_count(),
            1,
            "Single node should be in one cluster"
        );
    }

    #[test]
    fn test_disconnected_components() {
        let mut graph = UnGraph::new_undirected();

        // Create two disconnected components
        // Component 1: nodes 0,1 connected by edge weight 1.0
        // Component 2: nodes 2,3 connected by edge weight 1.0
        for _ in 0..4 {
            graph.add_node(1.0);
        }
        graph.add_edge(0.into(), 1.into(), 1.0);
        graph.add_edge(2.into(), 3.into(), 1.0);

        let network = Network::new_from_graph(graph);
        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut local_moving = StandardLocalMoving::new(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        local_moving.iterate(&network, &mut clustering, &mut rng);

        // Verify that nodes in same component are in same cluster
        assert_eq!(
            clustering.get_group(0),
            clustering.get_group(1),
            "Connected nodes 0 and 1 should be in same cluster"
        );
        assert_eq!(
            clustering.get_group(2),
            clustering.get_group(3),
            "Connected nodes 2 and 3 should be in same cluster"
        );
    }
}
