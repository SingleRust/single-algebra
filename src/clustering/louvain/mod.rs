// See https://www.nature.com/articles/s41598-019-41695-z#Sec2 and for the original publication: https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008 - https://en.wikipedia.org/wiki/Louvain_method & https://github.com/graphext/louvain-rs/tree/master
// Copyright 2018 Juan Morales (crispamares@gmail.com)
// Repository: https://github.com/graphext/louvain-rs/tree/master
// Licensed under the MIT License.
use std::collections::HashSet;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::iter::Sum;
use std::ops::MulAssign;
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;
use crate::local_moving::standard::StandardLocalMoving;
use crate::network::{Graph, Network};
use crate::network::clustering::{NetworkGrouping, VectorGrouping};

pub const DEF_RES: f64 = 1.0;

pub struct Louvain<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign, {
    rng: ChaCha20Rng,
    local_moving: StandardLocalMoving<T>
}

impl<T> Louvain<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign, {
    pub fn new(resolution: T, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_default();

        Louvain {
            rng: ChaCha20Rng::seed_from_u64(seed),
            local_moving: StandardLocalMoving::new(resolution)
        }
    }

    pub fn iterate_one_level<N, E>(
        &mut self,
        network: &Network<N, E>,
        clustering: &mut VectorGrouping
    ) -> bool
    where
    N: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign,
    E: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign, {
        self.local_moving.iterate(network, clustering, &mut self.rng)
    }

    pub fn iterate<N, E>(
        &mut self,
        network: &Network<N, E>,
        clustering: &mut VectorGrouping
    ) -> bool
    where
        N: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign,
        E: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + MulAssign {
        let mut update = self.local_moving.iterate(network, clustering, &mut self.rng);

        if clustering.group_count() == network.nodes() {
            return update;
        }

        let reduced_network = network.create_reduced_network(clustering);
        let mut reduced_clustering = VectorGrouping::create_isolated(reduced_network.nodes());
        update |= self.iterate(&reduced_network, &mut reduced_clustering);
        clustering.merge(&reduced_clustering);
        update
    }

    pub fn build_network<I>(
        n_nodes: usize,
        n_edges: usize,
        adjacency: I
    ) -> Network<f64, f64>
    where
        I: Iterator<Item = (u32, u32)> {
        let mut graph = Graph::with_capacity(n_nodes, n_edges);
        let mut node_indices = Vec::with_capacity(n_nodes);

        for _ in 0..n_nodes {
            node_indices.push(graph.add_node(1.0));
        }

        let mut seen = vec![HashSet::<u32>::new(); n_nodes];
        let mut node_weights = vec![0.0; n_nodes];

        for (i, j) in adjacency {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let i_ = i as usize;
            let j_ = j as usize;

            if seen[i_].insert(j) {
                graph.add_edge(
                    node_indices[i_],
                    node_indices[j_],
                    1.0
                );
                node_weights[j_] += 1.0;
                node_weights[i_] += 1.0;
            }
        }

        for &i in &node_indices {
            *graph.node_weight_mut(i).unwrap() = node_weights[i.index()];
        }

        Network::new_from_graph(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::NodeIndex;

    fn create_test_network() -> Network<f64, f64> {
        let mut graph = Graph::new_undirected();

        // Add 5 nodes
        for _ in 0..5 {
            graph.add_node(1.0);
        }

        // Add edges to create two communities
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(1), 1.0);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(2), 1.0);
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(2), 1.0);
        graph.add_edge(NodeIndex::new(3), NodeIndex::new(4), 1.0);

        Network::new_from_graph(graph)
    }

    #[test]
    fn test_louvain_clustering() {
        let network = create_test_network();
        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut louvain: Louvain<f64> = Louvain::new(DEF_RES.into(), Some(42));

        assert!(louvain.iterate(&network, &mut clustering));

        // Should identify two communities
        assert!(clustering.group_count() == 2);

        // Nodes 0,1,2 should be in same cluster
        let cluster1 = clustering.get_group(0);
        assert_eq!(clustering.get_group(1), cluster1);
        assert_eq!(clustering.get_group(2), cluster1);

        // Nodes 3,4 should be in different cluster
        let cluster2 = clustering.get_group(3);
        assert_eq!(clustering.get_group(4), cluster2);
        assert_ne!(cluster1, cluster2);
    }

    #[test]
    fn test_build_network() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4)];
        let network = Louvain::<f64>::build_network(5, edges.len(), edges.into_iter());

        assert_eq!(network.nodes(), 5);
        assert_eq!(network.graph.edge_count(), 4);

        // Check node weights (should equal degree)
        for i in 0..5 {
            let weight = network.weight(i);
            let expected = match i {
                0..=2 => 2.0,  // Nodes in triangle
                3..=4 => 1.0,  // Nodes in single edge
                _ => unreachable!(),
            };
            assert_eq!(weight, expected);
        }
    }
}