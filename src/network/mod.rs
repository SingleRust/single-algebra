// https://en.wikipedia.org/wiki/Louvain_method & https://github.com/graphext/louvain-rs/tree/master
// Copyright 2018 Juan Morales (crispamares@gmail.com)
// Repository: https://github.com/graphext/louvain-rs/tree/master
// Licensed under the MIT License.
use crate::network::clustering::{NetworkGrouping};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use petgraph::data::DataMap;
use petgraph::graph::UnGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoEdges, NodeCount};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use std::collections::HashMap;

pub mod clustering;

// for now
pub type Graph<N, E> = UnGraph<N, E>;

pub struct Network<N, E> {
    pub graph: Graph<N, E>,
}

pub struct NeighborAndWeightIterator<'a, N: 'a, E: 'a> {
    edge_iter: petgraph::graph::Edges<'a, E, petgraph::Undirected>,
    home_node: usize,
    _phantom: std::marker::PhantomData<&'a N>,
}

impl<'a, N, E> Iterator for NeighborAndWeightIterator<'a, N, E>
where
    E: Copy,
{
    type Item = (usize, E);

    fn next(&mut self) -> Option<Self::Item> {
        self.edge_iter.next().map(|edge_ref| {
            let neighbor = if edge_ref.source().index() == self.home_node {
                edge_ref.target().index()
            } else {
                edge_ref.source().index()
            };
            (neighbor, *edge_ref.weight())
        })
    }
}

impl<N, E> Network<N, E>
where
    N: Float + FromPrimitive + ToPrimitive + Send + Sync,
    E: Float + FromPrimitive + ToPrimitive + Send + Sync + std::iter::Sum + std::ops::MulAssign,
{
    pub fn new() -> Self {
        Network {
            graph: Graph::new_undirected(),
        }
    }

    pub fn new_from_graph(graph: Graph<N, E>) -> Self {
        Network { graph }
    }

    pub fn nodes(&self) -> usize {
        self.graph.node_count()
    }

    pub fn weight(&self, node: usize) -> N {
        *self
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(node))
            .unwrap()
    }

    pub fn neighbors(&self, node: usize) -> NeighborAndWeightIterator<'_, N, E> {
        NeighborAndWeightIterator {
            edge_iter: self.graph.edges(petgraph::graph::NodeIndex::new(node)),
            home_node: node,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_total_node_weight(&self) -> N {
        self.graph
            .node_weights()
            .fold(N::zero(), |sum, node| sum + *node)
    }

    pub fn get_total_edge_weight(&self) -> E {
        self.graph
            .edge_weights()
            .fold(E::zero(), |sum, edge| sum + *edge)
    }

    pub fn get_total_edge_weight_par(&self) -> E {
        let weights: Vec<_> = self.graph.edge_weights().collect();
        weights
            .par_chunks(256)
            .map(|chunk| chunk.iter().fold(E::zero(), |acc, &weight| acc + *weight))
            .sum()
    }

    pub fn get_total_edge_weight_per_node(&self, result: &mut Vec<E>) {
        result.clear();
        result.extend((0..self.nodes()).map(|i| {
            self.graph
                .edges(petgraph::graph::NodeIndex::new(i))
                .fold(E::zero(), |acc, edge| acc + *edge.weight())
        }));
    }

    pub fn create_reduced_network<T: NetworkGrouping>(&self, grouping: &T) -> Self {
        let mut cluster_g =
            Graph::with_capacity(grouping.group_count(), grouping.group_count() * 2);
        for _ in 0..grouping.group_count() {
            cluster_g.add_node(N::zero());
        }

        for node_idx in self.graph.node_indices() {
            let group = grouping.get_group(node_idx.index());
            let group_node = petgraph::graph::NodeIndex::new(group);
            let current_weight = self.graph.node_weight(node_idx).unwrap();
            let group_weight = cluster_g.node_weight_mut(group_node).unwrap();
            *group_weight = *group_weight + *current_weight;
        }

        let mut edge_memo = HashMap::new();

        for edge in self.graph.edge_references() {
            let g1 = grouping.get_group(edge.source().index());
            let g2 = grouping.get_group(edge.target().index());

            if g1 == g2 {
                continue;
            }

            let (min_g, max_g) = if g1 < g2 { (g1, g2) } else { (g2, g1) };
            *edge_memo.entry((min_g, max_g)).or_insert(E::zero()) *= *edge.weight();
        }

        for (&(g1, g2), &weight) in edge_memo.iter() {
            cluster_g.add_edge(
                petgraph::graph::NodeIndex::new(g1),
                petgraph::graph::NodeIndex::new(g2),
                weight,
            );
        }
        Network { graph: cluster_g }
    }

    pub fn create_subnetworks<T: NetworkGrouping>(&self, grouping: &T) -> Vec<Self> {
        let mut graphs = vec![Graph::new_undirected(); grouping.group_count()];
        let mut new_id_map = vec![0; self.nodes()];
        let mut counts = vec![0; grouping.group_count()];

        // Create nodes in each subgraph
        for node_idx in self.graph.node_indices() {
            let node = node_idx.index();
            let group = grouping.get_group(node);

            let new_id = counts[group];
            new_id_map[node] = new_id;
            counts[group] += 1;

            graphs[group].add_node(*self.graph.node_weight(node_idx).unwrap());
        }

        // Add edges to appropriate subgraphs
        for edge in self.graph.edge_references() {
            let n1 = edge.source().index();
            let g1 = grouping.get_group(n1);

            let n2 = edge.target().index();
            let g2 = grouping.get_group(n2);

            if g1 == g2 {
                graphs[g1].add_edge(
                    petgraph::graph::NodeIndex::new(new_id_map[n1]),
                    petgraph::graph::NodeIndex::new(new_id_map[n2]),
                    *edge.weight(),
                );
            }
        }

        graphs
            .into_iter()
            .map(Network::new_from_graph)
            .collect::<Vec<_>>()
    }

    fn create_subnetwork_from_group<T: NetworkGrouping>(&self, grouping: &T, group: usize) -> Self {
        let mut subgraph = Graph::new_undirected();
        let mut old_to_new = HashMap::new();

        // Add nodes that belong to the specified group
        for node_idx in self.graph.node_indices() {
            if grouping.get_group(node_idx.index()) == group {
                let new_idx = subgraph.add_node(*self.graph.node_weight(node_idx).unwrap());
                old_to_new.insert(node_idx, new_idx);
            }
        }

        // Add edges between nodes in the group
        for edge in self.graph.edge_references() {
            let source = edge.source();
            let target = edge.target();

            if let (Some(&new_source), Some(&new_target)) =
                (old_to_new.get(&source), old_to_new.get(&target))
            {
                subgraph.add_edge(new_source, new_target, *edge.weight());
            }
        }

        Network { graph: subgraph }
    }
}

impl<N, E> Default for Network<N, E>
where
    N: Float + FromPrimitive + ToPrimitive + Send + Sync,
    E: Float + FromPrimitive + ToPrimitive + Send + Sync + std::iter::Sum + std::ops::MulAssign,
{
    fn default() -> Self {
        Self::new()
    }
}

pub fn network_from_matrix<T>(csr_matrix: &CsrMatrix<T>) -> Network<T, T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + std::iter::Sum + std::ops::MulAssign,
{
    let n_nodes = csr_matrix.ncols();
    let mut graph = UnGraph::with_capacity(n_nodes, csr_matrix.nnz());
    let mut node_indices = Vec::with_capacity(n_nodes);

    // Add nodes with weights based on column sums
    let mut node_weights = vec![T::zero(); n_nodes];
    for (col, col_vec) in csr_matrix.row_iter().enumerate() {
        let weight = T::from_f64(
            col_vec
                .values()
                .iter()
                .fold(0.0, |acc, &x| acc + x.to_f64().unwrap()),
        )
        .unwrap();
        node_weights[col] = weight;
        node_indices.push(graph.add_node(weight));
    }

    // Add edges from the matrix structure
    for (row, col, &weight) in csr_matrix.triplet_iter() {
        if row <= col {
            // Only add each edge once for undirected graph
            graph.add_edge(node_indices[row], node_indices[col], weight);
        }
    }

    Network::new_from_graph(graph)
}
