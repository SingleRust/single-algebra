// https://github.com/graphext/louvain-rs/tree/master
// Copyright 2018 Juan Morales (crispamares@gmail.com)
// Repository: https://github.com/graphext/louvain-rs/tree/master
// Licensed under the MIT License.
use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A trait for managing groups of nodes in a network.
/// Optimized for performance in large-scale network analysis.
pub trait NetworkGrouping: Debug + Send + Sync {
    /// Creates a new grouping where each node starts in its own group
    fn create_isolated(node_count: usize) -> Self;

    /// Creates a new grouping where all nodes start in the same group
    fn create_unified(node_count: usize) -> Self;

    /// Creates a grouping from a predefined set of group assignments
    fn from_assignments(assignments: &[usize]) -> Self;

    /// Gets all nodes belonging to each group
    fn get_group_members(&self) -> Vec<Vec<usize>>;

    /// Gets the group ID for a given node
    fn get_group(&self, node: usize) -> usize;

    /// Gets the group IDs for a range of nodes
    fn get_groups_range(&self, range: std::ops::Range<usize>) -> &[usize];

    /// Sets the group for a given node
    fn set_group(&mut self, node: usize, group: usize);

    /// Sets groups for multiple nodes at once
    fn set_groups_bulk(&mut self, nodes: &[usize], group: usize);

    /// Gets the total number of nodes
    fn node_count(&self) -> usize;

    /// Gets the total number of groups
    fn group_count(&self) -> usize;

    /// Renumbers groups to eliminate gaps in group IDs
    fn normalize_groups(&mut self);

    /// Merges groups based on a higher-level grouping scheme
    fn merge<G: NetworkGrouping>(&mut self, arrangement: &G) {
        // Use parallel iterator for large datasets
        if self.node_count() > 1000 {
            let assignments: Vec<_> = (0..self.node_count())
                .into_par_iter()
                .map(|node| {
                    let current_group = self.get_group(node);
                    arrangement.get_group(current_group)
                })
                .collect();

            for (node, &group) in assignments.iter().enumerate() {
                self.set_group(node, group);
            }
        } else {
            // Use regular iterator for small datasets to avoid parallel overhead
            for node in 0..self.node_count() {
                let current_group = self.get_group(node);
                let new_group = arrangement.get_group(current_group);
                self.set_group(node, new_group);
            }
        }
        self.normalize_groups();
    }

    fn clear(&mut self) {
        for i in 0..self.node_count() {
            self.set_group(i, 0);
        }

        self.normalize_groups();
    }
}

/// An optimized implementation of NetworkGrouping using a vector
#[derive(Debug, Clone)]
pub struct VectorGrouping {
    assignments: Vec<usize>,
    group_count: usize,
    // Cache for frequently accessed group sizes
    group_sizes: Vec<usize>,
    needs_size_update: bool,
}

impl Default for VectorGrouping {
    fn default() -> Self {
        Self {
            assignments: Vec::new(),
            group_count: 0,
            group_sizes: Vec::new(),
            needs_size_update: false,
        }
    }
}

impl VectorGrouping {
    /// Updates the cached group sizes
    #[inline]
    fn update_group_sizes(&mut self) {
        if !self.needs_size_update {
            return;
        }

        self.group_sizes = vec![0; self.group_count];
        for &group in &self.assignments {
            self.group_sizes[group] += 1;
        }
        self.needs_size_update = false;
    }

    /// Gets the size of a specific group
    #[inline]
    pub fn get_group_size(&mut self, group: usize) -> usize {
        self.update_group_sizes();
        self.group_sizes[group]
    }

    /// Returns an iterator over groups with their sizes
    pub fn iter_group_sizes(&mut self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.update_group_sizes();
        self.group_sizes.iter().copied().enumerate()
    }
}

impl NetworkGrouping for VectorGrouping {
    fn create_isolated(node_count: usize) -> Self {
        let assignments = (0..node_count).collect();
        Self {
            assignments,
            group_count: node_count,
            group_sizes: vec![1; node_count],
            needs_size_update: false,
        }
    }

    fn create_unified(node_count: usize) -> Self {
        Self {
            assignments: vec![0; node_count],
            group_count: usize::from(node_count > 0),
            group_sizes: vec![node_count],
            needs_size_update: false,
        }
    }

    fn from_assignments(input: &[usize]) -> Self {
        if input.len() > 1000 {
            // Use parallel iterator for large inputs
            let max_group = AtomicUsize::new(0);

            let assignments: Vec<_> = input
                .par_iter()
                .map(|&group| {
                    // Update max_group atomically
                    let current_max = max_group.load(Ordering::Relaxed);
                    if group > current_max {
                        max_group.fetch_max(group, Ordering::Relaxed);
                    }
                    group
                })
                .collect();

            let mut grouping = Self {
                assignments,
                group_count: max_group.load(Ordering::Relaxed) + 1,
                group_sizes: Vec::new(),
                needs_size_update: true,
            };
            grouping.normalize_groups();
            grouping
        } else {
            // Use sequential approach for small inputs
            let mut max_group = 0;
            let assignments = input
                .iter()
                .map(|&group| {
                    max_group = max_group.max(group);
                    group
                })
                .collect();

            let mut grouping = Self {
                assignments,
                group_count: max_group + 1,
                group_sizes: Vec::new(),
                needs_size_update: true,
            };
            grouping.normalize_groups();
            grouping
        }
    }

    fn get_group_members(&self) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new(); self.group_count];

        // Pre-allocate space based on average group size
        let avg_size = self.assignments.len() / self.group_count;
        for group in groups.iter_mut() {
            group.reserve(avg_size);
        }

        for (node, &group) in self.assignments.iter().enumerate() {
            groups[group].push(node);
        }
        groups
    }

    #[inline]
    fn get_group(&self, node: usize) -> usize {
        self.assignments[node]
    }

    #[inline]
    fn get_groups_range(&self, range: std::ops::Range<usize>) -> &[usize] {
        &self.assignments[range]
    }

    #[inline]
    fn set_group(&mut self, node: usize, group: usize) {
        if self.assignments[node] != group {
            self.assignments[node] = group;
            self.group_count = self.group_count.max(group + 1);
            self.needs_size_update = true;
        }
    }

    fn set_groups_bulk(&mut self, nodes: &[usize], group: usize) {
        for &node in nodes {
            self.assignments[node] = group;
        }
        self.group_count = self.group_count.max(group + 1);
        self.needs_size_update = true;
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.assignments.len()
    }

    #[inline]
    fn group_count(&self) -> usize {
        self.group_count
    }

    fn normalize_groups(&mut self) {
        // Use parallel processing for large datasets
        if self.assignments.len() > 1000 {
            // Create a vector of atomic counters for thread-safe counting
            let sizes: Vec<_> = (0..self.group_count)
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect();

            // Count group sizes in parallel
            self.assignments.par_iter().for_each(|&group| {
                sizes[group].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });

            // Convert atomic counts back to regular numbers
            let sizes: Vec<_> = sizes
                .iter()
                .map(|atomic| atomic.load(std::sync::atomic::Ordering::Relaxed))
                .collect();

            // Create new ID mapping
            let mut new_ids = Vec::with_capacity(self.group_count);
            let mut next_id = 0;

            for &size in &sizes {
                if size == 0 {
                    new_ids.push(usize::MAX);
                } else {
                    new_ids.push(next_id);
                    next_id += 1;
                }
            }

            // Update assignments in parallel
            self.assignments.par_iter_mut().for_each(|group| {
                let new_id = new_ids[*group];
                debug_assert!(new_id != usize::MAX, "Invalid group assignment");
                *group = new_id;
            });

            self.group_count = next_id;
        } else {
            // Use original sequential approach for small datasets
            let mut sizes = vec![0; self.group_count];
            for &group in &self.assignments {
                sizes[group] += 1;
            }

            let mut new_ids = Vec::with_capacity(self.group_count);
            let mut next_id = 0;

            for size in sizes {
                if size == 0 {
                    new_ids.push(usize::MAX);
                } else {
                    new_ids.push(next_id);
                    next_id += 1;
                }
            }

            for group in self.assignments.iter_mut() {
                let new_id = new_ids[*group];
                debug_assert!(new_id != usize::MAX, "Invalid group assignment");
                *group = new_id;
            }

            self.group_count = next_id;
        }

        self.needs_size_update = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_groups_parallel() {
        // Create a grouping with gaps
        let assignments: Vec<_> = (0..2000).map(|i| i % 3 * 2).collect(); // Creates groups [0, 2, 4]
        let mut grouping = VectorGrouping::from_assignments(&assignments);

        grouping.normalize_groups();

        // Should be normalized to [0, 1, 2]
        assert_eq!(grouping.group_count(), 3);
        for i in 0..2000 {
            assert_eq!(grouping.get_group(i), (i % 3));
        }
    }

    #[test]
    fn test_normalize_groups_sequential() {
        // Test with small dataset
        let mut grouping = VectorGrouping::from_assignments(&[0, 2, 4, 2, 0]);

        grouping.normalize_groups();

        assert_eq!(grouping.group_count(), 3);
        let expected = vec![0, 1, 2, 1, 0];
        assert_eq!(grouping.assignments, expected);
    }

    #[test]
    fn test_normalize_groups_empty_groups() {
        // Test handling of empty groups
        let mut grouping = VectorGrouping::from_assignments(&[0, 3, 3, 0]); // Group 1 and 2 are empty

        grouping.normalize_groups();

        assert_eq!(grouping.group_count(), 2);
        let expected = vec![0, 1, 1, 0];
        assert_eq!(grouping.assignments, expected);
    }

    #[test]
    fn test_parallel_operations() {
        let size = 10_000;
        let mut grouping = VectorGrouping::create_isolated(size);

        // Test bulk operations
        let nodes: Vec<_> = (0..1000).collect();
        grouping.set_groups_bulk(&nodes, 1);

        for &node in &nodes {
            assert_eq!(grouping.get_group(node), 1);
        }
    }

    #[test]
    fn test_group_sizes() {
        let mut grouping = VectorGrouping::from_assignments(&[0, 1, 1, 2, 2, 2]);
        assert_eq!(grouping.get_group_size(0), 1);
        assert_eq!(grouping.get_group_size(1), 2);
        assert_eq!(grouping.get_group_size(2), 3);
    }

    #[test]
    fn test_range_access() {
        let grouping = VectorGrouping::from_assignments(&[0, 1, 2, 3, 4]);
        assert_eq!(grouping.get_groups_range(1..4), &[1, 2, 3]);
    }

    // Include previous inference...
    #[test]
    fn test_create_isolated() {
        let grouping = VectorGrouping::create_isolated(3);
        assert_eq!(grouping.node_count(), 3);
        assert_eq!(grouping.group_count(), 3);
        assert_eq!(grouping.assignments, vec![0, 1, 2]);
    }

    #[test]
    fn test_create_unified() {
        let grouping = VectorGrouping::create_unified(3);
        assert_eq!(grouping.node_count(), 3);
        assert_eq!(grouping.group_count(), 1);
        assert_eq!(grouping.assignments, vec![0, 0, 0]);
    }

    #[test]
    fn test_group_members() {
        let grouping = VectorGrouping::from_assignments(&[0, 1, 1, 2]);
        let members = grouping.get_group_members();
        assert_eq!(members[0], vec![0]);
        assert_eq!(members[1], vec![1, 2]);
        assert_eq!(members[2], vec![3]);
    }

    #[test]
    fn test_merge_groups() {
        let mut base = VectorGrouping::from_assignments(&[0, 1, 1, 2]);
        let higher = VectorGrouping::from_assignments(&[0, 0, 1]);

        base.merge(&higher);
        assert_eq!(base.assignments, vec![0, 0, 0, 1]);
        assert_eq!(base.group_count(), 2);
    }
}
