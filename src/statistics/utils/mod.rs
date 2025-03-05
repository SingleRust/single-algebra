pub fn extract_unique_groups(group_ids: &[usize]) -> Vec<usize> {
    let mut unique_groups = group_ids.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    unique_groups
}

/// Get indices for each group
pub fn get_group_indices(group_ids: &[usize], unique_groups: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let group1 = unique_groups[0];
    let group2 = unique_groups[1];

    let group1_indices = group_ids.iter()
        .enumerate()
        .filter_map(|(i, &g)| if g == group1 { Some(i) } else { None })
        .collect();

    let group2_indices = group_ids.iter()
        .enumerate()
        .filter_map(|(i, &g)| if g == group2 { Some(i) } else { None })
        .collect();

    (group1_indices, group2_indices)
}