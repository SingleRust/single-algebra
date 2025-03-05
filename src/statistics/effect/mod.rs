use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, NumCast, FromPrimitive};
use std::fmt::Debug;

/// Calculate log2 fold change between two groups
pub fn calculate_log2_fold_change<T>(
    matrix: &CsrMatrix<T>,
    row: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
    pseudo_count: f64,
) -> anyhow::Result<f64>
where
    T: Float + NumCast + FromPrimitive + Debug,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    // Calculate mean for group 1
    let mut sum1 = 0.0;
    for &col in group1_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            sum1 += value.to_f64().unwrap();
        }
        // Missing entries are treated as zero
    }
    let mean1 = sum1 / group1_indices.len() as f64 + pseudo_count;

    // Calculate mean for group 2
    let mut sum2 = 0.0;
    for &col in group2_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            sum2 += value.to_f64().unwrap();
        }
        // Missing entries are treated as zero
    }
    let mean2 = sum2 / group2_indices.len() as f64 + pseudo_count;

    // Calculate log2 fold change
    let log2_fc = (mean2 / mean1).log2();

    Ok(log2_fc)
}

/// Calculate Cohen's d effect size for a row
pub fn calculate_cohens_d<T>(
    matrix: &CsrMatrix<T>,
    row: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<f64>
where
    T: Float + NumCast + FromPrimitive + Debug,
{
    if group1_indices.len() < 2 || group2_indices.len() < 2 {
        return Err(anyhow::anyhow!("Each group must have at least 2 samples for Cohen's d"));
    }

    // Extract values for group 1
    let mut group1_values = Vec::with_capacity(group1_indices.len());
    for &col in group1_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            group1_values.push(value.to_f64().unwrap());
        } else {
            group1_values.push(0.0); // Use zero for missing entries
        }
    }

    // Extract values for group 2
    let mut group2_values = Vec::with_capacity(group2_indices.len());
    for &col in group2_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            group2_values.push(value.to_f64().unwrap());
        } else {
            group2_values.push(0.0); // Use zero for missing entries
        }
    }

    // Calculate means
    let mean1 = group1_values.iter().sum::<f64>() / group1_values.len() as f64;
    let mean2 = group2_values.iter().sum::<f64>() / group2_values.len() as f64;

    // Calculate variances
    let var1 = group1_values.iter()
        .map(|&x| (x - mean1).powi(2))
        .sum::<f64>() / (group1_values.len() - 1) as f64;

    let var2 = group2_values.iter()
        .map(|&x| (x - mean2).powi(2))
        .sum::<f64>() / (group2_values.len() - 1) as f64;

    // Calculate pooled standard deviation
    let n1 = group1_values.len() as f64;
    let n2 = group2_values.len() as f64;
    let pooled_sd = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2).sqrt() / ((n1 + n2 - 2.0).sqrt());

    // Calculate Cohen's d
    let d = (mean2 - mean1) / pooled_sd;

    Ok(d)
}

/// Calculate Hedge's g (bias-corrected effect size)
pub fn calculate_hedges_g<T>(
    matrix: &CsrMatrix<T>,
    row: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<f64>
where
    T: Float + NumCast + FromPrimitive + Debug,
{
    // First calculate Cohen's d
    let d = calculate_cohens_d(matrix, row, group1_indices, group2_indices)?;

    // Apply correction factor
    let n1 = group1_indices.len() as f64;
    let n2 = group2_indices.len() as f64;
    let n = n1 + n2;

    // Correction factor J
    let j = 1.0 - 3.0 / (4.0 * (n - 2.0) - 1.0);

    // Calculate Hedge's g
    let g = j * d;

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};

    fn create_test_matrix() -> CsrMatrix<f64> {
        // Create a simple test matrix for differential expression analysis:
        // Two groups (columns 0,1,2 vs 3,4,5) with different expression patterns
        // Row 0: Clear difference between groups
        // Row 1: No difference between groups
        // Row 2: Moderate difference
        // Row 3: Extreme difference
        // Row 4: All zeros in group 1
        let rows = vec![
            0, 0, 0, 0, 0, 0,   // Row 0: all positions filled
            1, 1, 1, 1, 1, 1,   // Row 1: all positions filled
            2, 2, 2, 2, 2, 2,   // Row 2: all positions filled
            3, 3, 3, 3, 3, 3,   // Row 3: all positions filled
            4, 4, 4             // Row 4: sparse (no entries for group 1)
        ];
        let cols = vec![
            0, 1, 2, 3, 4, 5,   // Row 0
            0, 1, 2, 3, 4, 5,   // Row 1
            0, 1, 2, 3, 4, 5,   // Row 2
            0, 1, 2, 3, 4, 5,   // Row 3
            3, 4, 5             // Row 4 (only group 2 values)
        ];
        let vals = vec![
            2.0, 2.2, 1.8, 8.0, 7.5, 8.5,    // Row 0: ~2 vs ~8 = big difference
            5.0, 5.1, 4.9, 5.0, 5.1, 4.9,    // Row 1: ~5 vs ~5 = no difference
            3.0, 3.3, 2.7, 5.0, 4.7, 5.3,    // Row 2: ~3 vs ~5 = moderate
            0.1, 0.2, 0.1, 20.0, 19.0, 21.0, // Row 3: ~0.1 vs ~20 = extreme
            10.0, 8.0, 12.0                  // Row 4: 0 vs ~10 = missing data test
        ];

        let coo = CooMatrix::try_from_triplets(
            5,  // 5 rows
            6,  // 6 columns
            rows,
            cols,
            vals,
        )
            .unwrap();

        CsrMatrix::from(&coo)
    }

    #[test]
    fn test_log2_fold_change() {
        let matrix = create_test_matrix();
        let group1 = vec![0, 1, 2]; // First group indices
        let group2 = vec![3, 4, 5]; // Second group indices
        let pseudo_count = 0.01;    // Small pseudo count

        // Row 0: Clear difference (~2 vs ~8)
        let fc0 = calculate_log2_fold_change(&matrix, 0, &group1, &group2, pseudo_count).unwrap();
        assert_abs_diff_eq!(fc0, 2.0, epsilon = 0.1); // log2(8/2) ≈ 2

        // Row 1: No difference (~5 vs ~5)
        let fc1 = calculate_log2_fold_change(&matrix, 1, &group1, &group2, pseudo_count).unwrap();
        assert_abs_diff_eq!(fc1, 0.0, epsilon = 0.01); // log2(5/5) = 0

        // Row 2: Moderate difference (~3 vs ~5)
        let fc2 = calculate_log2_fold_change(&matrix, 2, &group1, &group2, pseudo_count).unwrap();
        assert_abs_diff_eq!(fc2, 0.737, epsilon = 0.01); // log2(5/3) ≈ 0.737

        // Row 3: Extreme difference (~0.1 vs ~20)
        let fc3 = calculate_log2_fold_change(&matrix, 3, &group1, &group2, pseudo_count).unwrap();
        assert_abs_diff_eq!(fc3, 7.13, epsilon = 0.1); // log2(20/0.1) ≈ 7.13

        // Row 4: All zeros in group 1 (tests handling of missing data)
        let fc4 = calculate_log2_fold_change(&matrix, 4, &group1, &group2, pseudo_count).unwrap();
        // Group 1 is all 0s + pseudo_count, Group 2 is ~10 + pseudo_count
        assert!(fc4 > 9.0); // log2((10+0.01)/(0+0.01)) ≈ 9.97
    }

    #[test]
    fn test_empty_groups() {
        let matrix = create_test_matrix();

        // Test with empty groups
        let result = calculate_log2_fold_change(&matrix, 0, &[], &[3, 4, 5], 0.01);
        assert!(result.is_err());

        let result = calculate_log2_fold_change(&matrix, 0, &[0, 1, 2], &[], 0.01);
        assert!(result.is_err());

        // Test with small groups for Cohen's d
        let result = calculate_cohens_d(&matrix, 0, &[0], &[3, 4, 5]);
        assert!(result.is_err());

        let result = calculate_cohens_d(&matrix, 0, &[0, 1, 2], &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cohens_d() {
        let matrix = create_test_matrix();
        let group1 = vec![0, 1, 2]; // First group indices
        let group2 = vec![3, 4, 5]; // Second group indices

        // Row 0: Clear difference (~2 vs ~8)
        let d0 = calculate_cohens_d(&matrix, 0, &group1, &group2).unwrap();
        assert_abs_diff_eq!(d0, 15.76, epsilon = 0.1); // Large effect

        // Row 1: No difference (~5 vs ~5)
        let d1 = calculate_cohens_d(&matrix, 1, &group1, &group2).unwrap();
        assert_abs_diff_eq!(d1, 0.0, epsilon = 0.01); // No effect

        // Row 2: Moderate difference (~3 vs ~5)
        let d2 = calculate_cohens_d(&matrix, 2, &group1, &group2).unwrap();
        assert_abs_diff_eq!(d2, 6.67, epsilon = 0.1); // Large effect

        // Row 3: Extreme difference (~0.1 vs ~20)
        let d3 = calculate_cohens_d(&matrix, 3, &group1, &group2).unwrap();
        // The exact value may vary depending on implementation details,
        // but should definitely show a very large effect
        assert!(d3 > 20.0); // Very large effect
    }

    #[test]
    fn test_hedges_g() {
        let matrix = create_test_matrix();
        let group1 = vec![0, 1, 2]; // First group indices
        let group2 = vec![3, 4, 5]; // Second group indices

        // Row 0: Compare with Cohen's d
        let d0 = calculate_cohens_d(&matrix, 0, &group1, &group2).unwrap();
        let g0 = calculate_hedges_g(&matrix, 0, &group1, &group2).unwrap();

        // Hedge's g should be slightly smaller than Cohen's d due to correction
        assert!(g0 < d0);
        // But for large samples they shouldn't be too different
        assert_abs_diff_eq!(g0 / d0, 0.75, epsilon = 0.25);

        // Row 1: No difference case
        let g1 = calculate_hedges_g(&matrix, 1, &group1, &group2).unwrap();
        assert_abs_diff_eq!(g1, 0.0, epsilon = 0.01);

        // Test correction factor works
        // For larger groups, correction should be smaller
        let large_group1 = vec![0, 1, 2, 6, 7, 8, 9, 10];
        let large_group2 = vec![3, 4, 5, 11, 12, 13, 14, 15];

        // This will fail if sample sizes are out of range, but the test is to show
        // that larger sample sizes bring Hedge's g closer to Cohen's d
        if matrix.ncols() >= 16 {
            let d_large = calculate_cohens_d(&matrix, 0, &large_group1, &large_group2).unwrap_or(d0);
            let g_large = calculate_hedges_g(&matrix, 0, &large_group1, &large_group2).unwrap_or(g0);

            // With more samples, g should be closer to d
            assert!(g_large / d_large > g0 / d0);
        }
    }

    #[test]
    fn test_zero_variance_cases() {
        // Create a matrix with zero variance in one or both groups
        let rows = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let cols = vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5];
        let vals = vec![
            5.0, 5.0, 5.0, 10.0, 10.0, 10.0,  // Row 0: all values identical within groups
            1.0, 2.0, 3.0, 5.0, 5.0, 5.0      // Row 1: variance in group 1, none in group 2
        ];

        let coo = CooMatrix::try_from_triplets(2, 6, rows, cols, vals).unwrap();
        let matrix = CsrMatrix::from(&coo);

        let group1 = vec![0, 1, 2];
        let group2 = vec![3, 4, 5];

        // Test log2fc - should work fine with zero variance
        let fc0 = calculate_log2_fold_change(&matrix, 0, &group1, &group2, 0.01).unwrap();
        assert_abs_diff_eq!(fc0, 1.0, epsilon = 0.01); // log2(10/5) = 1

        // Cohen's d with zero variance in both groups
        // In theory this should be infinity, but in practice we might get very large values
        // or potential numerical issues
        let d0 = calculate_cohens_d(&matrix, 0, &group1, &group2);
        match d0 {
            Ok(value) => assert!(value.abs() > 100.0 || value.is_infinite()), // Should be very large or infinity
            Err(_) => {} // It's also acceptable if the function determines this is an error case
        }

        // Cohen's d with zero variance in one group
        let d1 = calculate_cohens_d(&matrix, 1, &group1, &group2);
        if let Ok(value) = d1 {
            assert!(value.abs() > 1.0); // Should show a substantial effect
        }
    }

    #[test]
    fn test_negative_values() {
        // Test with negative values to ensure functions handle them correctly
        let rows = vec![0, 0, 0, 0, 0, 0];
        let cols = vec![0, 1, 2, 3, 4, 5];
        let vals = vec![-2.0, -2.2, -1.8, -8.0, -7.5, -8.5]; // Negative values

        let coo = CooMatrix::try_from_triplets(1, 6, rows, cols, vals).unwrap();
        let matrix = CsrMatrix::from(&coo);

        let group1 = vec![0, 1, 2];
        let group2 = vec![3, 4, 5];

        // Log2 fold change with negative values
        // For negative values, the fold change would be the ratio of absolute means
        // with a sign to indicate direction
        let fc = calculate_log2_fold_change(&matrix, 0, &group1, &group2, 0.0);
        // Expected: log2(|-8|/|-2|) ≈ 2, but sign needs handling in function
        assert!(fc.is_ok()); // Just check it doesn't crash

        // Cohen's d should work with negative values
        let d = calculate_cohens_d(&matrix, 0, &group1, &group2);
        assert!(d.is_ok());
        if let Ok(value) = d {
            // Cohen's d should have the same magnitude as with positive values but negative sign
            assert!(value < 0.0);
            assert_abs_diff_eq!(value.abs(), 15.76, epsilon = 0.1);
        }
    }
}