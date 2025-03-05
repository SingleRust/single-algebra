use crate::statistics::{Alternative, TestResult};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, FromPrimitive, NumCast};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use statrs::distribution::{ContinuousCDF, Normal};
use std::cmp::Ordering;

pub fn mann_whitney_matrix_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult>>
where
    T: Float + NumCast + FromPrimitive + Send + Sync + std::fmt::Debug,
    f64: std::convert::From<T>,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    let nrows = matrix.nrows();

    let results: Vec<_> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let mut group1_values: Vec<f64> = Vec::with_capacity(group1_indices.len());
            let mut group2_values: Vec<f64> = Vec::with_capacity(group2_indices.len());

            for &col in group1_indices {
                if let Some(entry) = matrix.get_entry(row, col) {
                    let value = entry.into_value();
                    group1_values.push(value.into());
                }
            }

            for &col in group2_indices {
                if let Some(entry) = matrix.get_entry(row, col) {
                    let value = entry.into_value();
                    group2_values.push(value.into());
                }
            }
            mann_whitney(&group1_values, &group2_values, alternative)
        })
        .collect();

    Ok(results)
}

pub fn mann_whitney(x: &[f64], y: &[f64], alternative: Alternative) -> TestResult {
    let nx = x.len();
    let ny = y.len();

    if nx == 0 || ny == 0 {
        return TestResult::new(f64::NAN, 1.0); // Insufficient data
    }

    // Combine samples and assign group labels (0 for x, 1 for y)
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(nx + ny);
    combined.extend(x.iter().map(|&v| (v, 0)));
    combined.extend(y.iter().map(|&v| (v, 1)));

    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Assign ranks (with ties averaged)
    let mut ranks = vec![0.0; nx + ny];
    let mut i = 0;
    while i < combined.len() {
        let val = combined[i].0;
        let mut j = i + 1;

        // Find tied values
        while j < combined.len() && combined[j].0 == val {
            j += 1;
        }

        // Assign average rank to ties
        let rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[k] = rank;
        }

        i = j;
    }

    // Calculate rank sum for group X
    let mut rank_sum_x = 0.0;
    for i in 0..combined.len() {
        if combined[i].1 == 0 {
            rank_sum_x += ranks[i];
        }
    }

    let u_x = rank_sum_x - (nx * (nx + 1)) as f64 / 2.0;
    let u_y = (nx * ny) as f64 - u_x;

    let u = match alternative {
        Alternative::TwoSided => u_x.min(u_y),
        Alternative::Less => u_x,
        Alternative::Greater => u_y,
    };

    let mean_u = (nx * ny) as f64 / 2.0;
    let var_u = (nx * ny * (ny + nx + 1)) as f64 / 12.0;

    let correction = 0.5;

    let z = match alternative {
        Alternative::TwoSided => {
            let z_score = (u.max(mean_u) - mean_u - correction) / var_u.sqrt();
            z_score.abs()
        }
        Alternative::Less => (u_x - mean_u + correction) / var_u.sqrt(),
        Alternative::Greater => (u_y - mean_u + correction) / var_u.sqrt(),
    };

    let normal = Normal::new(0.0, 1.0).unwrap();

    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z)),
        _ => 1.0 - normal.cdf(z),
    };

    let effect_size = z / ((nx + ny) as f64).sqrt();

    // Standard error of U
    let standard_error = var_u.sqrt();

    TestResult::with_effect_size(u, p_value, effect_size)
        .with_standard_error(standard_error)
        .with_metadata("z_score", z)
        .with_metadata("mean_u", mean_u)
        .with_metadata("var_u", var_u)
        .with_metadata("nx", nx as f64)
        .with_metadata("ny", ny as f64)
}
