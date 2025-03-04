use crate::statistics::{Alternative, TTestType, TestResult};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, FromPrimitive, NumCast};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt::Debug;

pub fn t_test_matrix_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    test_type: TTestType,
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult>>
where
    T: Float + NumCast + FromPrimitive + Send + Sync + Debug,
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

            t_test(&group1_values, &group2_values, test_type, alternative)
        })
        .collect();

    Ok(results)
}

pub fn t_test(x: &[f64], y: &[f64], test_type: TTestType, alternative: Alternative) -> TestResult {
    let nx = x.len();
    let ny = y.len();

    if nx < 2 || ny < 2 {
        return TestResult::new(-1f64, 1.0);
    }

    let mean_x = x.iter().sum::<f64>() / nx as f64;
    let mean_y = y.iter().sum::<f64>() / ny as f64;

    let var_x: f64 = x.iter().map(|&val| (val - mean_x).powi(2)).sum::<f64>() / (nx - 1) as f64;
    let var_y: f64 = y.iter().map(|&val| (val - mean_y).powi(2)).sum::<f64>() / (ny - 1) as f64;

    let (t_stat, df) = match test_type {
        TTestType::Student => {
            // with pooled variance
            let pooled_var =
                ((nx - 1) as f64 * var_x + (ny - 1) as f64 * var_y) / (nx + ny - 2) as f64;
            let std_err = (pooled_var * (1.0 / nx as f64 + 1.0 / ny as f64)).sqrt();
            let t = (mean_x - mean_y) / std_err;
            (t, (nx + ny - 2) as f64)
        }
        TTestType::Welch => {
            let std_err = (var_x / nx as f64 + var_y / ny as f64).sqrt();
            let t = (mean_x - mean_y) / std_err;
            let term1 = var_x / nx as f64;
            let term2 = var_y / ny as f64;

            let num = (term1 + term2).powi(2);
            let denom = term1.powi(2) / (nx - 1) as f64 + term2.powi(2) / (ny - 1) as f64;
            (t, num / denom)
        }
    };

    // Create a Student's t distribution with the appropriate degrees of freedom
    let t_dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => dist,
        Err(_) => return TestResult::new(t_stat, 1.0), // Error case, return p-value of 1.0
    };

    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_stat.abs())),
        Alternative::Less => t_dist.cdf(-t_stat),
        Alternative::Greater => 1.0 - t_dist.cdf(t_stat),
    };

    TestResult::new(t_stat, p_value)
}

pub fn student_t_quantile(p: f64, df: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        panic!("Probability must be between 0 and 1 (exclusive)");
    }
    if df <= 0.0 {
        panic!("Degrees of freedom must be positive");
    }

    // Create a Student's t distribution with the specified degrees of freedom
    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => dist.inverse_cdf(p),
        Err(_) => panic!("Failed to create StudentsT distribution"),
    }
}
