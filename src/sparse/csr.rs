use std::ops::{Add, AddAssign};

use crate::{
    utils::{Normalize, NumericNormalize},
    NumericOps,
};
use anyhow::{anyhow, Ok};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, NumCast, One, PrimInt, Unsigned, Zero};

use super::{MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance};

impl<M: NumericOps> MatrixNonZero for CsrMatrix<M> {
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let mut result = vec![T::zero(); self.ncols()];
        for &col_index in self.col_indices() {
            result[col_index] += T::one();
        }
        Ok(result)
    }

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let data = self
            .row_offsets()
            .windows(2)
            .map(|window| {
                let diff = window[1]
                    .checked_sub(window[0])
                    .ok_or_else(|| anyhow!("Subtraction overflow"))
                    .expect("Subtraction overflow");
                T::from(diff)
                    .ok_or_else(|| anyhow!("Failed to convert to target type"))
                    .expect("Failed to convert to targat type")
            })
            .collect();
        Ok(data)
    }

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for &col_index in self.col_indices() {
            if col_index < reference.len() {
                reference[col_index] += T::one();
            }
        }
        Ok(())
    }

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for (i, window) in self.row_offsets().windows(2).enumerate() {
            let count = window[1]
                .checked_sub(window[0])
                .ok_or_else(|| anyhow!("Subtraction overflow"))?;
            let count_transformed =
                T::from(count).ok_or_else(|| anyhow!("Failed to convert to target type"))?;
            if i < reference.len() {
                reference[i] += count_transformed;
            }
        }
        Ok(())
    }
}

impl<M: NumericOps> MatrixSum for CsrMatrix<M> {
    type Item = M;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        let mut result = vec![T::zero(); self.ncols()];
        let col_indices = self.col_indices();
        let values = self.values();

        // Process values directly in chunks to better utilize cache
        const CHUNK_SIZE: usize = 256;
        for chunk_start in (0..values.len()).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(values.len());

            // Direct accumulation without temporary storage
            for (&col_idx, &value) in col_indices[chunk_start..chunk_end]
                .iter()
                .zip(&values[chunk_start..chunk_end])
            {
                result[col_idx] += T::from(value).unwrap();
            }
        }

        Ok(result)
    }

    fn sum_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        let mut result = vec![T::zero(); self.nrows()];
        let values = self.values();
        let row_offsets = self.row_offsets();

        // Since each row is likely to have moderate length,
        // we can process them directly without chunking
        for row in 0..self.nrows() {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];

            // Direct accumulation in original type
            let mut sum = M::zero();
            for &val in &values[start..end] {
                sum = sum + val;
            }

            // Single conversion per row
            result[row] = T::from(sum).unwrap();
        }

        Ok(result)
    }

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        for (&col_index, &value) in self.col_indices().iter().zip(self.values().iter()) {
            if col_index < reference.len() {
                reference[col_index] += T::from(value).unwrap();
            }
        }
        Ok(())
    }

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        for (row, row_vec) in self.row_iter().enumerate() {
            reference[row] = row_vec.values().iter().map(|&v| T::from(v).unwrap()).sum();
        }
        Ok(())
    }
}

impl<M> MatrixVariance for CsrMatrix<M>
where
    M: NumericOps + NumCast,
    CsrMatrix<M>: MatrixSum + MatrixNonZero,
{
    type Item = M;

    fn var_col<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
    {
        let sum: Vec<T> = self.sum_col()?;
        let count: Vec<I> = self.nonzero_col()?;
        let mut result = vec![T::zero(); self.ncols()];
        let mut squared_sums = vec![T::zero(); self.ncols()];

        // First pass: calculate squared sums for each column
        for (value, &col) in self.values().iter().zip(self.col_indices().iter()) {
            let val = T::from(*value).unwrap();
            squared_sums[col] += val * val;
        }

        // Second pass: calculate variances
        for col in 0..self.ncols() {
            if count[col] > I::zero() {
                let mean = sum[col] / count[col].into();
                result[col] = squared_sums[col] / count[col].into() - mean * mean;
            }
        }

        Ok(result)
    }

    fn var_row<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        let sum: Vec<T> = self.sum_row()?;
        let count: Vec<I> = self.nonzero_row()?;
        let mut result = vec![T::zero(); self.nrows()];

        // Calculate variance for each row
        for row in 0..self.nrows() {
            if count[row] > I::zero() {
                let row_start = self.row_offsets()[row];
                let row_end = self
                    .row_offsets()
                    .get(row + 1)
                    .copied()
                    .unwrap_or(self.values().len());
                let mean = sum[row] / count[row].into();

                // Calculate sum of squared differences for this row
                let variance = self.values()[row_start..row_end]
                    .iter()
                    .filter_map(|&v| T::from(v))
                    .map(|v| {
                        let diff = v - mean;
                        diff * diff
                    })
                    .sum::<T>()
                    / count[row].into();

                result[row] = variance;
            }
        }

        Ok(result)
    }

    /// Calculate column-wise variance and store results in the provided slice
    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        let ncols = self.ncols();
        if reference.len() != ncols {
            return Err(anyhow::anyhow!(
                "Reference slice length {} does not match number of columns {}",
                reference.len(),
                ncols
            ));
        }

        let sum: Vec<T> = self.sum_col()?;
        let count: Vec<I> = self.nonzero_col()?;
        let mut squared_sums = vec![T::zero(); ncols];

        // First pass: calculate squared sums for each column
        for (value, &col) in self.values().iter().zip(self.col_indices().iter()) {
            if let Some(val) = T::from(*value) {
                squared_sums[col] += val * val;
            }
        }

        // Second pass: calculate variances in-place
        for col in 0..ncols {
            reference[col] = if count[col] > I::zero() {
                let mean = sum[col] / count[col].into();
                squared_sums[col] / count[col].into() - mean * mean
            } else {
                T::zero()
            };
        }

        Ok(())
    }

    fn var_row_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        let nrows = self.nrows();
        if reference.len() != nrows {
            return Err(anyhow::anyhow!(
                "Reference slice length {} does not match number of rows {}",
                reference.len(),
                nrows
            ));
        }

        let sum: Vec<T> = self.sum_row()?;
        let count: Vec<I> = self.nonzero_row()?;

        // Calculate variance for each row in-place
        for row in 0..nrows {
            let row_start = self.row_offsets()[row];
            let row_end = self
                .row_offsets()
                .get(row + 1)
                .copied()
                .unwrap_or(self.values().len());

            reference[row] = if count[row] > I::zero() {
                let mean = sum[row] / count[row].into();

                // Calculate variance for this row
                self.values()[row_start..row_end]
                    .iter()
                    .filter_map(|&v| T::from(v))
                    .map(|v| {
                        let diff = v - mean;
                        diff * diff
                    })
                    .sum::<T>()
                    / count[row].into()
            } else {
                T::zero()
            };
        }

        Ok(())
    }
}

impl<M: NumCast + Copy + PartialOrd + NumericOps> MatrixMinMax for CsrMatrix<M> {
    type Item = M;

    fn min_max_col<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps,
    {
        let mut min: Vec<Item> = vec![Item::max_value(); self.ncols()];
        let mut max: Vec<Item> = vec![Item::min_value(); self.ncols()];

        self.min_max_col_chunk((&mut min, &mut max))?;
        Ok((min, max))
    }

    fn min_max_row<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps,
    {
        let mut min: Vec<Item> = vec![Item::max_value(); self.nrows()];
        let mut max: Vec<Item> = vec![Item::min_value(); self.nrows()];

        self.min_max_row_chunk((&mut min, &mut max))?;
        Ok((min, max))
    }

    fn min_max_col_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps,
    {
        let (min_vals, max_vals) = reference;

        let row_offsets = self.row_offsets();
        let col_indices = self.col_indices();
        let values = self.values();

        // For CsrMatrix, we need to traverse row by row
        for row in 0..self.nrows() {
            let start_idx = row_offsets[row];
            let end_idx = row_offsets[row + 1];

            // Process each non-zero element in this row
            for idx in start_idx..end_idx {
                let col = col_indices[idx];
                let value = Item::from(values[idx]).unwrap();

                // Update column minimum
                if value < min_vals[col] {
                    min_vals[col] = value;
                }

                // Update column maximum
                if value > max_vals[col] {
                    max_vals[col] = value;
                }
            }
        }

        Ok(())
    }

    fn min_max_row_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps,
    {
        let (min_vals, max_vals) = reference;

        let row_offsets = self.row_offsets();
        let values = self.values();

        (0..self.nrows()).for_each(|row| {
            let start_idx = row_offsets[row];
            let end_idx = row_offsets[row + 1];

            if start_idx < end_idx {
                let first_value = Item::from(values[start_idx]).unwrap();
                let mut row_min = first_value;
                let mut row_max = first_value;

                for &value in &values[start_idx..end_idx] {
                    let value_cast = Item::from(value).unwrap();

                    if value_cast < row_min {
                        row_min = value_cast;
                    }

                    if value_cast > row_max {
                        row_max = value_cast;
                    }
                }

                min_vals[row] = row_min;
                max_vals[row] = row_max;
            }
        });

        Ok(())
    }
}

impl<T: NumericNormalize> Normalize<T> for CsrMatrix<T> {
    fn normalize<U: NumericNormalize>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &crate::Direction,
    ) -> anyhow::Result<()> {
        // Pre-compute scaling factors to avoid repeated divisions
        let scaling_factors: Vec<U> = sums
            .iter()
            .map(|&sum| {
                if sum > U::zero() {
                    target / sum
                } else {
                    U::zero()
                }
            })
            .collect();

        match direction {
            crate::Direction::COLUMN => {
                // Get an iterator over column indices before mutating values
                let col_indices = self.col_indices().to_vec();
                let values = self.values_mut();

                // Process in one pass through the data
                for (val, &col) in values.iter_mut().zip(col_indices.iter()) {
                    let scale = scaling_factors[col];
                    if scale > U::zero() {
                        *val = T::from(U::from(*val).unwrap() * scale).unwrap();
                    }
                }
            }
            crate::Direction::ROW => {
                // Copy row offsets to avoid borrowing conflicts
                let row_offsets = self.row_offsets().to_vec();
                let nrows = self.nrows();
                let values = self.values_mut();

                // Process each row sequentially
                for row in 0..nrows {
                    let scale = scaling_factors[row];
                    if scale > U::zero() {
                        // Process all values in this row
                        let start = row_offsets[row];
                        let end = row_offsets[row + 1];
                        for val in &mut values[start..end] {
                            *val = T::from(U::from(*val).unwrap() * scale).unwrap();
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Direction;

    use super::*;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    fn create_test_matrix() -> CscMatrix<f64> {
        // Create a 4x3 sparse matrix with the following structure:
        // [1 0 2]
        // [0 0 0]
        // [3 4 0]
        // [0 5 6]

        // First create a COO matrix and then convert to CSC
        let mut coo = CooMatrix::new(4, 3);
        coo.push(0, 0, 1.0); // First column
        coo.push(2, 0, 3.0);

        coo.push(2, 1, 4.0); // Second column
        coo.push(3, 1, 5.0);

        coo.push(0, 2, 2.0); // Third column
        coo.push(3, 2, 6.0);

        CscMatrix::from(&coo)
    }

    #[test]
    fn test_nonzero_col() {
        let matrix = create_test_matrix();
        let result: Vec<u32> = matrix.nonzero_col().unwrap();

        // Expected number of nonzero elements in each column
        assert_eq!(result, vec![2, 2, 2]);
    }

    #[test]
    fn test_nonzero_row() {
        let matrix = create_test_matrix();
        let result: Vec<u32> = matrix.nonzero_row().unwrap();

        // Expected number of nonzero elements in each row
        assert_eq!(result, vec![2, 0, 2, 2]);
    }

    #[test]
    fn test_nonzero_col_chunk() {
        let matrix = create_test_matrix();
        let mut reference = vec![0u32; 4];
        matrix.nonzero_col_chunk(&mut reference).unwrap();

        // Only first 3 elements should be modified (matrix has 3 columns)
        assert_eq!(reference, vec![2, 2, 2, 0]);
    }

    #[test]
    fn test_nonzero_row_chunk() {
        let matrix = create_test_matrix();
        let mut reference = vec![0u32; 3];
        matrix.nonzero_row_chunk(&mut reference).unwrap();

        // Should only count nonzeros for rows within reference length
        assert_eq!(reference, vec![2, 0, 2]);
    }

    #[test]
    fn test_empty_matrix() {
        let matrix: CscMatrix<f64> = CscMatrix::zeros(0, 0);

        // Test empty matrix handling
        assert!(matrix.nonzero_col::<u32>().unwrap().is_empty());
        assert!(matrix.nonzero_row::<u32>().unwrap().is_empty());

        let mut empty_ref: Vec<u32> = Vec::new();
        assert!(matrix.nonzero_col_chunk(&mut empty_ref).is_ok());
        assert!(matrix.nonzero_row_chunk(&mut empty_ref).is_ok());
    }

    #[test]
    fn test_different_integer_types() {
        let matrix = create_test_matrix();

        // Test with u8
        let result_u8: Vec<u8> = matrix.nonzero_col().unwrap();
        assert_eq!(result_u8, vec![2, 2, 2]);

        // Test with u64
        let result_u64: Vec<u64> = matrix.nonzero_col().unwrap();
        assert_eq!(result_u64, vec![2, 2, 2]);
    }

    #[test]
    fn test_large_sparse_matrix() {
        // Create a larger sparse matrix to test potential overflow conditions
        let mut coo = CooMatrix::new(1000, 1000);

        // Add some sparse data
        for i in 0..999 {
            coo.push(i, i, 1.0);
            coo.push(i + 1, i, 1.0);
        }

        let matrix = CscMatrix::from(&coo);
        let result: Vec<u32> = matrix.nonzero_col().unwrap();
        assert_eq!(result.len(), 1000);

        // Most columns should have 2 nonzero elements
        assert_eq!(result[500], 2);
    }

    #[test]
    fn test_chunk_smaller_than_matrix() {
        let matrix = create_test_matrix();

        // Test with smaller reference slices
        let mut col_ref = vec![0u32; 2];
        matrix.nonzero_col_chunk(&mut col_ref).unwrap();
        assert_eq!(col_ref, vec![2, 2]);

        let mut row_ref = vec![0u32; 2];
        matrix.nonzero_row_chunk(&mut row_ref).unwrap();
        assert_eq!(row_ref, vec![2, 0]);
    }

    #[test]
    fn test_zero_matrix() {
        // Test a non-empty matrix with all zero elements
        let matrix: CscMatrix<f64> = CscMatrix::zeros(5, 4);

        let col_result: Vec<u32> = matrix.nonzero_col().unwrap();
        assert_eq!(col_result, vec![0, 0, 0, 0]);

        let row_result: Vec<u32> = matrix.nonzero_row().unwrap();
        assert_eq!(row_result, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_csr_normalize() {
        // Create a simple CSR matrix from COO format
        let coo = CooMatrix::try_from_triplets(
            3,
            3,
            vec![0, 0, 1, 1, 2],           // row indices
            vec![0, 1, 1, 2, 2],           // col indices
            vec![2.0, 3.0, 4.0, 1.0, 2.0], // values
        )
        .unwrap();
        let mut csr: CsrMatrix<f64> = (&coo).into();

        // Test column normalization
        let col_sums = vec![2.0, 7.0, 3.0]; // Sum of each column
        let target = 1.0;
        csr.normalize(&col_sums, target, &Direction::COLUMN)
            .unwrap();

        // Verify results
        let expected_values = [1.0, 3.0 / 7.0, 4.0 / 7.0, 1.0 / 3.0, 2.0 / 3.0];
        for ((_, _, val), expected) in csr.triplet_iter().zip(expected_values.iter()) {
            assert!((val - expected).abs() < 1e-10);
        }

        // Test row normalization
        let mut csr: CsrMatrix<f64> = (&coo).into(); // Reset matrix
        let row_sums = vec![5.0, 5.0, 2.0]; // Sum of each row
        csr.normalize(&row_sums, target, &Direction::ROW).unwrap();

        // Verify results
        let expected_values = [0.4, 0.6, 0.8, 0.2, 1.0];
        for ((_, _, val), expected) in csr.triplet_iter().zip(expected_values.iter()) {
            assert!((val - expected).abs() < 1e-10);
        }
    }
}
