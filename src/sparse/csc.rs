use nalgebra_sparse::CscMatrix;
use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use std::ops::Add;
use std::ops::AddAssign;

use crate::{
    utils::{Normalize, NumericNormalize},
    NumericOps,
};

use super::{MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance};
use anyhow::anyhow;

impl<M: NumericOps + NumCast> MatrixNonZero for CscMatrix<M> {
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let data = self
            .col_offsets()
            .windows(2)
            .map(|window| {
                let diff = window[1]
                    .checked_sub(window[0])
                    .ok_or_else(|| anyhow!("Subtraction overflow"))
                    .expect("Subtraction overflow");
                T::from(diff)
                    .ok_or_else(|| anyhow!("Failed to convert to target type"))
                    .expect("Failed to convert to a target type")
            })
            .collect();
        Ok(data)
    }

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let mut result = vec![T::zero(); self.nrows()];
        for &row_index in self.row_indices() {
            result[row_index] += T::one();
        }
        Ok(result)
    }

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for (i, window) in self.col_offsets().windows(2).enumerate() {
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

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for &row_index in self.row_indices() {
            if row_index < reference.len() {
                reference[row_index] += T::one();
            }
        }
        Ok(())
    }
}

impl<M> MatrixSum for CscMatrix<M>
where
    M: NumericOps + NumCast,
{
    type Item = M;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
    {
        // Pre-allocate result with zeros
        let mut result = vec![T::zero(); self.ncols()];
        let values = self.values();
        let col_offsets = self.col_offsets();

        // Process columns in chunks for better cache utilization
        const CHUNK_SIZE: usize = 1024;
        for chunk_start in (0..self.ncols()).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(self.ncols());

            // Pre-fetch next chunk's offsets
            for col in chunk_start..chunk_end {
                let start = col_offsets[col];
                let end = col_offsets[col + 1];

                // Direct accumulation without iterator adaptors
                let mut sum = T::zero();
                let col_values = &values[start..end];

                // Manual SIMD-like accumulation
                for values in col_values.chunks_exact(4) {
                    sum += T::from(values[0]).unwrap();
                    sum += T::from(values[1]).unwrap();
                    sum += T::from(values[2]).unwrap();
                    sum += T::from(values[3]).unwrap();
                }

                // Handle remaining values
                for &val in col_values.chunks_exact(4).remainder() {
                    sum += T::from(val).unwrap();
                }

                result[col] = sum;
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
        let row_indices = self.row_indices();
        let values = self.values();

        // Process in larger chunks for better vectorization
        const CHUNK_SIZE: usize = 1024;
        for chunk in (0..values.len()).step_by(CHUNK_SIZE) {
            let end = (chunk + CHUNK_SIZE).min(values.len());

            // Process chunk of indices and values together
            for (&row_idx, &value) in row_indices[chunk..end]
                .iter()
                .zip(&values[chunk..end])
            {
                // Accumulate directly without type conversion until necessary
                result[row_idx] += T::from(value).unwrap();
            }
        }
        Ok(result)
    }

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        for (col, col_vec) in self.col_iter().enumerate() {
            if col < reference.len() {
                reference[col] += col_vec.values().iter().map(|&v| T::from(v).unwrap()).sum();
            }
        }
        Ok(())
    }

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        for (&row_index, &value) in self.row_indices().iter().zip(self.values().iter()) {
            if row_index < reference.len() {
                reference[row_index] += T::from(value).unwrap();
            }
        }
        Ok(())
    }
}

impl<M> MatrixVariance for CscMatrix<M>
where
    CscMatrix<M>: MatrixSum + MatrixNonZero,
    M: NumericOps + NumCast,
{
    type Item = M;

    fn var_col<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        let sum: Vec<T> = self.sum_row()?;
        let count: Vec<I> = self.nonzero_row()?;

        let mut result = vec![T::zero(); self.ncols()];
        for (col, col_vec) in self.col_iter().enumerate() {
            let mean = sum[col] / count[col].into();
            let variance = col_vec
                .values()
                .iter()
                .map(|&v| {
                    let diff = T::from(v).unwrap() - mean;
                    diff * diff
                })
                .sum::<T>()
                / count[col].into();
            result[col] = variance;
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
        let mut squared_sum = vec![T::zero(); self.nrows()];
        for (&row_index, &value) in self.row_indices().iter().zip(self.values().iter()) {
            let val = T::from(value).unwrap();
            squared_sum[row_index] += val * val;
        }
        for row in 0..self.nrows() {
            if count[row] > I::zero() {
                let mean = sum[row] / count[row].into();
                result[row] = squared_sum[row] / count[row].into() - mean * mean;
            }
        }
        Ok(result)
    }

    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        // Validate input slice length matches number of columns
        if reference.len() != self.ncols() {
            return Err(anyhow::anyhow!(
                "Reference slice length {} does not match number of columns {}",
                reference.len(),
                self.ncols()
            ));
        }

        let sum: Vec<T> = self.sum_row()?;
        let count: Vec<I> = self.nonzero_row()?;

        // Calculate variance for each column in-place
        for (col, col_vec) in self.col_iter().enumerate() {
            if count[col] > I::zero() {
                let mean = sum[col] / count[col].into();
                let variance = col_vec
                    .values()
                    .iter()
                    .map(|&v| {
                        let diff = T::from(v).unwrap() - mean;
                        diff * diff
                    })
                    .sum::<T>()
                    / count[col].into();
                reference[col] = variance;
            } else {
                reference[col] = T::zero();
            }
        }
        Ok(())
    }

    fn var_row_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        Self::Item: NumCast,
    {
        // Validate input slice length matches number of rows
        if reference.len() != self.nrows() {
            return Err(anyhow::anyhow!(
                "Reference slice length {} does not match number of rows {}",
                reference.len(),
                self.nrows()
            ));
        }

        let sum: Vec<T> = self.sum_row()?;
        let count: Vec<I> = self.nonzero_row()?;

        // Initialize squared sum vector
        let mut squared_sum = vec![T::zero(); self.nrows()];

        // Calculate squared sums
        for (&row_index, &value) in self.row_indices().iter().zip(self.values().iter()) {
            let val = T::from(value).unwrap();
            squared_sum[row_index] += val * val;
        }

        // Calculate variance for each row in-place
        for row in 0..self.nrows() {
            if count[row] > I::zero() {
                let mean = sum[row] / count[row].into();
                reference[row] = squared_sum[row] / count[row].into() - mean * mean;
            } else {
                reference[row] = T::zero();
            }
        }
        Ok(())
    }
}

impl<M: NumCast + Copy + PartialOrd + NumericOps> MatrixMinMax for CscMatrix<M> {
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

    fn min_max_col_chunk<Item>(
        &self,
        reference: (&mut [Item], &mut [Item]),
    ) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps,
    {
        let (min_vals, max_vals) = reference;

        let col_offsets = self.col_offsets();
        let values = self.values();

        (0..self.ncols()).for_each(|col| {
            let start_idx = col_offsets[col];
            let end_idx = col_offsets[col + 1];

            if start_idx < end_idx {
                let first_value = Item::from(values[start_idx]).unwrap();
                let mut col_min = first_value;
                let mut col_max = first_value;

                for &value in &values[start_idx..end_idx] {
                    let value_cast = Item::from(value).unwrap();

                    if value_cast < col_min {
                        col_min = value_cast;
                    }

                    if value_cast > col_max {
                        col_max = value_cast;
                    }
                }
                min_vals[col] = col_min;
                max_vals[col] = col_max;
            }
        });

        Ok(())
    }

    fn min_max_row_chunk<Item>(
        &self,
        reference: (&mut [Item], &mut [Item]),
    ) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps,
    {
        let (min_vals, max_vals) = reference;

        let col_offsets = self.col_offsets();
        let row_indices = self.row_indices();
        let values = self.values();

        for col in 0..self.ncols() {
            let start_idx = col_offsets[col];
            let end_idx = col_offsets[col + 1];

            for idx in start_idx..end_idx {
                let row = row_indices[idx];
                let value: Item = Item::from(values[idx]).unwrap();

                if value < min_vals[row] {
                    min_vals[row] = value;
                }

                if value > max_vals[row] {
                    max_vals[row] = value;
                }
            }
        }
        Ok(())
    }
}

impl<T: NumericNormalize> Normalize<T> for CscMatrix<T> {
    fn normalize<U: NumericNormalize>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &crate::Direction,
    ) -> anyhow::Result<()> {
        // Pre-compute scaling factors to avoid repeated divisions
        let scaling_factors: Vec<U> = sums
            .iter()
            .map(|&sum| if sum > U::zero() { target / sum } else { U::zero() })
            .collect();

        match direction {
            crate::Direction::COLUMN => {
                // Copy column offsets to avoid borrowing conflicts
                let col_offsets = self.col_offsets().to_vec();
                let ncols = self.ncols();
                let values = self.values_mut();


                // Process each column sequentially
                for col in 0..ncols {
                    let scale = scaling_factors[col];
                    if scale > U::zero() {
                        // Process all values in this column
                        let start = col_offsets[col];
                        let end = col_offsets[col + 1];
                        for val in &mut values[start..end] {
                            *val = T::from(U::from(*val).unwrap() * scale).unwrap();
                        }
                    }
                }
            }
            crate::Direction::ROW => {
                // Get row indices before mutating values
                let row_indices = self.row_indices().to_vec();
                let values = self.values_mut();

                // Process in one pass through the data
                for (val, &row) in values.iter_mut().zip(row_indices.iter()) {
                    let scale = scaling_factors[row];
                    if scale > U::zero() {
                        *val = T::from(U::from(*val).unwrap() * scale).unwrap();
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
        use nalgebra_sparse::CooMatrix;

        // Create initial matrix with triplets for:
        // [1.0, 0.0, 2.0]
        // [0.0, 3.0, 0.0]
        // [4.0, 0.0, 5.0]
        let mut coo = CooMatrix::try_from_triplets(
            3,
            3,
            vec![0, 2],     // row indices
            vec![0, 0],     // column indices
            vec![1.0, 4.0], // values
        )
        .unwrap();

        // Push remaining entries
        coo.push(1, 1, 3.0);
        coo.push(0, 2, 2.0);
        coo.push(2, 2, 5.0);

        // Convert to CscMatrix
        CscMatrix::from(&coo)
    }

    #[test]
    fn test_matrix_nonzero() {
        let matrix = create_test_matrix();

        // Test nonzero_col
        let col_counts: Vec<u32> = matrix.nonzero_col().unwrap();
        assert_eq!(
            col_counts,
            vec![2, 1, 2],
            "Column nonzero counts should match"
        );

        // Test nonzero_row
        let row_counts: Vec<u32> = matrix.nonzero_row().unwrap();
        assert_eq!(row_counts, vec![2, 1, 2], "Row nonzero counts should match");

        // Test nonzero_col_chunk
        let mut col_chunk = vec![0u32; 3];
        matrix.nonzero_col_chunk(&mut col_chunk).unwrap();
        assert_eq!(col_chunk, vec![2, 1, 2], "Column chunk counts should match");

        // Test nonzero_row_chunk
        let mut row_chunk = vec![0u32; 3];
        matrix.nonzero_row_chunk(&mut row_chunk).unwrap();
        assert_eq!(row_chunk, vec![2, 1, 2], "Row chunk counts should match");
    }

    #[test]
    fn test_matrix_sum() {
        let matrix = create_test_matrix();

        // Test sum_col
        let col_sums: Vec<f64> = matrix.sum_col().unwrap();
        assert_eq!(col_sums, vec![5.0, 3.0, 7.0], "Column sums should match");

        // Test sum_row
        let row_sums: Vec<f64> = matrix.sum_row().unwrap();
        assert_eq!(row_sums, vec![3.0, 3.0, 9.0], "Row sums should match");

        // Test sum_col_chunk
        let mut col_chunk = vec![0.0; 3];
        matrix.sum_col_chunk(&mut col_chunk).unwrap();
        assert_eq!(
            col_chunk,
            vec![5.0, 3.0, 7.0],
            "Column chunk sums should match"
        );

        // Test sum_row_chunk
        let mut row_chunk = vec![0.0; 3];
        matrix.sum_row_chunk(&mut row_chunk).unwrap();
        assert_eq!(
            row_chunk,
            vec![3.0, 3.0, 9.0],
            "Row chunk sums should match"
        );
    }

    #[test]
    fn test_matrix_variance() {
        let matrix = create_test_matrix();

        // Test var_col
        let col_vars: Vec<f64> = matrix.var_col::<u32, f64>().unwrap();
        assert!(
            col_vars.iter().all(|&x| x >= 0.0),
            "Variances should be non-negative"
        );

        // Test var_row
        let row_vars: Vec<f64> = matrix.var_row::<u32, f64>().unwrap();
        assert!(
            row_vars.iter().all(|&x| x >= 0.0),
            "Variances should be non-negative"
        );

        // Test var_col_chunk
        let mut col_chunk = vec![0.0; matrix.ncols()];
        matrix.var_col_chunk::<u32, f64>(&mut col_chunk).unwrap();
        assert!(
            col_chunk.iter().all(|&x| x >= 0.0),
            "Chunk variances should be non-negative"
        );

        // Test var_row_chunk
        let mut row_chunk = vec![0.0; matrix.nrows()];
        matrix.var_row_chunk::<u32, f64>(&mut row_chunk).unwrap();
        assert!(
            row_chunk.iter().all(|&x| x >= 0.0),
            "Chunk variances should be non-negative"
        );
    }

    #[test]
    fn test_matrix_min_max() {
        let matrix = create_test_matrix();

        // Test min_max_col
        let (col_mins, col_maxs): (Vec<f64>, Vec<f64>) = matrix.min_max_col().unwrap();
        assert_eq!(
            col_mins.len(),
            3,
            "Should have correct number of column minimums"
        );
        assert_eq!(
            col_maxs.len(),
            3,
            "Should have correct number of column maximums"
        );
        assert!(
            col_mins[0] <= col_maxs[0],
            "First column min should be <= max"
        );
        assert_eq!(col_mins[0], 1.0, "First column minimum should be 1.0");
        assert_eq!(col_maxs[0], 4.0, "First column maximum should be 4.0");

        // Test min_max_row
        let (row_mins, row_maxs): (Vec<f64>, Vec<f64>) = matrix.min_max_row().unwrap();
        assert_eq!(
            row_mins.len(),
            3,
            "Should have correct number of row minimums"
        );
        assert_eq!(
            row_maxs.len(),
            3,
            "Should have correct number of row maximums"
        );
        assert!(row_mins[2] <= row_maxs[2], "Last row min should be <= max");
        assert_eq!(row_mins[2], 4.0, "Last row minimum should be 4.0");
        assert_eq!(row_maxs[2], 5.0, "Last row maximum should be 5.0");

        // Test min_max_col_chunk and min_max_row_chunk
        let mut col_mins = vec![f64::MAX; 3];
        let mut col_maxs = vec![f64::MIN; 3];
        matrix
            .min_max_col_chunk((&mut col_mins, &mut col_maxs))
            .unwrap();
        assert!(
            col_mins
                .iter()
                .zip(col_maxs.iter())
                .all(|(min, max)| min <= max),
            "All column minimums should be <= maximums"
        );

        let mut row_mins = vec![f64::MAX; 3];
        let mut row_maxs = vec![f64::MIN; 3];
        matrix
            .min_max_row_chunk((&mut row_mins, &mut row_maxs))
            .unwrap();
        assert!(
            row_mins
                .iter()
                .zip(row_maxs.iter())
                .all(|(min, max)| min <= max),
            "All row minimums should be <= maximums"
        );
    }

    #[test]
    fn test_csc_normalization() {
        // Create test matrix:
        // 2 0 3
        // 0 4 0
        // 1 0 5
        let coo = CooMatrix::try_from_triplets(
            3,
            3,
            vec![0, 1, 2, 0, 2],
            vec![0, 1, 0, 2, 2],
            vec![2.0, 4.0, 1.0, 3.0, 5.0_f64],
        )
        .unwrap();
        let mut csc: CscMatrix<f64> = (&coo).into();

        // Test column normalization
        let col_sums = vec![3.0, 4.0, 8.0];
        let target = 1.0;
        csc.normalize(&col_sums, target, &Direction::COLUMN)
            .unwrap();

        // Verify each column sums to target
        let (cols, _rows, vals) = csc.csc_data();
        let mut col_sums_after = [0.0; 3];
        for i in 0..cols.len() - 1 {
            for j in cols[i]..cols[i + 1] {
                col_sums_after[i] += vals[j];
            }
            assert!((col_sums_after[i] - target).abs() < 1e-10);
        }

        // Test row normalization
        let mut csc: CscMatrix<f64> = (&coo).into();
        let row_sums = vec![5.0, 4.0, 6.0];
        csc.normalize(&row_sums, target, &Direction::ROW).unwrap();

        // Verify each row sums to target
        let mut row_sums_after = vec![0.0; 3];
        for (r, _, v) in csc.triplet_iter() {
            row_sums_after[r] += v;
        }
        for sum in row_sums_after {
            assert!((sum - target).abs() < 1e-10);
        }
    }
}


