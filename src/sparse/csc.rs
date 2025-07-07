use nalgebra_sparse::CscMatrix;
use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use single_utilities::types::Direction;
use std::collections::{HashMap, HashSet};
use std::iter::Sum;
use std::ops::AddAssign;

use crate::sparse::MatrixNTop;
use crate::utils::Normalize;

use super::{
    BatchMatrixMean, BatchMatrixVariance, MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance,
};
use crate::utils::{BatchIdentifier, Log1P};
use anyhow::anyhow;
use single_utilities::traits::{FloatOpsTS, NumericOps};

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

    fn nonzero_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        // Validate mask length
        if mask.len() < self.nrows() {
            return Err(anyhow::anyhow!(
                "Mask length ({}) is less than number of rows ({})",
                mask.len(),
                self.nrows()
            ));
        }

        let mut result = vec![T::zero(); self.ncols()];

        // Process each column
        for col in 0..self.ncols() {
            let col_start = self.col_offsets()[col];
            let col_end = self.col_offsets()[col + 1];

            // Count non-zero elements in this column that are in masked-in rows
            for idx in col_start..col_end {
                let row = self.row_indices()[idx];

                // Skip this row if masked out
                if !mask[row] {
                    continue;
                }

                result[col] += T::one();
            }
        }

        Ok(result)
    }

    fn nonzero_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        // Validate mask length
        if mask.len() < self.ncols() {
            return Err(anyhow::anyhow!(
                "Mask length ({}) is less than number of columns ({})",
                mask.len(),
                self.ncols()
            ));
        }

        let mut result = vec![T::zero(); self.nrows()];

        // Process each column
        for col in 0..self.ncols() {
            // Skip this column if masked out
            if !mask[col] {
                continue;
            }

            let col_start = self.col_offsets()[col];
            let col_end = self.col_offsets()[col + 1];

            // Count non-zero elements in this column
            for idx in col_start..col_end {
                let row = self.row_indices()[idx];
                result[row] += T::one();
            }
        }

        Ok(result)
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
            for (&row_idx, &value) in row_indices[chunk..end].iter().zip(&values[chunk..end]) {
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

    fn sum_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + Sum,
    {
        // Validate mask length
        if mask.len() < self.nrows() {
            return Err(anyhow::anyhow!(
                "Mask length ({}) is less than number of rows ({})",
                mask.len(),
                self.nrows()
            ));
        }

        // Pre-allocate result with zeros
        let mut result = vec![T::zero(); self.ncols()];

        // Process each column
        for col in 0..self.ncols() {
            let col_start = self.col_offsets()[col];
            let col_end = self.col_offsets()[col + 1];

            // Process all non-zero elements in this column
            for idx in col_start..col_end {
                let row = self.row_indices()[idx];

                // Skip this row if masked out
                if !mask[row] {
                    continue;
                }

                let value = T::from(self.values()[idx]).unwrap();
                result[col] += value;
            }
        }

        Ok(result)
    }

    fn sum_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + Sum,
    {
        // Validate mask length
        if mask.len() < self.ncols() {
            return Err(anyhow::anyhow!(
                "Mask length ({}) is less than number of columns ({})",
                mask.len(),
                self.ncols()
            ));
        }

        // Pre-allocate result with zeros
        let mut result = vec![T::zero(); self.nrows()];

        // Process each column
        for col in 0..self.ncols() {
            // Skip this column if masked out
            if !mask[col] {
                continue;
            }

            let col_start = self.col_offsets()[col];
            let col_end = self.col_offsets()[col + 1];

            // Process all non-zero elements in this column
            for idx in col_start..col_end {
                let row = self.row_indices()[idx];
                let value = T::from(self.values()[idx]).unwrap();
                result[row] += value;
            }
        }

        Ok(result)
    }

    fn sum_col_squared<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + Sum,
    {
        let mut result = vec![T::zero(); self.ncols()];

        for (_, col, &value) in self.triplet_iter() {
            let val = T::from(value).unwrap();
            result[col] += val * val;
        }

        Ok(result)
    }

    fn sum_row_squared<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + Sum,
    {
        let mut result = vec![T::zero(); self.ncols()];

        for (row, _, &value) in self.triplet_iter() {
            let val = T::from(value).unwrap();
            result[row] += val * val;
        }

        Ok(result)
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
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
        Self::Item: NumCast,
    {
        let sum: Vec<T> = self.sum_col()?;
        let squared_sums: Vec<T> = self.sum_col_squared()?;
        let mut result = vec![T::zero(); self.ncols()];

        let n = T::from(self.nrows()).unwrap();
        let n_minus_one = n - T::one();
        for col in 0..self.ncols() {
            let mean = sum[col] / n;
            let population_var = squared_sums[col] / n - mean.powi(2);

            if n_minus_one > T::zero() {
                result[col] = population_var * (n / n_minus_one)
            } else {
                result[col] = T::zero();
            }
        }

        Ok(result)
    }

    fn var_row<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
        Self::Item: NumCast,
    {
        let sum: Vec<T> = self.sum_row()?;
        let squared_sums: Vec<T> = self.sum_row_squared()?;
        let mut result = vec![T::zero(); self.nrows()];
        let n = T::from(self.nrows()).unwrap();
        let n_minus_one = n - T::one();
        for row in 0..self.nrows() {
            let mean = sum[row] / n;
            let population_var = squared_sums[row] / n - mean.powi(2);

            if n_minus_one > T::zero() {
                result[row] = population_var * (n / n_minus_one);
            } else {
                result[row] = T::zero();
            }
        }

        Ok(result)
    }

    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
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
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
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

    fn var_col_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + Sum + Send + Sync,
    {
        // Validate mask length
        if mask.len() < self.nrows() {
            return Err(anyhow::anyhow!(
                "Mask length ({}) is less than number of rows ({})",
                mask.len(),
                self.nrows()
            ));
        }

        // Calculate masked sums and counts
        let sum: Vec<T> = self.sum_col_masked(mask)?;
        let count: Vec<I> = self.nonzero_col_masked(mask)?;

        let mut result = vec![T::zero(); self.ncols()];

        // Process each column to calculate variance
        for col in 0..self.ncols() {
            if count[col] > I::zero() {
                let mean = sum[col] / count[col].into();
                let col_start = self.col_offsets()[col];
                let col_end = self.col_offsets()[col + 1];

                // Calculate sum of squared differences for this column (only for masked-in rows)
                let mut sum_sq_diff = T::zero();
                for idx in col_start..col_end {
                    let row = self.row_indices()[idx];

                    // Skip masked out rows
                    if !mask[row] {
                        continue;
                    }

                    let val = T::from(self.values()[idx]).unwrap();
                    let diff = val - mean;
                    sum_sq_diff += diff * diff;
                }

                result[col] = sum_sq_diff / count[col].into();
            }
        }

        Ok(result)
    }

    fn var_row_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + Sum + Send + Sync
    {
        // Validate mask length
        if mask.len() < self.ncols() {
            return Err(anyhow::anyhow!(
                "Mask length ({}) is less than number of columns ({})",
                mask.len(),
                self.ncols()
            ));
        }

        // Calculate masked sums and counts
        let sum: Vec<T> = self.sum_row_masked(mask)?;
        let count: Vec<I> = self.nonzero_row_masked(mask)?;

        let mut result = vec![T::zero(); self.nrows()];
        let mut squared_sums = vec![T::zero(); self.nrows()];

        // Calculate sum of squares for each row (using only masked-in columns)
        for col in 0..self.ncols() {
            // Skip this column if masked out
            if !mask[col] {
                continue;
            }

            let col_start = self.col_offsets()[col];
            let col_end = self.col_offsets()[col + 1];

            for idx in col_start..col_end {
                let row = self.row_indices()[idx];
                let val = T::from(self.values()[idx]).unwrap();
                squared_sums[row] += val * val;
            }
        }

        // Calculate variance for each row
        for row in 0..self.nrows() {
            if count[row] > I::zero() {
                let mean = sum[row] / count[row].into();
                result[row] = squared_sums[row] / count[row].into() - mean * mean;
            }
        }

        Ok(result)
    }
}

impl<M: NumCast + Copy + PartialOrd + NumericOps> MatrixMinMax for CscMatrix<M> {
    type Item = M;

    fn min_max_col<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let mut min: Vec<Item> = vec![Item::max_value(); self.ncols()];
        let mut max: Vec<Item> = vec![Item::min_value(); self.ncols()];

        self.min_max_col_chunk((&mut min, &mut max))?;
        Ok((min, max))
    }

    fn min_max_row<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
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

    fn min_max_row_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
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

impl<T: FloatOpsTS> Normalize<T> for CscMatrix<T> {
    fn normalize<U: FloatOpsTS>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &Direction,
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
            Direction::COLUMN => {
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
            Direction::ROW => {
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

impl<T: FloatOpsTS> Log1P<T> for CscMatrix<T> {
    fn log1p_normalize(&mut self) -> anyhow::Result<()> {
        let values = self.values_mut();
        for val in values.iter_mut() {
            *val = T::one() + *val;
            *val = val.ln();
        }
        Ok(())
    }
}

impl<M> BatchMatrixVariance for CscMatrix<M>
where
    M: NumericOps + NumCast,
    CscMatrix<M>: MatrixSum + MatrixNonZero,
{
    type Item = M;

    fn var_batch_row<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.nrows() {
            return Err(anyhow::anyhow!(
                "Batch vector length ({}) doesn't match matrix row count ({})",
                batches.len(),
                self.nrows()
            ));
        }

        // Create map of row indices to batch identifiers
        let row_to_batch: Vec<&B> = batches.iter().collect();

        // Group rows by batch
        let mut batch_rows: HashMap<B, Vec<usize>> = HashMap::new();
        for (row_idx, &batch) in row_to_batch.iter().enumerate() {
            batch_rows.entry(batch.clone()).or_default().push(row_idx);
        }

        // Calculate variance for each batch
        let mut result: HashMap<B, Vec<T>> = HashMap::new();

        for (batch, row_indices) in batch_rows {
            // Calculate variance for each column across the rows in this batch
            let mut batch_vars = vec![T::zero(); self.ncols()];

            // Collect values for each column in this batch
            let mut col_values: Vec<Vec<T>> = vec![Vec::new(); self.ncols()];

            // Gather all values for each column from the batch's rows
            for col_idx in 0..self.ncols() {
                let col_start = self.col_offsets()[col_idx];
                let col_end = self.col_offsets()[col_idx + 1];

                for j in col_start..col_end {
                    let row = self.row_indices()[j];

                    // Check if this row is in the current batch
                    if row_indices.contains(&row) {
                        let val = T::from(self.values()[j]).unwrap();
                        col_values[col_idx].push(val);
                    }
                }
            }

            // Calculate variance for each column
            for (col_idx, values) in col_values.iter().enumerate() {
                if values.len() > 1 {
                    // Calculate mean
                    let mean = values.iter().copied().sum::<T>() / T::from(values.len()).unwrap();

                    // Calculate sum of squared differences
                    let sum_sq_diff = values
                        .iter()
                        .map(|&val| {
                            let diff = val - mean;
                            diff * diff
                        })
                        .sum::<T>();

                    // Calculate variance
                    batch_vars[col_idx] = sum_sq_diff / T::from(values.len() - 1).unwrap();
                }
                // If values.len() <= 1, variance remains 0
            }

            result.insert(batch, batch_vars);
        }

        Ok(result)
    }

    fn var_batch_col<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.ncols() {
            return Err(anyhow::anyhow!(
                "Batch vector length ({}) doesn't match matrix column count ({})",
                batches.len(),
                self.ncols()
            ));
        }

        // Group column indices by batch
        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (idx, batch) in batches.iter().enumerate() {
            batch_indices.entry(batch.clone()).or_default().push(idx);
        }

        // Calculate variance for each batch
        let mut result: HashMap<B, Vec<T>> = HashMap::new();

        for (batch, indices) in batch_indices {
            // Calculate variance for each row across the columns in this batch
            let mut batch_vars = vec![T::zero(); self.nrows()];
            let mut batch_means = vec![T::zero(); self.nrows()];
            let mut batch_counts = vec![0usize; self.nrows()];
            let mut batch_sum_sq = vec![T::zero(); self.nrows()];

            // First pass: calculate sum and count for each row
            for &col_idx in &indices {
                let col_start = self.col_offsets()[col_idx];
                let col_end = self.col_offsets()[col_idx + 1];

                for j in col_start..col_end {
                    let row = self.row_indices()[j];
                    let val = T::from(self.values()[j]).unwrap();
                    batch_means[row] = batch_means[row] + val;
                    batch_counts[row] += 1;
                }
            }

            // Calculate means
            for (mean, &count) in batch_means.iter_mut().zip(batch_counts.iter()) {
                if count > 0 {
                    *mean = *mean / T::from(count).unwrap();
                }
            }

            // Second pass: calculate sum of squared differences from mean
            for &col_idx in &indices {
                let col_start = self.col_offsets()[col_idx];
                let col_end = self.col_offsets()[col_idx + 1];

                for j in col_start..col_end {
                    let row = self.row_indices()[j];
                    let val = T::from(self.values()[j]).unwrap();
                    let diff = val - batch_means[row];
                    batch_sum_sq[row] = batch_sum_sq[row] + diff * diff;
                }
            }

            // Calculate variance
            for ((var, &count), &sum_sq) in batch_vars
                .iter_mut()
                .zip(batch_counts.iter())
                .zip(batch_sum_sq.iter())
            {
                if count > 1 {
                    *var = sum_sq / T::from(count - 1).unwrap();
                }
            }

            result.insert(batch, batch_vars);
        }

        Ok(result)
    }
}

impl<M: NumericOps + NumCast> BatchMatrixMean for CscMatrix<M> {
    type Item = M;

    fn mean_batch_row<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.ncols() {
            return Err(anyhow::anyhow!(
                "Number of batch identifiers ({}) must match number of columns ({})",
                batches.len(),
                self.ncols()
            ));
        }

        // Group columns by batch
        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (col_idx, batch) in batches.iter().enumerate() {
            batch_indices
                .entry(batch.clone())
                .or_default()
                .push(col_idx);
        }

        // Calculate mean for each batch and row
        let mut result: HashMap<B, Vec<T>> = HashMap::new();
        for (batch, col_indices) in batch_indices {
            let mut batch_sums = vec![T::zero(); self.nrows()];
            let mut batch_counts = vec![0usize; self.nrows()];

            // Calculate sums and counts for each column in this batch
            for &col_idx in &col_indices {
                let col_start = self.col_offsets()[col_idx];
                let col_end = self.col_offsets()[col_idx + 1];

                for idx in col_start..col_end {
                    let row_idx = self.row_indices()[idx];
                    batch_sums[row_idx] += T::from(self.values()[idx]).unwrap();
                    batch_counts[row_idx] += 1;
                }
            }

            // Calculate means
            let mut batch_means = vec![T::zero(); self.nrows()];
            for row_idx in 0..self.nrows() {
                if batch_counts[row_idx] > 0 {
                    batch_means[row_idx] =
                        batch_sums[row_idx] / T::from(batch_counts[row_idx]).unwrap();
                }
            }

            result.insert(batch, batch_means);
        }

        Ok(result)
    }

    fn mean_batch_col<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.nrows() {
            return Err(anyhow::anyhow!(
                "Number of batch identifiers ({}) must match number of rows ({})",
                batches.len(),
                self.nrows()
            ));
        }

        // Group rows by batch
        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (row_idx, batch) in batches.iter().enumerate() {
            batch_indices
                .entry(batch.clone())
                .or_default()
                .push(row_idx);
        }

        // Calculate mean for each batch and column
        let mut result: HashMap<B, Vec<T>> = HashMap::new();
        for (batch, row_indices) in batch_indices {
            let row_indices_set: HashSet<usize> = row_indices.iter().cloned().collect();
            let mut batch_sums = vec![T::zero(); self.ncols()];
            let mut batch_counts = vec![0usize; self.ncols()];

            // Calculate sums and counts for each column
            for col_idx in 0..self.ncols() {
                let col_start = self.col_offsets()[col_idx];
                let col_end = self.col_offsets()[col_idx + 1];

                for idx in col_start..col_end {
                    let row_idx = self.row_indices()[idx];
                    if row_indices_set.contains(&row_idx) {
                        batch_sums[col_idx] += T::from(self.values()[idx]).unwrap();
                        batch_counts[col_idx] += 1;
                    }
                }
            }

            // Calculate means
            let mut batch_means = vec![T::zero(); self.ncols()];
            for col_idx in 0..self.ncols() {
                if batch_counts[col_idx] > 0 {
                    batch_means[col_idx] =
                        batch_sums[col_idx] / T::from(batch_counts[col_idx]).unwrap();
                }
            }

            result.insert(batch, batch_means);
        }

        Ok(result)
    }
}

impl<M: NumericOps + NumCast> MatrixNTop for CscMatrix<M> {
    type Item = M;

    fn sum_row_n_top<T>(&self, n: usize) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + Sum {
        let mut result = vec![T::zero(); self.nrows()];
        
        let mut row_values: Vec<Vec<T>> = vec![Vec::new(); self.nrows()];
        
        for col_idx in 0..self.ncols() {
            let col_start = self.col_offsets()[col_idx];
            let col_end = self.col_offsets()[col_idx + 1];
            
            for idx in col_start..col_end {
                let row_idx = self.row_indices()[idx];
                if let Some(val) = T::from(self.values()[idx]) {
                    row_values[row_idx].push(val);
                }
            }
        }
        
        for (row_idx, mut values) in row_values.into_iter().enumerate() {
            if values.len() <= n {
                result[row_idx] = values.into_iter().sum();
            } else {
                values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                result[row_idx] = values.into_iter().take(n).sum();
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use Direction;

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

    #[test]
    fn test_zero_elements() {
        let coo =
            CooMatrix::try_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![0.0, 0.0f64]).unwrap();
        let mut csc: CscMatrix<f64> = (&coo).into();

        csc.log1p_normalize().unwrap();

        for (_, _, val) in csc.triplet_iter() {
            assert!((val - 0.0).abs() < 1e-10);
        }
    }
}
