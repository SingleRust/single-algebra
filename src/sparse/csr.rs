use std::collections::HashMap;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Add, AddAssign};

use super::{
    BatchMatrixMean, BatchMatrixVariance, MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance,
};
use crate::sparse::MatrixNTop;
use crate::utils::Normalize;
use crate::utils::{BatchIdentifier, Log1P};
use anyhow::{anyhow, Ok};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, NumCast, One, PrimInt, Unsigned, Zero};
use single_utilities::traits::{FloatOpsTS, NumericOps};
use single_utilities::types::Direction;

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
                    .expect("Failed to convert to target type")
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

        // Process each row
        for row in 0..self.nrows() {
            // Skip this row if masked out
            if !mask[row] {
                continue;
            }

            let row_start = self.row_offsets()[row];
            let row_end = self.row_offsets()[row + 1];

            // Process all non-zero elements in this row
            for idx in row_start..row_end {
                let col = self.col_indices()[idx];
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

        // Process each row
        for row in 0..self.nrows() {
            let row_start = self.row_offsets()[row];
            let row_end = self.row_offsets()[row + 1];

            // Count non-zero elements in this row that are in masked-in columns
            for idx in row_start..row_end {
                let col = self.col_indices()[idx];

                // Skip this column if masked out
                if !mask[col] {
                    continue;
                }

                result[row] += T::one();
            }
        }

        Ok(result)
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
        let nrows = self.nrows();
        let mut result = vec![T::zero(); nrows];
        let values = self.values();
        let row_offsets = self.row_offsets();

        // Process in chunks of 4 rows when possible
        let chunk_size = 4;
        let chunks = nrows / chunk_size;
        let remainder = nrows % chunk_size;

        // Process chunks
        for chunk in 0..chunks {
            let base = chunk * chunk_size;
            let mut sums = [M::zero(); 4];

            // Process 4 rows at once to improve instruction-level parallelism
            (0..chunk_size).enumerate().for_each(|(i, offset)| {
                let row = base + offset;
                let start = row_offsets[row];
                let end = row_offsets[row + 1];

                // Direct sum in original type
                for &val in &values[start..end] {
                    sums[i] += val;
                }
            });

            // Convert results for the chunk
            sums.iter().enumerate().for_each(|(i, &sum)| {
                result[base + i] = T::from(sum).unwrap();
            });
        }

        // Handle remaining rows
        let base = chunks * chunk_size;
        for row in base..nrows {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            let mut sum = M::zero();

            for &val in &values[start..end] {
                sum += val;
            }
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

        let mut result = vec![T::zero(); self.ncols()];

        // Process each row
        for row in 0..self.nrows() {
            // Skip this row if masked out
            if !mask[row] {
                continue;
            }

            let row_start = self.row_offsets()[row];
            let row_end = self.row_offsets()[row + 1];

            // Process all non-zero elements in this row
            for idx in row_start..row_end {
                let col = self.col_indices()[idx];
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

        let mut result = vec![T::zero(); self.nrows()];

        // Process each row
        for row in 0..self.nrows() {
            let row_start = self.row_offsets()[row];
            let row_end = self.row_offsets()[row + 1];

            // Process all non-zero elements in this row
            for idx in row_start..row_end {
                let col = self.col_indices()[idx];

                // Skip this column if masked out
                if !mask[col] {
                    continue;
                }

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

impl<M> MatrixVariance for CsrMatrix<M>
where
    M: NumericOps + NumCast,
    CsrMatrix<M>: MatrixSum + MatrixNonZero,
{
    type Item = M;

    fn var_col<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
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
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
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

    /// Calculate column-wise variance and store results in the provided slice
    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
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
        T: Float + NumCast + AddAssign + Sum,
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

    fn var_col_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
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

        // Calculate masked sums and counts
        let sum: Vec<T> = self.sum_col_masked(mask)?;
        let count: Vec<I> = self.nonzero_col_masked(mask)?;

        let mut result = vec![T::zero(); self.ncols()];
        let mut squared_sums = vec![T::zero(); self.ncols()];

        // Calculate sum of squares for each column (using only masked-in rows)
        for row in 0..self.nrows() {
            // Skip this row if masked out
            if !mask[row] {
                continue;
            }

            let row_start = self.row_offsets()[row];
            let row_end = self.row_offsets()[row + 1];

            for idx in row_start..row_end {
                let col = self.col_indices()[idx];
                let val = T::from(self.values()[idx]).unwrap();
                squared_sums[col] += val * val;
            }
        }

        // Calculate variance for each column
        for col in 0..self.ncols() {
            if count[col] > I::zero() {
                let mean = sum[col] / count[col].into();
                result[col] = squared_sums[col] / count[col].into() - mean * mean;
            }
        }

        Ok(result)
    }

    fn var_row_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
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

        // Calculate masked sums and counts
        let sum: Vec<T> = self.sum_row_masked(mask)?;
        let count: Vec<I> = self.nonzero_row_masked(mask)?;

        let mut result = vec![T::zero(); self.nrows()];

        // Process each row to calculate variance
        for row in 0..self.nrows() {
            if count[row] > I::zero() {
                let mean = sum[row] / count[row].into();
                let row_start = self.row_offsets()[row];
                let row_end = self.row_offsets()[row + 1];

                // Calculate sum of squared differences for this row (only for masked-in columns)
                let mut sum_sq_diff = T::zero();
                for idx in row_start..row_end {
                    let col = self.col_indices()[idx];

                    // Skip masked out columns
                    if !mask[col] {
                        continue;
                    }

                    let val = T::from(self.values()[idx]).unwrap();
                    let diff = val - mean;
                    sum_sq_diff += diff * diff;
                }

                result[row] = sum_sq_diff / count[row].into();
            }
        }

        Ok(result)
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

impl<T: FloatOpsTS> Normalize<T> for CsrMatrix<T> {
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
            Direction::ROW => {
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

impl<T: FloatOpsTS> Log1P<T> for CsrMatrix<T> {
    fn log1p_normalize(&mut self) -> anyhow::Result<()> {
        let values = self.values_mut();
        for val in values.iter_mut() {
            *val = T::one() + *val;
            *val = val.ln();
        }
        Ok(())
    }
}

impl<M> BatchMatrixVariance for CsrMatrix<M>
where
    M: NumericOps + NumCast,
    CsrMatrix<M>: MatrixSum + MatrixNonZero,
{
    type Item = M;

    fn var_batch_row<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.nrows() {
            return Err(anyhow::anyhow!(
                "Batch vector length ({}) doesn't match matrix row count ({})",
                batches.len(),
                self.nrows()
            ));
        }

        // Group row indices by batch
        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (idx, batch) in batches.iter().enumerate() {
            batch_indices.entry(batch.clone()).or_default().push(idx);
        }

        // Calculate variance for each batch
        let mut result: HashMap<B, Vec<T>> = HashMap::new();

        for (batch, indices) in batch_indices {
            // Calculate variance for each column across the rows in this batch
            let mut batch_vars = vec![T::zero(); self.ncols()];
            let mut batch_means = vec![T::zero(); self.ncols()];
            let mut batch_counts = vec![0usize; self.ncols()];
            let mut batch_sum_sq = vec![T::zero(); self.ncols()];

            // First pass: calculate sum and count for each column
            for &row_idx in &indices {
                let row_start = self.row_offsets()[row_idx];
                let row_end = self.row_offsets()[row_idx + 1];

                for j in row_start..row_end {
                    let col = self.col_indices()[j];
                    let val = T::from(self.values()[j]).unwrap();
                    batch_means[col] = batch_means[col] + val;
                    batch_counts[col] += 1;
                }
            }

            // Calculate means
            for (mean, &count) in batch_means.iter_mut().zip(batch_counts.iter()) {
                if count > 0 {
                    *mean = *mean / T::from(count).unwrap();
                }
            }

            // Second pass: calculate sum of squared differences from mean
            for &row_idx in &indices {
                let row_start = self.row_offsets()[row_idx];
                let row_end = self.row_offsets()[row_idx + 1];

                for j in row_start..row_end {
                    let col = self.col_indices()[j];
                    let val = T::from(self.values()[j]).unwrap();
                    let diff = val - batch_means[col];
                    batch_sum_sq[col] = batch_sum_sq[col] + diff * diff;
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

    fn var_batch_col<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.ncols() {
            return Err(anyhow::anyhow!(
                "Batch vector length ({}) doesn't match matrix column count ({})",
                batches.len(),
                self.ncols()
            ));
        }

        // Create map of column indices to batch identifiers
        let col_to_batch: Vec<&B> = batches.iter().collect();

        // Group columns by batch
        let mut batch_columns: HashMap<B, Vec<usize>> = HashMap::new();
        for (col_idx, &batch) in col_to_batch.iter().enumerate() {
            batch_columns
                .entry(batch.clone())
                .or_default()
                .push(col_idx);
        }

        // Calculate variance for each batch
        let mut result: HashMap<B, Vec<T>> = HashMap::new();

        for (batch, col_indices) in batch_columns {
            // Calculate variance for each row across the columns in this batch
            let mut batch_vars = vec![T::zero(); self.nrows()];

            // Collect values for each row in this batch
            let mut row_values: Vec<Vec<T>> = vec![Vec::new(); self.nrows()];

            // Gather all values for each row across the batch's columns
            for row_idx in 0..self.nrows() {
                let row_start = self.row_offsets()[row_idx];
                let row_end = self.row_offsets()[row_idx + 1];

                for j in row_start..row_end {
                    let col = self.col_indices()[j];

                    // Check if this column is in the current batch
                    if col_indices.contains(&col) {
                        let val = T::from(self.values()[j]).unwrap();
                        row_values[row_idx].push(val);
                    }
                }
            }

            // Calculate variance for each row
            for (row_idx, values) in row_values.iter().enumerate() {
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
                    batch_vars[row_idx] = sum_sq_diff / T::from(values.len() - 1).unwrap();
                }
                // If values.len() <= 1, variance remains 0
            }

            result.insert(batch, batch_vars);
        }

        Ok(result)
    }
}

impl<M: NumericOps + NumCast> BatchMatrixMean for CsrMatrix<M> {
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
            let mut batch_means = vec![T::zero(); self.nrows()];

            // First calculate sum for each row in this batch
            for &col_idx in &col_indices {
                for row in 0..self.nrows() {
                    if let Some(entry) = self.get_entry(row, col_idx) {
                        batch_means[row] += T::from(entry.into_value()).unwrap();
                    }
                }
            }

            // Divide by number of columns in this batch to get mean
            let col_count = T::from(col_indices.len()).unwrap();
            for mean in &mut batch_means {
                *mean = *mean / col_count;
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
            let mut batch_means = vec![T::zero(); self.ncols()];

            // First calculate sum for each column in this batch
            for &row_idx in &row_indices {
                for (col_idx, _, value) in self.triplet_iter().filter(|&(row, _, _)| row == row_idx)
                {
                    batch_means[col_idx] += T::from(*value).unwrap();
                }
            }

            // Divide by number of rows in this batch to get mean
            let row_count = T::from(row_indices.len()).unwrap();
            for mean in &mut batch_means {
                *mean = *mean / row_count;
            }

            result.insert(batch, batch_means);
        }

        Ok(result)
    }
}

impl<M: NumericOps + NumCast> MatrixNTop for CsrMatrix<M> {
    type Item = M;

    fn sum_row_n_top<T>(&self, n: usize) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + Sum {
        let mut result = vec![T::zero(); self.nrows()];

        for row_idx in 0..self.nrows() {
            let row_start = self.row_offsets()[row_idx];
            let row_end = self.row_offsets()[row_idx + 1];

            let mut row_values: Vec<T> = Vec::new();
            for idx in row_start..row_end {
                if let Some(val) = T::from(self.values()[idx]) {
                    row_values.push(val);
                }
            }

            if row_values.len() <= n {
                result[row_idx] = row_values.into_iter().sum();
            } else {
                row_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                result[row_idx] = row_values.into_iter().take(n).sum();
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
