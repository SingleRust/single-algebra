use std::collections::HashMap;
use std::iter::Sum;
use std::ops::AddAssign;

use super::{
    BatchMatrixMean, BatchMatrixVariance, MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance,
};
use crate::sparse::MatrixNTop;
use crate::utils::Normalize;
use crate::utils::{BatchIdentifier, Log1P};
use anyhow::anyhow;
use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use single_utilities::traits::{FloatOpsTS, NumericOps};
use single_utilities::types::Direction;
use sprs::{CompressedStorage, CsMatI, SpIndex};

const PARALLEL_THRESHOLD: usize = 200_000;

impl<M, I> MatrixNonZero for CsMatI<M, I>
where
    M: NumericOps + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        let mut result = vec![T::zero(); self.cols()];
        self.nonzero_col_chunk(&mut result)?;
        Ok(result)
    }

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        let mut result = vec![T::zero(); self.rows()];
        self.nonzero_row_chunk(&mut result)?;
        Ok(result)
    }

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        if reference.len() < self.cols() {
            return Err(anyhow!("Reference slice too small for columns"));
        }

        match self.storage() {
            CompressedStorage::CSR => {
                for (_, (_, col)) in self.iter() {
                    reference[col.index()] += T::one();
                }
            }
            CompressedStorage::CSC => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<T> = (0..self.cols())
                        .into_par_iter()
                        .map(|col_idx| T::from(self.outer_view(col_idx).unwrap().nnz()).unwrap())
                        .collect();
                    for (i, count) in results.into_iter().enumerate() {
                        reference[i] += count;
                    }
                } else {
                    for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                        reference[col_idx] += T::from(col_vec.nnz()).unwrap();
                    }
                }
            }
        }
        Ok(())
    }

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        if reference.len() < self.rows() {
            return Err(anyhow!("Reference slice too small for rows"));
        }

        match self.storage() {
            CompressedStorage::CSR => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<T> = (0..self.rows())
                        .into_par_iter()
                        .map(|row_idx| T::from(self.outer_view(row_idx).unwrap().nnz()).unwrap())
                        .collect();
                    for (i, count) in results.into_iter().enumerate() {
                        reference[i] += count;
                    }
                } else {
                    for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                        reference[row_idx] += T::from(row_vec.nnz()).unwrap();
                    }
                }
            }
            CompressedStorage::CSC => {
                for (_, (row, _)) in self.iter() {
                    reference[row.index()] += T::one();
                }
            }
        }
        Ok(())
    }

    fn nonzero_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        if mask.len() < self.rows() {
            return Err(anyhow!("Mask too small for rows"));
        }

        let mut result = vec![T::zero(); self.cols()];
        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    if mask[row_idx] {
                        for &col_idx in row_vec.indices() {
                            result[col_idx.index()] += T::one();
                        }
                    }
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    let mut count = T::zero();
                    for &row_idx in col_vec.indices() {
                        if mask[row_idx.index()] {
                            count += T::one();
                        }
                    }
                    result[col_idx] = count;
                }
            }
        }
        Ok(result)
    }

    fn nonzero_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        if mask.len() < self.cols() {
            return Err(anyhow!("Mask too small for columns"));
        }

        let mut result = vec![T::zero(); self.rows()];
        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    let mut count = T::zero();
                    for &col_idx in row_vec.indices() {
                        if mask[col_idx.index()] {
                            count += T::one();
                        }
                    }
                    result[row_idx] = count;
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    if mask[col_idx] {
                        for &row_idx in col_vec.indices() {
                            result[row_idx.index()] += T::one();
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

impl<M, I> MatrixSum for CsMatI<M, I>
where
    M: NumericOps + NumCast + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    type Item = M;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let mut result = vec![T::zero(); self.cols()];
        self.sum_col_chunk(&mut result)?;
        Ok(result)
    }

    fn sum_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let mut result = vec![T::zero(); self.rows()];
        self.sum_row_chunk(&mut result)?;
        Ok(result)
    }

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        if reference.len() < self.cols() {
            return Err(anyhow!("Reference slice too small for columns"));
        }

        match self.storage() {
            CompressedStorage::CSR => {
                for (val, (_, col)) in self.iter() {
                    reference[col.index()] += T::from(*val).unwrap();
                }
            }
            CompressedStorage::CSC => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<T> = (0..self.cols())
                        .into_par_iter()
                        .map(|col_idx| {
                            self.outer_view(col_idx)
                                .unwrap()
                                .data()
                                .iter()
                                .map(|&v| T::from(v).unwrap())
                                .sum()
                        })
                        .collect();
                    for (i, sum) in results.into_iter().enumerate() {
                        reference[i] += sum;
                    }
                } else {
                    for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                        let sum: T = col_vec.data().iter().map(|&v| T::from(v).unwrap()).sum();
                        reference[col_idx] += sum;
                    }
                }
            }
        }
        Ok(())
    }

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        if reference.len() < self.rows() {
            return Err(anyhow!("Reference slice too small for rows"));
        }

        match self.storage() {
            CompressedStorage::CSR => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<T> = (0..self.rows())
                        .into_par_iter()
                        .map(|row_idx| {
                            self.outer_view(row_idx)
                                .unwrap()
                                .data()
                                .iter()
                                .map(|&v| T::from(v).unwrap())
                                .sum()
                        })
                        .collect();
                    for (i, sum) in results.into_iter().enumerate() {
                        reference[i] += sum;
                    }
                } else {
                    for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                        let sum: T = row_vec.data().iter().map(|&v| T::from(v).unwrap()).sum();
                        reference[row_idx] += sum;
                    }
                }
            }
            CompressedStorage::CSC => {
                for (val, (row, _)) in self.iter() {
                    reference[row.index()] += T::from(*val).unwrap();
                }
            }
        }
        Ok(())
    }

    fn sum_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        if mask.len() < self.rows() {
            return Err(anyhow!("Mask too small for rows"));
        }

        let mut result = vec![T::zero(); self.cols()];
        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    if mask[row_idx] {
                        for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter())
                        {
                            result[col_idx.index()] += T::from(val).unwrap();
                        }
                    }
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    let mut sum = T::zero();
                    for (&row_idx, &val) in col_vec.indices().iter().zip(col_vec.data().iter()) {
                        if mask[row_idx.index()] {
                            sum += T::from(val).unwrap();
                        }
                    }
                    result[col_idx] = sum;
                }
            }
        }
        Ok(result)
    }

    fn sum_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        if mask.len() < self.cols() {
            return Err(anyhow!("Mask too small for columns"));
        }

        let mut result = vec![T::zero(); self.rows()];
        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    let mut sum = T::zero();
                    for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter()) {
                        if mask[col_idx.index()] {
                            sum += T::from(val).unwrap();
                        }
                    }
                    result[row_idx] = sum;
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    if mask[col_idx] {
                        for (&row_idx, &val) in col_vec.indices().iter().zip(col_vec.data().iter())
                        {
                            result[row_idx.index()] += T::from(val).unwrap();
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    fn sum_col_squared<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let mut result = vec![T::zero(); self.cols()];
        match self.storage() {
            CompressedStorage::CSR => {
                for (val, (_, col)) in self.iter() {
                    let v = T::from(*val).unwrap();
                    result[col.index()] += v * v;
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    let sum_sq: T = col_vec
                        .data()
                        .iter()
                        .map(|&v| {
                            let val = T::from(v).unwrap();
                            val * val
                        })
                        .sum();
                    result[col_idx] = sum_sq;
                }
            }
        }
        Ok(result)
    }

    fn sum_row_squared<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let mut result = vec![T::zero(); self.rows()];
        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    let sum_sq: T = row_vec
                        .data()
                        .iter()
                        .map(|&v| {
                            let val = T::from(v).unwrap();
                            val * val
                        })
                        .sum();
                    result[row_idx] = sum_sq;
                }
            }
            CompressedStorage::CSC => {
                for (val, (row, _)) in self.iter() {
                    let v = T::from(*val).unwrap();
                    result[row.index()] += v * v;
                }
            }
        }
        Ok(result)
    }
}

impl<M, I> MatrixVariance for CsMatI<M, I>
where
    M: NumericOps + NumCast + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
    CsMatI<M, I>: MatrixSum + MatrixNonZero,
{
    type Item = M;

    fn var_col<V, T>(&self) -> anyhow::Result<Vec<T>>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let n = T::from(self.rows()).unwrap();
        if n <= T::one() {
            return Ok(vec![T::zero(); self.cols()]);
        }

        let means: Vec<T> = self.sum_col::<T>()?.into_iter().map(|s| s / n).collect();
        let mut sum_sq_diffs = vec![T::zero(); self.cols()];

        match self.storage() {
            CompressedStorage::CSR => {
                for (val, (_, col)) in self.iter() {
                    let diff = T::from(*val).unwrap() - means[col.index()];
                    sum_sq_diffs[col.index()] += diff * diff;
                }
                let nz_counts: Vec<V> = self.nonzero_col()?;
                for col in 0..self.cols() {
                    let z_count = n - T::from(nz_counts[col]).unwrap();
                    if z_count > T::zero() {
                        let diff = T::zero() - means[col];
                        sum_sq_diffs[col] += z_count * diff * diff;
                    }
                }
            }
            CompressedStorage::CSC => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<T> = (0..self.cols())
                        .into_par_iter()
                        .map(|col_idx| {
                            let col_vec = self.outer_view(col_idx).unwrap();
                            let mean = means[col_idx];
                            let mut ssd = T::zero();
                            for &val in col_vec.data() {
                                let diff = T::from(val).unwrap() - mean;
                                ssd += diff * diff;
                            }
                            let z_count = n - T::from(col_vec.nnz()).unwrap();
                            if z_count > T::zero() {
                                let diff = T::zero() - mean;
                                ssd += z_count * diff * diff;
                            }
                            ssd
                        })
                        .collect();
                    sum_sq_diffs = results;
                } else {
                    for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                        let mean = means[col_idx];
                        let mut ssd = T::zero();
                        for &val in col_vec.data() {
                            let diff = T::from(val).unwrap() - mean;
                            ssd += diff * diff;
                        }
                        let z_count = n - T::from(col_vec.nnz()).unwrap();
                        if z_count > T::zero() {
                            let diff = T::zero() - mean;
                            ssd += z_count * diff * diff;
                        }
                        sum_sq_diffs[col_idx] = ssd;
                    }
                }
            }
        }

        let n_minus_1 = n - T::one();
        Ok(sum_sq_diffs
            .into_iter()
            .map(|ssd| ssd / n_minus_1)
            .collect())
    }

    fn var_row<V, T>(&self) -> anyhow::Result<Vec<T>>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let n = T::from(self.cols()).unwrap();
        if n <= T::one() {
            return Ok(vec![T::zero(); self.rows()]);
        }

        let means: Vec<T> = self.sum_row::<T>()?.into_iter().map(|s| s / n).collect();
        let mut sum_sq_diffs = vec![T::zero(); self.rows()];

        match self.storage() {
            CompressedStorage::CSR => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<T> = (0..self.rows())
                        .into_par_iter()
                        .map(|row_idx| {
                            let row_vec = self.outer_view(row_idx).unwrap();
                            let mean = means[row_idx];
                            let mut ssd = T::zero();
                            for &val in row_vec.data() {
                                let diff = T::from(val).unwrap() - mean;
                                ssd += diff * diff;
                            }
                            let z_count = n - T::from(row_vec.nnz()).unwrap();
                            if z_count > T::zero() {
                                let diff = T::zero() - mean;
                                ssd += z_count * diff * diff;
                            }
                            ssd
                        })
                        .collect();
                    sum_sq_diffs = results;
                } else {
                    for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                        let mean = means[row_idx];
                        let mut ssd = T::zero();
                        for &val in row_vec.data() {
                            let diff = T::from(val).unwrap() - mean;
                            ssd += diff * diff;
                        }
                        let z_count = n - T::from(row_vec.nnz()).unwrap();
                        if z_count > T::zero() {
                            let diff = T::zero() - mean;
                            ssd += z_count * diff * diff;
                        }
                        sum_sq_diffs[row_idx] = ssd;
                    }
                }
            }
            CompressedStorage::CSC => {
                for (val, (row, _)) in self.iter() {
                    let diff = T::from(*val).unwrap() - means[row.index()];
                    sum_sq_diffs[row.index()] += diff * diff;
                }
                let nz_counts: Vec<V> = self.nonzero_row()?;
                for row in 0..self.rows() {
                    let z_count = n - T::from(nz_counts[row]).unwrap();
                    if z_count > T::zero() {
                        let diff = T::zero() - means[row];
                        sum_sq_diffs[row] += z_count * diff * diff;
                    }
                }
            }
        }

        let n_minus_1 = n - T::one();
        Ok(sum_sq_diffs
            .into_iter()
            .map(|ssd| ssd / n_minus_1)
            .collect())
    }

    fn var_col_chunk<V, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let vars = self.var_col::<V, T>()?;
        reference[..vars.len()].copy_from_slice(&vars);
        Ok(())
    }

    fn var_row_chunk<V, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let vars = self.var_row::<V, T>()?;
        reference[..vars.len()].copy_from_slice(&vars);
        Ok(())
    }

    fn var_col_masked<V, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let n = T::from(mask.iter().filter(|&&m| m).count()).unwrap();
        if n <= T::one() {
            return Ok(vec![T::zero(); self.cols()]);
        }

        let sums: Vec<T> = self.sum_col_masked(mask)?;
        let means: Vec<T> = sums.into_iter().map(|s| s / n).collect();
        let mut sum_sq_diffs = vec![T::zero(); self.cols()];
        let mut nz_counts = vec![V::zero(); self.cols()];

        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    if mask[row_idx] {
                        for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter())
                        {
                            let diff = T::from(val).unwrap() - means[col_idx.index()];
                            sum_sq_diffs[col_idx.index()] += diff * diff;
                            nz_counts[col_idx.index()] += V::one();
                        }
                    }
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    let mean = means[col_idx];
                    let mut ssd = T::zero();
                    let mut count = V::zero();
                    for (&row_idx, &val) in col_vec.indices().iter().zip(col_vec.data().iter()) {
                        if mask[row_idx.index()] {
                            let diff = T::from(val).unwrap() - mean;
                            ssd += diff * diff;
                            count += V::one();
                        }
                    }
                    sum_sq_diffs[col_idx] = ssd;
                    nz_counts[col_idx] = count;
                }
            }
        }

        for col in 0..self.cols() {
            let z_count = n - T::from(nz_counts[col]).unwrap();
            if z_count > T::zero() {
                let diff = T::zero() - means[col];
                sum_sq_diffs[col] += z_count * diff * diff;
            }
        }

        let n_minus_1 = n - T::one();
        Ok(sum_sq_diffs
            .into_iter()
            .map(|ssd| ssd / n_minus_1)
            .collect())
    }

    fn var_row_masked<V, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let n = T::from(mask.iter().filter(|&&m| m).count()).unwrap();
        if n <= T::one() {
            return Ok(vec![T::zero(); self.rows()]);
        }

        let sums: Vec<T> = self.sum_row_masked(mask)?;
        let means: Vec<T> = sums.into_iter().map(|s| s / n).collect();
        let mut sum_sq_diffs = vec![T::zero(); self.rows()];
        let mut nz_counts = vec![V::zero(); self.rows()];

        match self.storage() {
            CompressedStorage::CSR => {
                for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                    let mean = means[row_idx];
                    let mut ssd = T::zero();
                    let mut count = V::zero();
                    for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter()) {
                        if mask[col_idx.index()] {
                            let diff = T::from(val).unwrap() - mean;
                            ssd += diff * diff;
                            count += V::one();
                        }
                    }
                    sum_sq_diffs[row_idx] = ssd;
                    nz_counts[row_idx] = count;
                }
            }
            CompressedStorage::CSC => {
                for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                    if mask[col_idx] {
                        for (&row_idx, &val) in col_vec.indices().iter().zip(col_vec.data().iter())
                        {
                            let diff = T::from(val).unwrap() - means[row_idx.index()];
                            sum_sq_diffs[row_idx.index()] += diff * diff;
                            nz_counts[row_idx.index()] += V::one();
                        }
                    }
                }
            }
        }

        for row in 0..self.rows() {
            let z_count = n - T::from(nz_counts[row]).unwrap();
            if z_count > T::zero() {
                let diff = T::zero() - means[row];
                sum_sq_diffs[row] += z_count * diff * diff;
            }
        }

        let n_minus_1 = n - T::one();
        Ok(sum_sq_diffs
            .into_iter()
            .map(|ssd| ssd / n_minus_1)
            .collect())
    }
}

impl<M, I> MatrixMinMax for CsMatI<M, I>
where
    M: NumericOps + NumCast + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    type Item = M;

    fn min_max_col<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let mut min = vec![Item::max_value(); self.cols()];
        let mut max = vec![Item::min_value(); self.cols()];
        self.min_max_col_chunk((&mut min, &mut max))?;
        Ok((min, max))
    }

    fn min_max_row<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let mut min = vec![Item::max_value(); self.rows()];
        let mut max = vec![Item::min_value(); self.rows()];
        self.min_max_row_chunk((&mut min, &mut max))?;
        Ok((min, max))
    }

    fn min_max_col_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let (min_ref, max_ref) = reference;
        match self.storage() {
            CompressedStorage::CSR => {
                for (val, (_, col)) in self.iter() {
                    let v = Item::from(*val).unwrap();
                    if v < min_ref[col.index()] {
                        min_ref[col.index()] = v;
                    }
                    if v > max_ref[col.index()] {
                        max_ref[col.index()] = v;
                    }
                }
            }
            CompressedStorage::CSC => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<(Item, Item)> = (0..self.cols())
                        .into_par_iter()
                        .map(|col_idx| {
                            let mut c_min = Item::max_value();
                            let mut c_max = Item::min_value();
                            for &val in self.outer_view(col_idx).unwrap().data() {
                                let v = Item::from(val).unwrap();
                                if v < c_min {
                                    c_min = v;
                                }
                                if v > c_max {
                                    c_max = v;
                                }
                            }
                            (c_min, c_max)
                        })
                        .collect();
                    for (i, (c_min, c_max)) in results.into_iter().enumerate() {
                        min_ref[i] = c_min;
                        max_ref[i] = c_max;
                    }
                } else {
                    for (col_idx, col_vec) in self.outer_iterator().enumerate() {
                        let mut c_min = Item::max_value();
                        let mut c_max = Item::min_value();
                        for &val in col_vec.data() {
                            let v = Item::from(val).unwrap();
                            if v < c_min {
                                c_min = v;
                            }
                            if v > c_max {
                                c_max = v;
                            }
                        }
                        min_ref[col_idx] = c_min;
                        max_ref[col_idx] = c_max;
                    }
                }
            }
        }
        Ok(())
    }

    fn min_max_row_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let (min_ref, max_ref) = reference;
        match self.storage() {
            CompressedStorage::CSR => {
                if self.nnz() > PARALLEL_THRESHOLD {
                    let results: Vec<(Item, Item)> = (0..self.rows())
                        .into_par_iter()
                        .map(|row_idx| {
                            let mut r_min = Item::max_value();
                            let mut r_max = Item::min_value();
                            for &val in self.outer_view(row_idx).unwrap().data() {
                                let v = Item::from(val).unwrap();
                                if v < r_min {
                                    r_min = v;
                                }
                                if v > r_max {
                                    r_max = v;
                                }
                            }
                            (r_min, r_max)
                        })
                        .collect();
                    for (i, (r_min, r_max)) in results.into_iter().enumerate() {
                        min_ref[i] = r_min;
                        max_ref[i] = r_max;
                    }
                } else {
                    for (row_idx, row_vec) in self.outer_iterator().enumerate() {
                        let mut r_min = Item::max_value();
                        let mut r_max = Item::min_value();
                        for &val in row_vec.data() {
                            let v = Item::from(val).unwrap();
                            if v < r_min {
                                r_min = v;
                            }
                            if v > r_max {
                                r_max = v;
                            }
                        }
                        min_ref[row_idx] = r_min;
                        max_ref[row_idx] = r_max;
                    }
                }
            }
            CompressedStorage::CSC => {
                for (val, (row, _)) in self.iter() {
                    let v = Item::from(*val).unwrap();
                    if v < min_ref[row.index()] {
                        min_ref[row.index()] = v;
                    }
                    if v > max_ref[row.index()] {
                        max_ref[row.index()] = v;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<M, I> MatrixNTop for CsMatI<M, I>
where
    M: NumericOps + NumCast + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    type Item = M;

    fn sum_row_n_top<T>(&self, n: usize) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let mut result = vec![T::zero(); self.rows()];
        match self.storage() {
            CompressedStorage::CSR => {
                let results: Vec<T> = (0..self.rows())
                    .into_par_iter()
                    .map(|row_idx| {
                        let row_vec = self.outer_view(row_idx).unwrap();
                        let mut data: Vec<T> = row_vec
                            .data()
                            .iter()
                            .map(|&v| T::from(v).unwrap())
                            .collect();
                        if data.len() <= n {
                            data.into_iter().sum()
                        } else {
                            data.select_nth_unstable_by(n - 1, |a, b| {
                                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            data.into_iter().take(n).sum()
                        }
                    })
                    .collect();
                result = results;
            }
            CompressedStorage::CSC => {
                let mut row_values: Vec<Vec<T>> = vec![Vec::new(); self.rows()];
                for (val, (row, _)) in self.iter() {
                    row_values[row.index()].push(T::from(*val).unwrap());
                }
                let results: Vec<T> = row_values
                    .into_par_iter()
                    .map(|mut data| {
                        if data.len() <= n {
                            data.into_iter().sum()
                        } else {
                            data.select_nth_unstable_by(n - 1, |a, b| {
                                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            data.into_iter().take(n).sum()
                        }
                    })
                    .collect();
                result = results;
            }
        }
        Ok(result)
    }
}

impl<T, I> Normalize<T> for CsMatI<T, I>
where
    T: FloatOpsTS + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    fn normalize<U: FloatOpsTS>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &Direction,
    ) -> anyhow::Result<()> {
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

        let storage = self.storage();
        for (outer_idx, mut outer_vec) in self.outer_iterator_mut().enumerate() {
            let scale = match storage {
                CompressedStorage::CSR => match direction {
                    Direction::ROW => scaling_factors[outer_idx],
                    Direction::COLUMN => U::zero(),
                },
                CompressedStorage::CSC => match direction {
                    Direction::ROW => U::zero(),
                    Direction::COLUMN => scaling_factors[outer_idx],
                },
            };

            if scale > U::zero() {
                for (_, val) in outer_vec.iter_mut() {
                    *val = T::from(U::from(*val).unwrap() * scale).unwrap();
                }
            } else {
                for (inner_idx, val) in outer_vec.iter_mut() {
                    let s = scaling_factors[inner_idx.index()];
                    if s > U::zero() {
                        *val = T::from(U::from(*val).unwrap() * s).unwrap();
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T, I> Log1P<T> for CsMatI<T, I>
where
    T: FloatOpsTS + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    fn log1p_normalize(&mut self) -> anyhow::Result<()> {
        self.data_mut().into_par_iter().for_each(|val| {
            *val = (*val + T::one()).ln();
        });
        Ok(())
    }
}

impl<M, I> BatchMatrixVariance for CsMatI<M, I>
where
    M: NumericOps + NumCast + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
    CsMatI<M, I>: MatrixSum + MatrixNonZero,
{
    type Item = M;

    fn var_batch_row<V, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.rows() {
            return Err(anyhow!("Batches length must match rows"));
        }

        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (idx, b) in batches.iter().enumerate() {
            batch_indices.entry(b.clone()).or_default().push(idx);
        }

        let mut result = HashMap::new();
        for (batch, indices) in batch_indices {
            let n = T::from(indices.len()).unwrap();
            let mut b_sums = vec![T::zero(); self.cols()];

            for &row_idx in &indices {
                if let Some(row_vec) = self.outer_view(row_idx) {
                    for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter()) {
                        b_sums[col_idx.index()] += T::from(val).unwrap();
                    }
                }
            }

            let mut b_vars = vec![T::zero(); self.cols()];
            if n > T::one() {
                let n_minus_1 = n - T::one();
                let b_means: Vec<T> = b_sums.into_iter().map(|s| s / n).collect();
                let mut sum_sq_diffs = vec![T::zero(); self.cols()];
                let mut nz_counts = vec![V::zero(); self.cols()];

                for &row_idx in &indices {
                    if let Some(row_vec) = self.outer_view(row_idx) {
                        for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter())
                        {
                            let diff = T::from(val).unwrap() - b_means[col_idx.index()];
                            sum_sq_diffs[col_idx.index()] += diff * diff;
                            nz_counts[col_idx.index()] += V::one();
                        }
                    }
                }

                for i in 0..self.cols() {
                    let z_count = n - T::from(nz_counts[i]).unwrap();
                    if z_count > T::zero() {
                        let diff = T::zero() - b_means[i];
                        sum_sq_diffs[i] += z_count * diff * diff;
                    }
                    b_vars[i] = sum_sq_diffs[i] / n_minus_1;
                }
            }
            result.insert(batch, b_vars);
        }
        Ok(result)
    }

    fn var_batch_col<V, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        V: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.cols() {
            return Err(anyhow!("Batches length must match cols"));
        }

        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (idx, b) in batches.iter().enumerate() {
            batch_indices.entry(b.clone()).or_default().push(idx);
        }

        let mut result = HashMap::new();
        for (batch, indices) in batch_indices {
            let n = T::from(indices.len()).unwrap();
            let mut b_sums = vec![T::zero(); self.rows()];

            for &col_idx in &indices {
                match self.storage() {
                    CompressedStorage::CSC => {
                        if let Some(col_vec) = self.outer_view(col_idx) {
                            for (&row_idx, &val) in
                                col_vec.indices().iter().zip(col_vec.data().iter())
                            {
                                b_sums[row_idx.index()] += T::from(val).unwrap();
                            }
                        }
                    }
                    CompressedStorage::CSR => {
                        for row_idx in 0..self.rows() {
                            if let Some(val) = self.get(row_idx, col_idx) {
                                b_sums[row_idx] += T::from(*val).unwrap();
                            }
                        }
                    }
                }
            }

            let mut b_vars = vec![T::zero(); self.rows()];
            if n > T::one() {
                let n_minus_1 = n - T::one();
                let b_means: Vec<T> = b_sums.into_iter().map(|s| s / n).collect();
                let mut sum_sq_diffs = vec![T::zero(); self.rows()];
                let mut nz_counts = vec![V::zero(); self.rows()];

                for &col_idx in &indices {
                    match self.storage() {
                        CompressedStorage::CSC => {
                            if let Some(col_vec) = self.outer_view(col_idx) {
                                for (&row_idx, &val) in
                                    col_vec.indices().iter().zip(col_vec.data().iter())
                                {
                                    let diff = T::from(val).unwrap() - b_means[row_idx.index()];
                                    sum_sq_diffs[row_idx.index()] += diff * diff;
                                    nz_counts[row_idx.index()] += V::one();
                                }
                            }
                        }
                        CompressedStorage::CSR => {
                            for row_idx in 0..self.rows() {
                                if let Some(val) = self.get(row_idx, col_idx) {
                                    let diff = T::from(*val).unwrap() - b_means[row_idx];
                                    sum_sq_diffs[row_idx] += diff * diff;
                                    nz_counts[row_idx] += V::one();
                                }
                            }
                        }
                    }
                }

                for i in 0..self.rows() {
                    let z_count = n - T::from(nz_counts[i]).unwrap();
                    if z_count > T::zero() {
                        let diff = T::zero() - b_means[i];
                        sum_sq_diffs[i] += z_count * diff * diff;
                    }
                    b_vars[i] = sum_sq_diffs[i] / n_minus_1;
                }
            }
            result.insert(batch, b_vars);
        }
        Ok(result)
    }
}

impl<M, I> BatchMatrixMean for CsMatI<M, I>
where
    M: NumericOps + NumCast + Send + Sync,
    I: SpIndex + PrimInt + Unsigned + Send + Sync,
{
    type Item = M;

    fn mean_batch_row<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.cols() {
            return Err(anyhow!("Batches length must match cols"));
        }

        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (idx, b) in batches.iter().enumerate() {
            batch_indices.entry(b.clone()).or_default().push(idx);
        }

        let mut result = HashMap::new();
        for (batch, indices) in batch_indices {
            let n = T::from(indices.len()).unwrap();
            let mut b_sums = vec![T::zero(); self.rows()];

            for &col_idx in &indices {
                match self.storage() {
                    CompressedStorage::CSC => {
                        if let Some(col_vec) = self.outer_view(col_idx) {
                            for (&row_idx, &val) in
                                col_vec.indices().iter().zip(col_vec.data().iter())
                            {
                                b_sums[row_idx.index()] += T::from(val).unwrap();
                            }
                        }
                    }
                    CompressedStorage::CSR => {
                        for row_idx in 0..self.rows() {
                            if let Some(val) = self.get(row_idx, col_idx) {
                                b_sums[row_idx] += T::from(*val).unwrap();
                            }
                        }
                    }
                }
            }

            for s in &mut b_sums {
                *s = *s / n;
            }
            result.insert(batch, b_sums);
        }
        Ok(result)
    }

    fn mean_batch_col<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier,
    {
        if batches.len() != self.rows() {
            return Err(anyhow!("Batches length must match rows"));
        }

        let mut batch_indices: HashMap<B, Vec<usize>> = HashMap::new();
        for (idx, b) in batches.iter().enumerate() {
            batch_indices.entry(b.clone()).or_default().push(idx);
        }

        let mut result = HashMap::new();
        for (batch, indices) in batch_indices {
            let n = T::from(indices.len()).unwrap();
            let mut b_sums = vec![T::zero(); self.cols()];

            for &row_idx in &indices {
                if let Some(row_vec) = self.outer_view(row_idx) {
                    for (&col_idx, &val) in row_vec.indices().iter().zip(row_vec.data().iter()) {
                        b_sums[col_idx.index()] += T::from(val).unwrap();
                    }
                }
            }

            for s in &mut b_sums {
                *s = *s / n;
            }
            result.insert(batch, b_sums);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    fn create_test_csr() -> CsMatI<f64, usize> {
        let mat = CsMatI::new_csc(
            (3, 3),
            vec![0, 2, 4, 5],
            vec![0, 2, 0, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );
        mat.to_csr()
    }

    #[test]
    fn test_sprs_nonzero() {
        let mat = create_test_csr();
        let counts: Vec<u32> = mat.nonzero_row().unwrap();
        assert_eq!(counts, vec![2, 1, 2]);

        let col_counts: Vec<u32> = mat.nonzero_col().unwrap();
        assert_eq!(col_counts, vec![2, 2, 1]);
    }

    #[test]
    fn test_sprs_sum() {
        let mat = create_test_csr();
        let sums: Vec<f64> = mat.sum_row().unwrap();
        assert_eq!(sums, vec![4.0, 4.0, 7.0]);

        let col_sums: Vec<f64> = mat.sum_col().unwrap();
        assert_eq!(col_sums, vec![3.0, 7.0, 5.0]);
    }

    #[test]
    fn test_sprs_variance() {
        let mat = create_test_csr();
        let vars: Vec<f64> = mat.var_row::<u32, f64>().unwrap();
        assert!((vars[0] - 2.3333333333333335).abs() < 1e-10);
        assert!((vars[1] - 5.333333333333333).abs() < 1e-10);
        assert!((vars[2] - 6.333333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_sprs_min_max() {
        let mat = create_test_csr();
        let (mins, maxs) = mat.min_max_row::<f64>().unwrap();
        assert_eq!(mins[0], 1.0);
        assert_eq!(maxs[0], 3.0);
        assert_eq!(mins[1], 4.0);
        assert_eq!(maxs[1], 4.0);
    }

    #[test]
    fn test_sprs_masked() {
        let mat = create_test_csr(); // 3x3 CSR
        let mask = vec![true, false, true]; // Include rows 0 and 2
        let sums: Vec<f64> = mat.sum_col_masked(&mask).unwrap();
        // actual mat CSR:
        // row 0: {0: 1.0, 1: 3.0}
        // row 1: {1: 4.0}
        // row 2: {0: 2.0, 2: 5.0}
        // Masked rows (0, 2): Col 0: 1+2=3, Col 1: 3, Col 2: 5
        assert_eq!(sums, vec![3.0, 3.0, 5.0]);
    }

    #[test]
    fn test_sprs_batch() {
        let mat = create_test_csr();
        let batches = vec!["A", "A", "B"];
        let means = mat.mean_batch_col::<f64, _>(&batches).unwrap();

        // Batch A (Rows 0, 1):
        // row 0: {0: 1.0, 1: 3.0}
        // row 1: {1: 4.0}
        // Mean Batch A (Col Sums / 2): Col 0: 1/2=0.5, Col 1: (3+4)/2=3.5, Col 2: 0/2=0
        assert_eq!(means.get("A").unwrap(), &vec![0.5, 3.5, 0.0]);
    }

    #[test]
    fn test_sprs_ntop() {
        let mat = create_test_csr();
        // row 0: [1, 3, 0] -> Top 1: 3
        // row 1: [0, 4, 0] -> Top 1: 4
        // row 2: [2, 0, 5] -> Top 1: 5
        let top1 = mat.sum_row_n_top::<f64>(1).unwrap();
        assert_eq!(top1, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sprs_csc_native() {
        let mat = CsMatI::new_csc(
            (3, 3),
            vec![0usize, 2, 4, 5],
            vec![0usize, 2, 0, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );
        let sums: Vec<f64> = mat.sum_col().unwrap();
        assert_eq!(sums, vec![3.0, 7.0, 5.0]);
    }

    #[test]
    fn test_sprs_numerical_stability() {
        // Values with large offset: 1e9 + [1.0, 2.0, 3.0]
        // Variance should be exactly 1.0
        let offset = 1_000_000_000.0;
        let mat = CsMatI::new_csc(
            (3, 1),
            vec![0usize, 3],
            vec![0usize, 1, 2],
            vec![offset + 1.0, offset + 2.0, offset + 3.0],
        );

        let vars = mat.var_col::<u32, f64>().unwrap();
        // Stable two-pass algorithm should be very accurate
        assert!(
            (vars[0] - 1.0).abs() < 1e-10,
            "Variance {} should be exactly 1.0",
            vars[0]
        );
    }

    #[test]
    fn test_sprs_empty_and_singular() {
        // 0x0 Matrix
        let empty = CsMatI::<f64, usize>::new_csc((0, 0), vec![0], vec![], vec![]);
        assert!(empty.sum_row::<f64>().unwrap().is_empty());
        assert!(empty.nonzero_col::<u32>().unwrap().is_empty());

        // 10x10 Matrix with only one element
        let mut coo = sprs::TriMat::new((10, 10));
        coo.add_triplet(5, 5, 42.0);
        let sparse = coo.to_csr::<usize>();

        let sums = sparse.sum_row::<f64>().unwrap();
        assert_eq!(sums[5], 42.0);
        assert_eq!(sums[0], 0.0);

        let nz = sparse.nonzero_col::<u32>().unwrap();
        assert_eq!(nz[5], 1);
        assert_eq!(nz[0], 0);
    }

    #[test]
    fn test_sprs_batch_edge_cases() {
        let mat = create_test_csr();
        // Batch with only one row - variance should be 0.0 or handled
        let batches = vec!["Single", "Other", "Other"];
        let vars = mat.var_batch_row::<u32, f64, _>(&batches).unwrap();

        // "Single" batch only has Row 0. Sample variance requires N > 1.
        let single_vars = vars.get("Single").unwrap();
        for &v in single_vars {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_sprs_f32_u32() {
        // Test with f32 values and u32 indices
        let mat = CsMatI::<f32, u32>::new_csc((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        let sums = mat.sum_col::<f32>().unwrap();
        assert_eq!(sums, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sprs_normalization_inplace() {
        let mut mat = create_test_csr(); // original sums: [4, 4, 7]
        let row_sums = vec![4.0, 4.0, 7.0];
        mat.normalize(&row_sums, 1.0, &Direction::ROW).unwrap();

        let new_sums = mat.sum_row::<f64>().unwrap();
        for &s in &new_sums {
            assert!((s - 1.0).abs() < 1e-9);
        }

        // Test Log1P on the normalized matrix (all non-zeros should be scaled)
        mat.log1p_normalize().unwrap();
        // The values were normalized such that row sums are 1.0.
        // For a row with 2 non-zeros (like row 0), values might be 1/4 and 3/4.
        // Let's just check that all values are now transformed.
        for (val, _) in mat.iter() {
            assert!(*val > 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_sprs_large_parallel() {
        let size = 1000;
        let mut coo = sprs::TriMat::new((size, size));
        for i in 0..size {
            coo.add_triplet(i, i, 1.0);
        }
        let mat = coo.to_csr::<usize>();

        // This should trigger parallel logic if NNZ is high, but we can just check correctness
        let sums = mat.sum_row::<f64>().unwrap();
        assert_eq!(sums.len(), size);
        for s in sums {
            assert_eq!(s, 1.0);
        }
    }
}
