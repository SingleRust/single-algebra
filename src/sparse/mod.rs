use std::collections::HashMap;
use std::hash::Hash;
use std::ops::AddAssign;

use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};

use crate::NumericOps;
use crate::utils::BatchIdentifier;

pub mod csc;
pub mod csr;

pub trait MatrixNonZero {
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;

    /// Calculate masked non-zero counts for columns
    fn nonzero_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;

    /// Calculate masked non-zero counts for rows
    fn nonzero_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;
}

pub trait MatrixSum {
    type Item: NumCast;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn sum_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn sum_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum;

    /// Calculate masked sum for rows
    fn sum_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum;
}

pub trait MatrixVariance {
    type Item: NumCast;

    fn var_col<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn var_row<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn var_row_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    /// Calculate masked variance for columns
    fn var_col_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum;

    /// Calculate masked variance for rows
    fn var_row_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum;
}

pub trait MatrixMinMax {
    type Item;

    fn min_max_col<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_row<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_col_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_row_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;
}

pub trait BatchMatrixVariance {
    type Item: NumCast;

    /// Calculate row-wise variance for each batch
    fn var_batch_row<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier;

    /// Calculate column-wise variance for each batch
    fn var_batch_col<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier;
}

pub trait BatchMatrixMean {
    type Item: NumCast;

    /// Calculate row-wise mean for each batch
    fn mean_batch_row<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier;

    /// Calculate column-wise mean for each batch
    fn mean_batch_col<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum,
        B: BatchIdentifier;
}


