//! # Sparse Matrix Module
//!
//! This module provides trait-based operations for sparse matrices in CSR (Compressed Sparse Row)
//! and CSC (Compressed Sparse Column) formats. It implements efficient statistical computations
//! including non-zero counting, summation, variance calculation, and batch operations.
//!
//! ## Matrix Formats
//! - **CSR (Compressed Sparse Row)**: Efficient for row-wise operations and matrix-vector multiplication
//! - **CSC (Compressed Sparse Column)**: Efficient for column-wise operations and transposed operations
//!
//! ## Core Traits
//! - [`MatrixNonZero`]: Count non-zero elements per row/column
//! - [`MatrixSum`]: Sum elements per row/column with squared variants
//! - [`MatrixVariance`]: Calculate variance per row/column
//! - [`MatrixMinMax`]: Find minimum and maximum values per row/column
//! - [`BatchMatrixVariance`] & [`BatchMatrixMean`]: Batch-wise statistical operations
//! - [`MatrixNTop`]: Sum of top-n elements per row
//!
//! All operations support both regular and masked computations for selective analysis.

use std::collections::HashMap;
use std::ops::AddAssign;

use crate::utils::BatchIdentifier;
use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use single_utilities::traits::NumericOps;

pub mod csc;
pub mod csr;

/// Trait for counting non-zero elements in sparse matrices.
/// 
/// Provides methods to count non-zero elements per row or column, with support
/// for masked operations and in-place chunk processing for memory efficiency.
pub trait MatrixNonZero {
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    /// Calculate masked non-zero counts for columns
    fn nonzero_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    /// Calculate masked non-zero counts for rows
    fn nonzero_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;
}

/// Trait for summing elements in sparse matrices.
/// 
/// Provides methods to sum elements per row or column, including squared sums
/// and masked operations for selective computation.
pub trait MatrixSum {
    type Item: NumCast;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_col_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;

    /// Calculate masked sum for rows
    fn sum_row_masked<T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_col_squared<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_row_squared<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;
}

/// Trait for calculating variance in sparse matrices.
/// 
/// Computes sample variance per row or column with support for masked operations
/// and in-place chunk processing.
pub trait MatrixVariance {
    type Item: NumCast;

    fn var_col<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn var_row<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn var_row_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    /// Calculate masked variance for columns
    fn var_col_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;

    /// Calculate masked variance for rows
    fn var_row_masked<I, T>(&self, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;
}

/// Trait for finding minimum and maximum values in sparse matrices.
/// 
/// Efficiently computes min/max values per row or column with support for
/// in-place chunk processing to reduce memory allocation.
pub trait MatrixMinMax {
    type Item;

    fn min_max_col<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync;

    fn min_max_row<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync;

    fn min_max_col_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync;

    fn min_max_row_chunk<Item>(&self, reference: (&mut [Item], &mut [Item])) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps + Send + Sync;
}

/// Trait for batch-wise variance calculations in sparse matrices.
/// 
/// Enables computation of variance statistics grouped by batch identifiers,
/// useful for analyzing data with categorical groupings.
pub trait BatchMatrixVariance {
    type Item: NumCast;

    /// Calculate row-wise variance for each batch
    fn var_batch_row<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
        B: BatchIdentifier;

    /// Calculate column-wise variance for each batch
    fn var_batch_col<I, T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
        B: BatchIdentifier;
}

/// Trait for batch-wise mean calculations in sparse matrices.
/// 
/// Enables computation of mean statistics grouped by batch identifiers,
/// complementing variance calculations for comprehensive batch analysis.
pub trait BatchMatrixMean {
    type Item: NumCast;

    /// Calculate row-wise mean for each batch
    fn mean_batch_row<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
        B: BatchIdentifier;

    /// Calculate column-wise mean for each batch
    fn mean_batch_col<T, B>(&self, batches: &[B]) -> anyhow::Result<HashMap<B, Vec<T>>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync,
        B: BatchIdentifier;
}

/// Trait for top-N element operations in sparse matrices.
/// 
/// Provides methods to sum the top N largest elements per row,
/// useful for feature selection and dimension reduction algorithms.
pub trait MatrixNTop {
    type Item: NumCast;

    fn sum_row_n_top<T>(&self, n: usize) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum + Send + Sync;
}
