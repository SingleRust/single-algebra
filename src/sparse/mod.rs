use std::ops::AddAssign;

use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};

use crate::NumericOps;

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
}

pub trait MatrixSum {
    type Item;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast;

    fn sum_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast;

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast;

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast;
}

pub trait MatrixVariance {
    type Item;

    fn var_col<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast + MatrixSum + MatrixNonZero;

    fn var_row<I, T>(&self) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast + MatrixSum + MatrixNonZero;

    fn var_col_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast;

    fn var_row_chunk<I, T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast;
}

pub trait MatrixMinMax {
    type Item;

    fn min_max_col<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_row<Item>(&self) -> anyhow::Result<(Vec<Item>, Vec<Item>)>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_col_chunk<Item>(&self, reference: (&mut Vec<Item>, &mut Vec<Item>)) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_row_chunk<Item>(&self, reference: (&mut Vec<Item>, &mut Vec<Item>)) -> anyhow::Result<()>
    where
        Item: NumCast + Copy + PartialOrd + NumericOps;
}
