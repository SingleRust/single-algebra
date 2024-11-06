use std::ops::{AddAssign};

use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};

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
    fn sum_col<T>(&self) -> anyhow::Result<T>
    where
        T: Float;

    fn sum_row<T>(&self) -> anyhow::Result<T>
    where
        T: Float;

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float;

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float;
}

pub trait MatrixVariance {
    fn var_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float;

    fn var_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float;

    fn var_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float;

    fn var_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float;
}

pub trait MatrixMinMax {
    fn min_max_col<T>(&self) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: NumCast + Copy + PartialOrd;

    fn min_max_row<T>(&self) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: NumCast + Copy + PartialOrd;

    fn min_max_col_chunk<T>(&self, reference: &mut (Vec<T>, Vec<T>)) -> anyhow::Result<()>
    where
        T: NumCast + Copy + PartialOrd;

    fn min_max_row_chunk<T>(&self, reference: &mut (Vec<T>, Vec<T>)) -> anyhow::Result<()>
    where
        T: NumCast + Copy + PartialOrd;
}
