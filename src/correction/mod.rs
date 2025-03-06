use crate::utils::BatchIdentifier;
use crate::NumericOps;
use anyhow::Result;
use std::hash::Hash;
use std::ops::AddAssign;
use num_traits::{Float, NumCast};

/// Core trait for batch correction algorithms
pub trait BatchCorrection<T, B>
where
    T: NumericOps,
    B: BatchIdentifier,
{
    /// Fit the correction model to the data and batches
    fn fit(&mut self, data: &impl CorrectionMatrix<Item = T>, batches: &[B]) -> Result<()>;

    /// Apply correction to data using a previously fitted model
    fn transform(
        &self,
        data: &impl CorrectionMatrix<Item = T>,
    ) -> Result<impl CorrectionMatrix<Item = T>>;

    /// Fit the model and transform the data in a single operation
    fn fit_transform(
        &mut self,
        data: &impl CorrectionMatrix<Item = T>,
        batches: &[B],
    ) -> Result<impl CorrectionMatrix<Item = T>> {
        self.fit(data, batches)?;
        self.transform(data)
    }
}

pub trait CorrectionMatrix: Sized {
    type Item: NumericOps + NumCast;

    /// Center columns by subtracting column means
    fn center_columns<T>(&mut self, means: &[T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum;

    /// Center rows by subtracting row means
    fn center_rows<T>(&mut self, means: &[T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum;

    /// Scale columns by dividing by column scaling factors
    fn scale_columns<T>(&mut self, factors: &[T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum;

    /// Scale rows by dividing by row scaling factors
    fn scale_rows<T>(&mut self, factors: &[T]) -> anyhow::Result<()>
    where
        T: Float + NumCast + AddAssign + std::iter::Sum;

    /// Create a new matrix with the same dimensions and structure
    fn like(&self) -> Self;
}