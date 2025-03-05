use crate::NumericOps;
use anyhow::Result;
use std::hash::Hash;

/// Trait for types that can be used to identify batches
pub trait BatchIdentifier: Clone + Eq + Hash {}

// Implement BatchIdentifier for common types
impl BatchIdentifier for String {}
impl BatchIdentifier for &str {}
impl BatchIdentifier for i32 {}
impl BatchIdentifier for u32 {}
impl BatchIdentifier for usize {}

/// Matrix trait required for batch correction operations
pub trait CorrectionMatrix {
    type Item;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    /// Extract a view of rows belonging to a specific batch
    fn batch_view<B: BatchIdentifier>(&self, batch_indices: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Create a new matrix with the same structure but different values
    fn with_same_structure(&self, values: Vec<Self::Item>) -> Result<Self> where Self: std::marker::Sized;
}

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
