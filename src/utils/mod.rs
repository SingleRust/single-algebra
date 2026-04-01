use std::hash::Hash;
use single_utilities::traits::FloatOpsTS;
use single_utilities::types::Direction;

pub trait Normalize<T: FloatOpsTS> {
    fn normalize<U: FloatOpsTS>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &Direction,
    ) -> anyhow::Result<()>;
}

pub trait Log1P<T: FloatOpsTS> {
    fn log1p_normalize(&mut self) -> anyhow::Result<()>;
}

/// Trait for types that can be used to identify batches
pub trait BatchIdentifier: Clone + Eq + Hash {}

// Implement BatchIdentifier for common types
impl BatchIdentifier for String {}
impl BatchIdentifier for &str {}
impl BatchIdentifier for i32 {}
impl BatchIdentifier for u32 {}
impl BatchIdentifier for usize {}
