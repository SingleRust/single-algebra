use std::hash::Hash;
use std::iter::repeat;
use std::ops::{Add, AddAssign};

use num_traits::{AsPrimitive, Bounded, NumCast, One, Zero};

pub enum Direction {
    COLUMN,
    ROW,
}

pub trait NumericOps:
    Zero + One + NumCast + Copy + std::ops::AddAssign + PartialOrd + Bounded + Add<Output = Self>
{
}
impl<
        T: Zero
            + One
            + NumCast
            + Copy
            + std::ops::AddAssign
            + PartialOrd
            + Bounded
            + Add<Output = Self>,
    > NumericOps for T
{
}

pub trait FloatOps: NumericOps + num_traits::Float {}
impl<T: NumericOps + num_traits::Float> FloatOps for T {}

// Define a type alias for our numeric constraints
pub trait NumericNormalize:
    num_traits::Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast
{
}

// Blanket implementation for any type that satisfies the bounds
impl<T> NumericNormalize for T where
    T: num_traits::Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast
{
}

pub trait Normalize<T: NumericNormalize> {
    fn normalize<U: NumericNormalize>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &Direction,
    ) -> anyhow::Result<()>;
}

pub trait Log1P<T: NumericNormalize> {
    fn log1p_normalize(&mut self) -> anyhow::Result<()>;
}

pub trait ZeroVec {
    fn zero_len(&mut self, len: usize);
}

impl<T: Default + Clone> ZeroVec for Vec<T> {
    fn zero_len(&mut self, len: usize) {
        self.clear();
        self.reserve(len);
        self.extend(repeat(T::default()).take(len));
    }
}

/// Trait for types that can be used to identify batches
pub trait BatchIdentifier: Clone + Eq + Hash {}

// Implement BatchIdentifier for common types
impl BatchIdentifier for String {}
impl BatchIdentifier for &str {}
impl BatchIdentifier for i32 {}
impl BatchIdentifier for u32 {}
impl BatchIdentifier for usize {}

