use std::ops::Add;

use num_traits::{Bounded, NumCast, One, Zero};

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
