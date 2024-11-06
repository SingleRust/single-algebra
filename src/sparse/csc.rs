use std::ops::AddAssign;

use nalgebra_sparse::CscMatrix;
use num_traits::{PrimInt, Unsigned, Zero};

use crate::NumericOps;

use anyhow::anyhow;

use super::MatrixNonZero;

impl<M: NumericOps> MatrixNonZero for CscMatrix<M> {
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let data = self
            .col_offsets()
            .windows(2)
            .map(|window| {
                let diff = window[1]
                    .checked_sub(window[0])
                    .ok_or_else(|| anyhow!("Subtraction overflow")).expect("Subtraction overflow");
                T::from(diff).ok_or_else(|| anyhow!("Failed to convert to target type")).expect("Failed to convert to a target type")
            })
            .collect();
        Ok(data)
    }

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let mut result = vec![T::zero(); self.nrows()];
        for &row_index in self.row_indices() {
            result[row_index] += T::one();
        }
        Ok(result)
    }

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for (i, window) in self.col_offsets().windows(2).enumerate() {
            let count = window[1]
                .checked_sub(window[0])
                .ok_or_else(|| anyhow!("Subtraction overflow"))?;
            let count_transformed =
                T::from(count).ok_or_else(|| anyhow!("Failed to convert to target type"))?;
            if i < reference.len() {
                reference[i] += count_transformed;
            } 
        }
        Ok(())
    }

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for &row_index in self.row_indices() {
            if row_index < reference.len() {
                reference[row_index] += T::one();
            }
        }
        Ok(())
    }
}
