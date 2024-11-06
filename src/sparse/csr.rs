use std::ops::AddAssign;

use anyhow::{anyhow, Ok};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, PrimInt, Unsigned, Zero};

use crate::NumericOps;

use super::{MatrixNonZero, MatrixSum};

impl<M: NumericOps> MatrixNonZero for CsrMatrix<M> {
    fn nonzero_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let mut result = vec![T::zero(); self.ncols()];
        for &col_index in self.col_indices() {
            result[col_index] += T::one();
        }
        Ok(result)
    }

    fn nonzero_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let data = self
            .row_offsets()
            .windows(2)
            .map(|window| {
                let diff = window[1]
                    .checked_sub(window[0])
                    .ok_or_else(|| anyhow!("Subtraction overflow"))
                    .expect("Subtraction overflow");
                T::from(diff)
                    .ok_or_else(|| anyhow!("Failed to convert to target type"))
                    .expect("Failed to convert to targat type")
            })
            .collect();
        Ok(data)
    }

    fn nonzero_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for &col_index in self.col_indices() {
            if col_index < reference.len() {
                reference[col_index] += T::one();
            }
        }
        Ok(())
    }

    fn nonzero_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        for (i, window) in self.row_offsets().windows(2).enumerate() {
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
}

impl<M: NumericOps> MatrixSum for CsrMatrix<M> {
    type Item = M;

    fn sum_col<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast,
    {
        let mut result = vec![T::zero(); self.ncols()];
        for (&col_indices, &value) in self.col_indices().iter().zip(self.values().iter()) {
            result[col_indices] += T::from(value).unwrap();
        }

        Ok(result)
    }

    fn sum_row<T>(&self) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast,
    {
        let mut result = vec![T::zero(); self.nrows()];
        for (row, row_vec) in self.row_iter().enumerate() {
            result[row] = row_vec.values().iter().map(|&v| T::from(v).unwrap()).sum();
        }
        Ok(result)
    }

    fn sum_col_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast,
    {
        for (&col_index, &value) in self.col_indices().iter().zip(self.values().iter()) {
            if col_index < reference.len() {
                reference[col_index] += T::from(value).unwrap();
            }
        }
        Ok(())
    }

    fn sum_row_chunk<T>(&self, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast+ AddAssign + std::iter::Sum,
        Self::Item: num_traits::NumCast,
    {
        for (row, row_vec) in self.row_iter().enumerate() {
            reference[row] = row_vec.values().iter().map(|&v| T::from(v).unwrap()).sum();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    fn create_test_matrix() -> CscMatrix<f64> {
        // Create a 4x3 sparse matrix with the following structure:
        // [1 0 2]
        // [0 0 0]
        // [3 4 0]
        // [0 5 6]

        // First create a COO matrix and then convert to CSC
        let mut coo = CooMatrix::new(4, 3);
        coo.push(0, 0, 1.0); // First column
        coo.push(2, 0, 3.0);

        coo.push(2, 1, 4.0); // Second column
        coo.push(3, 1, 5.0);

        coo.push(0, 2, 2.0); // Third column
        coo.push(3, 2, 6.0);

        CscMatrix::from(&coo)
    }

    #[test]
    fn test_nonzero_col() {
        let matrix = create_test_matrix();
        let result: Vec<u32> = matrix.nonzero_col().unwrap();

        // Expected number of nonzero elements in each column
        assert_eq!(result, vec![2, 2, 2]);
    }

    #[test]
    fn test_nonzero_row() {
        let matrix = create_test_matrix();
        let result: Vec<u32> = matrix.nonzero_row().unwrap();

        // Expected number of nonzero elements in each row
        assert_eq!(result, vec![2, 0, 2, 2]);
    }

    #[test]
    fn test_nonzero_col_chunk() {
        let matrix = create_test_matrix();
        let mut reference = vec![0u32; 4];
        matrix.nonzero_col_chunk(&mut reference).unwrap();

        // Only first 3 elements should be modified (matrix has 3 columns)
        assert_eq!(reference, vec![2, 2, 2, 0]);
    }

    #[test]
    fn test_nonzero_row_chunk() {
        let matrix = create_test_matrix();
        let mut reference = vec![0u32; 3];
        matrix.nonzero_row_chunk(&mut reference).unwrap();

        // Should only count nonzeros for rows within reference length
        assert_eq!(reference, vec![2, 0, 2]);
    }

    #[test]
    fn test_empty_matrix() {
        let matrix: CscMatrix<f64> = CscMatrix::zeros(0, 0);

        // Test empty matrix handling
        assert!(matrix.nonzero_col::<u32>().unwrap().is_empty());
        assert!(matrix.nonzero_row::<u32>().unwrap().is_empty());

        let mut empty_ref: Vec<u32> = Vec::new();
        assert!(matrix.nonzero_col_chunk(&mut empty_ref).is_ok());
        assert!(matrix.nonzero_row_chunk(&mut empty_ref).is_ok());
    }

    #[test]
    fn test_different_integer_types() {
        let matrix = create_test_matrix();

        // Test with u8
        let result_u8: Vec<u8> = matrix.nonzero_col().unwrap();
        assert_eq!(result_u8, vec![2, 2, 2]);

        // Test with u64
        let result_u64: Vec<u64> = matrix.nonzero_col().unwrap();
        assert_eq!(result_u64, vec![2, 2, 2]);
    }

    #[test]
    fn test_large_sparse_matrix() {
        // Create a larger sparse matrix to test potential overflow conditions
        let mut coo = CooMatrix::new(1000, 1000);

        // Add some sparse data
        for i in 0..999 {
            coo.push(i, i, 1.0);
            coo.push(i + 1, i, 1.0);
        }

        let matrix = CscMatrix::from(&coo);
        let result: Vec<u32> = matrix.nonzero_col().unwrap();
        assert_eq!(result.len(), 1000);

        // Most columns should have 2 nonzero elements
        assert_eq!(result[500], 2);
    }

    #[test]
    fn test_chunk_smaller_than_matrix() {
        let matrix = create_test_matrix();

        // Test with smaller reference slices
        let mut col_ref = vec![0u32; 2];
        matrix.nonzero_col_chunk(&mut col_ref).unwrap();
        assert_eq!(col_ref, vec![2, 2]);

        let mut row_ref = vec![0u32; 2];
        matrix.nonzero_row_chunk(&mut row_ref).unwrap();
        assert_eq!(row_ref, vec![2, 0]);
    }

    #[test]
    fn test_zero_matrix() {
        // Test a non-empty matrix with all zero elements
        let matrix: CscMatrix<f64> = CscMatrix::zeros(5, 4);

        let col_result: Vec<u32> = matrix.nonzero_col().unwrap();
        assert_eq!(col_result, vec![0, 0, 0, 0]);

        let row_result: Vec<u32> = matrix.nonzero_row().unwrap();
        assert_eq!(row_result, vec![0, 0, 0, 0, 0]);
    }
}
