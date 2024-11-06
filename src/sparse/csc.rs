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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CscMatrix;

    fn create_test_matrix() -> CscMatrix<f64> {
        // Create a 4x3 CSC matrix directly with the following structure:
        // [1 0 2]
        // [0 0 0]
        // [3 4 0]
        // [0 5 6]
        
        // For CSC format we need:
        // 1. Values in column-major order
        // 2. Row indices for each value
        // 3. Column pointers indicating where each column starts
        
        let values = vec![1.0, 3.0, 4.0, 5.0, 2.0, 6.0];
        let row_indices = vec![0, 2, 2, 3, 0, 3];
        let col_ptrs = vec![0, 2, 4, 6];
        
        CscMatrix::try_from_csc_data(4, 3, col_ptrs, row_indices, values).unwrap()
    }


    #[test]
    fn test_nonzero_row() {
        let matrix = create_test_matrix();
        let result: Vec<u32> = matrix.nonzero_row().unwrap();
        
        // Row-wise nonzero counts
        assert_eq!(result, vec![2, 0, 2, 2]);
        
        // Verify through column iteration
        let mut row_counts = vec![0u32; matrix.nrows()];
        for col_idx in 0..matrix.ncols() {
            for &row_idx in matrix.col(col_idx).row_indices() {
                row_counts[row_idx] += 1;
            }
        }
        assert_eq!(result, row_counts);
    }

    #[test]
    fn test_empty_and_zero_matrices() {
        // Empty matrix
        let empty: CscMatrix<f64> = CscMatrix::zeros(0, 0);
        assert!(empty.nonzero_col::<u32>().unwrap().is_empty());
        assert!(empty.nonzero_row::<u32>().unwrap().is_empty());
        
        // Zero matrix
        let zero: CscMatrix<f64> = CscMatrix::zeros(4, 3);
        assert_eq!(zero.nonzero_col::<u32>().unwrap(), vec![0, 0, 0]);
        assert_eq!(zero.nonzero_row::<u32>().unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_large_sparse_matrix() {
        let n = 1000;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0];
        let mut current_nnz = 0;
        
        // Create a tridiagonal matrix in CSC format
        for col in 0..n {
            if col > 0 {
                values.push(1.0);
                row_indices.push(col - 1);
                current_nnz += 1;
            }
            values.push(2.0);
            row_indices.push(col);
            current_nnz += 1;
            if col < n - 1 {
                values.push(1.0);
                row_indices.push(col + 1);
                current_nnz += 1;
            }
            col_ptrs.push(current_nnz);
        }
        
        let matrix = CscMatrix::try_from_csc_data(
            n, n, col_ptrs, row_indices, values
        ).unwrap();
        
        // Verify structure
        assert_eq!(matrix.nrows(), n);
        assert_eq!(matrix.ncols(), n);
        
        // Test column nonzeros
        let col_nnz: Vec<u32> = matrix.nonzero_col().unwrap();
        assert_eq!(col_nnz[0], 2); // First column: 2 nonzeros
        assert_eq!(col_nnz[n/2], 3); // Middle columns: 3 nonzeros
        assert_eq!(col_nnz[n-1], 2); // Last column: 2 nonzeros
    }

    #[test]
    fn test_chunk_operations() {
        let matrix = create_test_matrix();
        
        // Test column chunks
        let mut col_chunk = vec![0u32; 2];
        matrix.nonzero_col_chunk(&mut col_chunk).unwrap();
        assert_eq!(col_chunk, vec![2, 2]);
        
        // Test row chunks
        let mut row_chunk = vec![0u32; 3];
        matrix.nonzero_row_chunk(&mut row_chunk).unwrap();
        assert_eq!(row_chunk, vec![2, 0, 2]);
    }
}
