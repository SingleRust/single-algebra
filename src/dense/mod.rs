use anyhow::bail;
use ndarray::Array2;
use single_utilities::traits::FloatOpsTS;
use single_utilities::types::Direction;
use crate::{
    utils::{Normalize}
};

impl<T: FloatOpsTS> Normalize<T> for Array2<T> {
    fn normalize<U: FloatOpsTS>(
        &mut self,
        sums: &[U],
        target: U,
        direction: &Direction,
    ) -> anyhow::Result<()> {
        match direction {
            Direction::ROW => {
                if sums.len() != self.nrows() {
                    bail!(
                        "Length of sums ({}) does not match number of rows ({})",
                        sums.len(),
                        self.nrows()
                    );
                }

                for (i, row) in self.rows_mut().into_iter().enumerate() {
                    let scale = target / sums[i];
                    for val in row {
                        *val = T::from(U::from(*val).unwrap() * scale)
                            .ok_or_else(|| anyhow::anyhow!("Numeric conversion failed"))?;
                    }
                }
            }
            Direction::COLUMN => {
                if sums.len() != self.ncols() {
                    bail!(
                        "Length of sums ({}) does not match number of columns ({})",
                        sums.len(),
                        self.ncols()
                    );
                }

                for (j, col) in self.columns_mut().into_iter().enumerate() {
                    let scale = target / sums[j];
                    for val in col {
                        *val = T::from(U::from(*val).unwrap() * scale)
                            .ok_or_else(|| anyhow::anyhow!("Numeric conversion failed"))?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_normalize() {
        let mut arr = array![[1.0, 2.0], [3.0, 4.0]];
        let row_sums = vec![3.0, 7.0]; // Sum of each row
        let target = 1.0;

        arr.normalize(&row_sums, target, &Direction::ROW).unwrap();

        // Row normalized values should sum to target
        assert_relative_eq!(arr.row(0).sum(), target);
        assert_relative_eq!(arr.row(1).sum(), target);
        assert_relative_eq!(arr[[0, 0]], 1.0 / 3.0);
        assert_relative_eq!(arr[[0, 1]], 2.0 / 3.0);
        assert_relative_eq!(arr[[1, 0]], 3.0 / 7.0);
        assert_relative_eq!(arr[[1, 1]], 4.0 / 7.0);

        let mut arr = array![[1.0, 2.0], [3.0, 4.0]];
        let col_sums = vec![4.0, 6.0]; // Sum of each column

        arr.normalize(&col_sums, target, &Direction::COLUMN)
            .unwrap();

        // Column normalized values should sum to target
        assert_relative_eq!(arr.column(0).sum(), target);
        assert_relative_eq!(arr.column(1).sum(), target);
        assert_relative_eq!(arr[[0, 0]], 1.0 / 4.0);
        assert_relative_eq!(arr[[1, 0]], 3.0 / 4.0);
        assert_relative_eq!(arr[[0, 1]], 2.0 / 6.0);
        assert_relative_eq!(arr[[1, 1]], 4.0 / 6.0);
    }

    #[test]
    fn test_normalize_errors() {
        let mut arr = Array2::<f64>::zeros((2, 2));

        // Wrong size row sums
        assert!(arr.normalize(&[1.0], 1.0, &Direction::ROW).is_err());

        // Wrong size column sums
        assert!(arr.normalize(&[1.0], 1.0, &Direction::COLUMN).is_err());
    }
}
