use crate::statistics::{correction, utils, Alternative, MultipleTestResults, TTestType, TestMethod, TestResult};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, FromPrimitive, NumCast};

pub mod discrete;
pub mod nonparametric;
pub mod parametric;

pub trait MatrixStatTests<T>
where
    T: Float + NumCast + FromPrimitive + Send + Sync + std::fmt::Debug,
{
    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult>>;

    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult>>;

    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults>;
}

impl<T> MatrixStatTests<T> for CsrMatrix<T>
where
    T: Float + NumCast + FromPrimitive + Send + Sync + std::fmt::Debug,
    f64: std::convert::From<T>,
{
    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult>> {
        parametric::t_test_matrix_groups(
            self,
            group1_indices,
            group2_indices,
            test_type,
            alternative,
        )
    }

    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult>> {
        nonparametric::mann_whitney_matrix_groups(self, group1_indices, group2_indices, alternative)
    }

    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults> {
        let unique_groups = utils::extract_unique_groups(group_ids);
        if unique_groups.len() != 2 {
            return Err(anyhow::anyhow!(
                "Currently only two-group comparisons are supported"
            ));
        }

        let (group1_indices, group2_indices) = utils::get_group_indices(group_ids, &unique_groups);

        match test_method {
            TestMethod::TTest(test_type) => {
                // Run t-tests
                let results = self.t_test(
                    &group1_indices,
                    &group2_indices,
                    test_type,
                    Alternative::TwoSided,
                )?;

                // Extract statistics and p-values
                let statistics: Vec<_> = results.iter().map(|r| r.statistic).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value).collect();

                // Apply multiple testing correction
                let adjusted_p_values =
                    correction::benjamini_hochberg_correction(&p_values)?;

                // Extract effect sizes if available
                let effect_sizes = results
                    .iter()
                    .map(|r| r.effect_size)
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(|| vec![0.0; results.len()])
                    .into_iter()
                    .filter_map(|x| Option::from(x))
                    .collect::<Vec<_>>();

                let mut result = MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_global_metadata("test_type", "t_test");

                if !effect_sizes.is_empty() {
                    result = result.with_effect_sizes(effect_sizes);
                }

                Ok(result)
            }

            TestMethod::MannWhitney => {
                // Run Mann-Whitney tests
                let results = self.mann_whitney_test(
                    &group1_indices,
                    &group2_indices,
                    Alternative::TwoSided,
                )?;

                // Extract statistics and p-values
                let statistics: Vec<_> = results.iter().map(|r| r.statistic).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value).collect();

                // Apply multiple testing correction
                let adjusted_p_values =
                    correction::benjamini_hochberg_correction(&p_values)?;

                Ok(MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_global_metadata("test_type", "mann_whitney"))
            }

            // Implement other test methods similarly
            _ => Err(anyhow::anyhow!("Test method not implemented yet")),
        }
    }
}
