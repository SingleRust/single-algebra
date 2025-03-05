use std::collections::HashMap;

/// Result from a single statistical test
#[derive(Debug, Clone)]
pub struct TestResult {
    /// The test statistic value (e.g., t-statistic, U statistic)
    pub statistic: f64,
    /// The p-value of the test
    pub p_value: f64,
    /// Confidence interval for the effect size/difference (if available)
    pub confidence_interval: Option<(f64, f64)>,
    /// Degrees of freedom (for parametric inference)
    pub degrees_of_freedom: Option<f64>,
    /// Effect size measurement
    pub effect_size: Option<f64>,
    /// Standard error of the effect size or test statistic
    pub standard_error: Option<f64>,
    /// Additional test-specific information
    pub metadata: HashMap<String, f64>,
}

impl TestResult {
    /// Create a new test result with minimal information
    pub fn new(statistic: f64, p_value: f64) -> Self {
        TestResult {
            statistic,
            p_value,
            confidence_interval: None,
            degrees_of_freedom: None,
            effect_size: None,
            standard_error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new test result with effect size
    pub fn with_effect_size(statistic: f64, p_value: f64, effect_size: f64) -> Self {
        TestResult {
            statistic,
            p_value,
            confidence_interval: None,
            degrees_of_freedom: None,
            effect_size: Some(effect_size),
            standard_error: None,
            metadata: HashMap::new(),
        }
    }

    /// Add confidence interval to the result
    pub fn with_confidence_interval(mut self, lower: f64, upper: f64) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }

    /// Add degrees of freedom to the result
    pub fn with_degrees_of_freedom(mut self, df: f64) -> Self {
        self.degrees_of_freedom = Some(df);
        self
    }

    /// Add standard error to the result
    pub fn with_standard_error(mut self, se: f64) -> Self {
        self.standard_error = Some(se);
        self
    }

    /// Add additional metadata
    pub fn with_metadata(mut self, key: &str, value: f64) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Check if the result is statistically significant at the given threshold
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

#[derive(Debug, Clone)]
pub struct MultipleTestResults {
    /// Test statistics for each feature/gene
    pub statistics: Vec<f64>,
    /// Raw (unadjusted) p-values
    pub p_values: Vec<f64>,
    /// Adjusted p-values (after multiple testing correction)
    pub adjusted_p_values: Option<Vec<f64>>,
    /// Effect sizes (if calculated)
    pub effect_sizes: Option<Vec<f64>>,
    /// Confidence intervals (if calculated)
    pub confidence_intervals: Option<Vec<(f64, f64)>>,
    /// Feature-specific metadata
    pub feature_metadata: Option<Vec<HashMap<String, f64>>>,
    /// Global metadata about the test
    pub global_metadata: HashMap<String, String>,
}

impl MultipleTestResults {
    /// Create a new results object from p-values
    pub fn new(statistics: Vec<f64>, p_values: Vec<f64>) -> Self {
        MultipleTestResults {
            statistics,
            p_values,
            adjusted_p_values: None,
            effect_sizes: None,
            confidence_intervals: None,
            feature_metadata: None,
            global_metadata: HashMap::new(),
        }
    }

    /// Add adjusted p-values to the results
    pub fn with_adjusted_p_values(mut self, adjusted_p_values: Vec<f64>) -> Self {
        self.adjusted_p_values = Some(adjusted_p_values);
        self
    }

    /// Add effect sizes to the results
    pub fn with_effect_sizes(mut self, effect_sizes: Vec<f64>) -> Self {
        self.effect_sizes = Some(effect_sizes);
        self
    }

    /// Add confidence intervals to the results
    pub fn with_confidence_intervals(mut self, confidence_intervals: Vec<(f64, f64)>) -> Self {
        self.confidence_intervals = Some(confidence_intervals);
        self
    }

    /// Add global metadata about the test
    pub fn with_global_metadata(mut self, key: &str, value: &str) -> Self {
        self.global_metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get indices of significant features at the given threshold
    pub fn significant_indices(&self, alpha: f64) -> Vec<usize> {
        match &self.adjusted_p_values {
            Some(adj_p_values) => adj_p_values.iter()
                .enumerate()
                .filter_map(|(i, &p)| if p < alpha { Some(i) } else { None })
                .collect(),
            None => self.p_values.iter()
                .enumerate()
                .filter_map(|(i, &p)| if p < alpha { Some(i) } else { None })
                .collect(),
        }
    }

    /// Get the number of significant features at the given threshold
    pub fn num_significant(&self, alpha: f64) -> usize {
        self.significant_indices(alpha).len()
    }

    /// Get top n features by p-value
    pub fn top_features(&self, n: usize) -> Vec<usize> {
        let p_values = match &self.adjusted_p_values {
            Some(adj_p) => adj_p,
            None => &self.p_values,
        };

        let mut indices: Vec<usize> = (0..p_values.len()).collect();
        indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(n);
        indices
    }
}






