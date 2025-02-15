// Based on https://medium.com/@gshriya195/top-5-distance-similarity-measures-implementation-in-machine-learning-1f68b9ecb0a3
use std::f64::EPSILON;
use ndarray::ArrayView1;
use num_traits::{Float, FromPrimitive, ToPrimitive};

pub trait SimilarityMeasure {
    fn calculate<T>(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Float + FromPrimitive + ToPrimitive;
}

pub struct CosineSimilarity;

impl SimilarityMeasure for CosineSimilarity {
    fn calculate<T>(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Float + FromPrimitive + ToPrimitive,
    {
        let mut dot_product = T::zero();
        let mut norm_a = T::zero();
        let mut norm_b = T::zero();

        for i in 0..a.len() {
            dot_product = dot_product + a[i] * b[i];
            norm_a = norm_a + a[i] * a[i];
            norm_b = norm_b + b[i] * b[i];
        }

        let norm_product = (norm_a * norm_b).sqrt();
        if norm_product > T::epsilon() {
            (dot_product / norm_product).to_f64().unwrap()
        } else {
            0.0
        }
    }
}

pub struct EuclideanSimilarity {
    gamma: f64, // Parameter for distance to similarity conversion
}

impl EuclideanSimilarity {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl Default for EuclideanSimilarity {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

impl SimilarityMeasure for EuclideanSimilarity {
    fn calculate<T>(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Float + FromPrimitive + ToPrimitive,
    {
        let mut squared_dist = T::zero();
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            squared_dist = squared_dist + diff * diff;
        }
        let dist = squared_dist.sqrt().to_f64().unwrap();
        (-self.gamma * dist).exp()
    }
}

pub struct PearsonSimilarity;

impl SimilarityMeasure for PearsonSimilarity {
    fn calculate<T>(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Float + FromPrimitive + ToPrimitive,
    {
        let n = T::from_usize(a.len()).unwrap();
        let mut sum_a = T::zero();
        let mut sum_b = T::zero();
        let mut sum_ab = T::zero();
        let mut sum_a_sq = T::zero();
        let mut sum_b_sq = T::zero();

        for i in 0..a.len() {
            sum_a = sum_a + a[i];
            sum_b = sum_b + b[i];
            sum_ab = sum_ab + a[i] * b[i];
            sum_a_sq = sum_a_sq + a[i] * a[i];
            sum_b_sq = sum_b_sq + b[i] * b[i];
        }

        let numerator = sum_ab - (sum_a * sum_b) / n;
        let denominator =
            ((sum_a_sq - (sum_a * sum_a) / n) * (sum_b_sq - (sum_b * sum_b) / n)).sqrt();

        if denominator > T::epsilon() {
            (numerator / denominator).to_f64().unwrap()
        } else {
            0.0
        }
    }
}

pub struct ManhattanSimilarity {
    gamma: f64,
}

impl ManhattanSimilarity {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl Default for ManhattanSimilarity {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

impl SimilarityMeasure for ManhattanSimilarity {
    fn calculate<T>(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Float + FromPrimitive + ToPrimitive,
    {
        let mut dist = T::zero();
        for i in 0..a.len() {
            dist = dist + (a[i] - b[i]).abs();
        }
        (-self.gamma * dist.to_f64().unwrap()).exp()
    }
}

pub struct JaccardSimilarity {
    threshold: f64, // Threshold for considering values as equal
}

impl JaccardSimilarity {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for JaccardSimilarity {
    fn default() -> Self {
        Self { threshold: EPSILON }
    }
}

impl SimilarityMeasure for JaccardSimilarity {
    fn calculate<T>(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Float + FromPrimitive + ToPrimitive,
    {
        let mut intersection = 0;
        let mut union = 0;

        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs().to_f64().unwrap();
            if diff < self.threshold {
                intersection += 1;
            }
            if a[i] > T::zero() || b[i] > T::zero() {
                union += 1;
            }
        }

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
}
