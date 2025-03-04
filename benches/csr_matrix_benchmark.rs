use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use rand::{rngs::StdRng, SeedableRng};
use single_algebra::sparse::{MatrixNonZero, MatrixSum};
use std::time::Duration;
use rand::distr::{Distribution, Uniform};

#[derive(Clone)]
pub struct SparseMatrixConfig {
    seed: u64,
    matrix_sizes: Vec<(usize, usize)>,
    densities: Vec<f64>,
    measurement_time: u64,
    sample_size: usize,
}

impl Default for SparseMatrixConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            matrix_sizes: vec![
                (100, 100),
                (1000, 1000),
                (5000, 5000),
                (10000, 10000),
                (100000, 50000),
                (500000, 50000),
            ],
            densities: vec![0.01, 0.1],
            measurement_time: 10,
            sample_size: 10,
        }
    }
}
fn create_test_matrix(rows: usize, cols: usize, density: f64, seed: u64) -> CsrMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut coo = CooMatrix::new(rows, cols);
    let total_elements = (rows * cols) as f64 * density;
    let value_dist = Uniform::try_from(0.0..1.0).unwrap();
    let row_dist = Uniform::try_from(0..rows).unwrap();
    let col_dist = Uniform::try_from(0..cols).unwrap();

    for _ in 0..total_elements as usize {
        let row = row_dist.sample(&mut rng);
        let col = col_dist.sample(&mut rng);
        let value = value_dist.sample(&mut rng);
        coo.push(row, col, value);
    }

    (&coo).into()
}

fn configure_group<'a, M: Measurement>(
    c: &'a mut Criterion<M>,
    name: &str,
    config: &SparseMatrixConfig,
) -> BenchmarkGroup<'a, M> {
    let mut group = c.benchmark_group(name);
    group.measurement_time(Duration::from_secs(config.measurement_time));
    group.sample_size(config.sample_size);
    group
}

pub fn bench_csr_nonzero_counts(c: &mut Criterion) {
    let config = SparseMatrixConfig::default();
    let mut group = configure_group(c, "CSR_Nonzero_Counts", &config);

    for &(rows, cols) in config.matrix_sizes.iter() {
        for &density in config.densities.iter() {
            let seed = config.seed + (rows * cols) as u64;
            let matrix = create_test_matrix(rows, cols, density, seed);

            // Column counts
            group.bench_with_input(
                BenchmarkId::new("col_count", format!("{}x{}_d{}", rows, cols, density)),
                &(rows, cols, density),
                |b, _| {
                    b.iter(|| matrix.nonzero_col::<u32>().unwrap());
                },
            );

            // Row counts
            group.bench_with_input(
                BenchmarkId::new("row_count", format!("{}x{}_d{}", rows, cols, density)),
                &(rows, cols, density),
                |b, _| {
                    b.iter(|| matrix.nonzero_row::<u32>().unwrap());
                },
            );
        }
    }
    group.finish();
}

pub fn bench_csr_sums(c: &mut Criterion) {
    let config = SparseMatrixConfig::default();
    let mut group = configure_group(c, "CSR_Sum_Operations", &config);

    for &(rows, cols) in config.matrix_sizes.iter() {
        for &density in config.densities.iter() {
            let seed = config.seed + (rows * cols) as u64;
            let matrix = create_test_matrix(rows, cols, density, seed);

            // Column sums
            group.bench_with_input(
                BenchmarkId::new("col_sum", format!("{}x{}_d{}", rows, cols, density)),
                &(rows, cols, density),
                |b, _| {
                    b.iter(|| matrix.sum_col::<f64>().unwrap());
                },
            );

            // Row sums
            group.bench_with_input(
                BenchmarkId::new("row_sum", format!("{}x{}_d{}", rows, cols, density)),
                &(rows, cols, density),
                |b, _| {
                    b.iter(|| matrix.sum_row::<f64>().unwrap());
                },
            );
        }
    }
    group.finish();
}
// bench_csr_nonzero_counts
criterion_group!(csr_benches, bench_csr_sums);
criterion_main!(csr_benches);
