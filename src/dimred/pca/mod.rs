pub mod dense;
mod sparse;

mod sparse_masked;

use single_svdlib::randomized::PowerIterationNormalizer;
pub use sparse::SparsePCA;
pub use sparse::SparsePCABuilder;
pub use sparse_masked::MaskedSparsePCA;
pub use sparse_masked::MaskedSparsePCABuilder;


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVDMethod {
    Lanczos,
    Random {
        n_oversamples: usize,
        n_power_iterations: usize,
        normalizer: PowerIterationNormalizer,
    },
}

impl Default for SVDMethod {
    fn default() -> Self {
        Self::Lanczos
    }
}

