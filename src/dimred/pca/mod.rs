mod sparse;

mod sparse_masked;

pub use sparse::SparsePCA;
pub use sparse::SparsePCABuilder;
pub use sparse_masked::MaskedSparsePCA;
pub use sparse_masked::MaskedSparsePCABuilder;
pub use single_svdlib::randomized::PowerIterationNormalizer;
pub use single_svdlib::SvdFloat;

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

