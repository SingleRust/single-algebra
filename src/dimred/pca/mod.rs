use rayon::prelude::*;

pub mod dense;
mod sparse;

mod sparse_masked;
pub use sparse::SparsePCA;
pub use sparse::SparsePCABuilder;
pub use sparse_masked::MaskedSparsePCA;
pub use sparse_masked::MaskedSparsePCABuilder;
