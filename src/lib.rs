pub mod dense;
pub mod sparse;
pub mod svd;

pub mod dimred;

#[cfg(feature = "clustering")]
pub mod clustering;

#[cfg(feature = "network")]
pub mod network;

#[cfg(feature = "local_moving")]
pub(crate) mod local_moving;

pub(crate) mod similarity;



mod utils;

pub use utils::Direction;
pub use utils::FloatOps;
pub use utils::Normalize;
pub use utils::NumericNormalize;
pub use utils::NumericOps;
pub use utils::Log1P;

