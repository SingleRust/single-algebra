pub mod dense;
pub mod sparse;
pub mod svd;

pub mod dimred;

//#[cfg(feature="correction")]
//pub mod correction;

mod utils;

pub use utils::Normalize;
pub use utils::Log1P;

