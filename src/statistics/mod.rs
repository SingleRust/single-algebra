pub mod correction;
pub mod effect;
mod types;
pub mod utils;
pub mod inference;

pub use types::*;

#[derive(Debug, Clone, Copy)]
pub enum TestMethod {
    TTest(TTestType),
    MannWhitney,
    NegativeBinomial,
    ZeroInflated,
}

#[derive(Debug, Clone, Copy)]
pub enum TTestType {
    Student, // Equal variance
    Welch,   // Unequal variance
}

#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}
