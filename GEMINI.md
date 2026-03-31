# single-algebra

## Project Overview
`single-algebra` is a high-performance linear algebra library for the `single-rust` ecosystem. It is optimized for sparse matrices (CSR/CSC) and dimensionality reduction algorithms such as Principal Component Analysis (PCA). The library is designed for machine learning, data analysis, and scientific computing applications where memory efficiency and performance with sparse data are critical.

Key features include:
- Sparse Matrix Operations (CSR/CSC formats)
- Advanced dimensionality reduction algorithms (Sparse PCA, Masked Sparse PCA, SVD)
- Data preprocessing (Normalization, Log1P transformations)
- High-performance, memory-efficient processing with multi-threading via Rayon

## Building and Running
This is a standard Rust project using Cargo.
- **Build**: `cargo build`
- **Test**: `cargo test`
- **Benchmark**: `cargo bench` (runs benchmarks in the `benches/` directory)
- **Lint**: `cargo clippy`
- **Format**: `cargo fmt`

## Development Conventions
- **Language**: Adhere to standard Rust coding styles and idioms.
- **Performance**: Focus on memory and computational efficiency, especially for large, high-dimensional sparse datasets (>90% zero values).
- **Benchmarking**: New performance-sensitive features or matrix operations should include benchmarks in the `benches/` directory.
- **Typing**: Support type flexibility, providing generic implementations for both `f32` and `f64` where applicable.
- **Documentation**: Provide clear, comprehensive rustdoc comments for public APIs, as this library serves as a public utility within `single-rust`.
