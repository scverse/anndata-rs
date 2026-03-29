# AnnData-RS

A high-performance, out-of-core implementation of the AnnData format for Rust and Python.

`anndata-rs` complements the original Python `anndata` package by providing an implementation designed for datasets that are too large to fit in memory. Unlike "backed mode" in other libraries, `anndata-rs` is built from the ground up for efficient partial I/O and cloud-native storage.

## Key Features

- **Multi-Backend Support**: Native support for **HDF5** (`.h5ad`) and **Zarr V3**.
- **Cloud Native**: Directly read and write Zarr data from **S3, HTTP, and Azure Blob Storage** via `object_store`.
- **Zero-Memory Loading**: Fields are loaded lazily. Opening a 100GB file consumes nearly zero bytes of RAM.
- **Lean Extraction**: Special `take()` and `drain()` semantics allow you to move data directly from disk into your ownership (e.g., into `anndata-memory`) without memory duplication.
- **High-Performance Sparse Matrices**: Optimized `sprs`-based backend with `u64` index pointers, supporting datasets with billions of non-zero elements.
- **Parallel I/O**: Multi-threaded reading and writing via `AnnDataSet`, utilizing Rayon for maximum throughput.

## Quick Start (Rust)

```rust
use anndata::AnnData;
use anndata_hdf5::H5;

// Open a file lazily
let adata = AnnData::<H5>::open(H5::open("data.h5ad")?)?;

// Load a subset of the X matrix into memory
let x = adata.x().slice::<ArrayData, _>(&[s![0..100], s![..]])?;
```

For detailed examples on cloud storage, subsetting, and memory-efficient loading, see the **[Usage Guide](USAGE.md)**.

## Installation

### Rust
Add to your `Cargo.toml`:
```toml
[dependencies]
anndata = "0.7"
anndata-hdf5 = "0.5" # For HDF5 support
anndata-zarr = "0.2" # For Zarr V3 support
```

### Python
```bash
pip install pyanndata
```

## Documentation
- **Usage Guide**: [USAGE.md](USAGE.md)
- **Tutorials**: [kzhang.org/epigenomics-analysis/anndata.html](https://kzhang.org/epigenomics-analysis/anndata.html)
- **API Reference**: [docs.rs/anndata](https://docs.rs/anndata)

## License
Licensed under the Apache License, Version 2.0.
