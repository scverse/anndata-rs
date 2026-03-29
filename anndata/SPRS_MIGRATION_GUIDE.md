# Migration Guide: `nalgebra-sparse` to `sprs`

This document outlines the architectural changes, performance optimizations, and usage instructions following the complete migration of the `anndata-rs` sparse matrix backend from `nalgebra-sparse` to `sprs`.

## 1. Architectural Changes

### Unified Sparse Matrix Representation
Previously, `anndata-rs` maintained separate definitions and parallel structures for `DynCsrMatrix` and `DynCscMatrix`. 
Because the `sprs` library utilizes a unified `CsMatI` struct that tracks its layout (CSR vs. CSC) internally, we have removed `DynCsrMatrix` and `DynCscMatrix` from `src/data/array/sparse/dynamic.rs`. 

The single source of truth for dynamic sparse matrices is now `DynIndSparseMatrix`.

### `ArrayData` Variants Preserved
To maintain a clean and ergonomic API for downstream consumers, the top-level `ArrayData` enum retains its distinct variants:
*   `ArrayData::CsrMatrix(DynIndSparseMatrix)`
*   `ArrayData::CscMatrix(DynIndSparseMatrix)`

Conversions seamlessly route the internal layout of the `DynIndSparseMatrix` (via `get_sparse_layout()`) to the correct `ArrayData` variant.

### Non-Canonical & COO Parsing
The legacy `nalgebra_sparse::coo::CooMatrix` usage has been replaced with `sprs::TriMatI`. 
The `CsrNonCanonical` type natively consumes `sprs::TriMatI` for parsing non-canonical triplets with duplicate entries. Its `.canonicalize()` method directly yields an `sprs::CsMatI`.

## 2. Performance and Memory Optimizations

### Zero-Allocation Chunked Writes
In `src/data/array/chunks.rs`, the `write_by_chunk` implementation for sparse matrices was rewritten to eliminate heap allocations within the inner loop. We now utilize persistent workspace buffers (`indptr_workspace` and `indices_workspace`) that are cleared and re-populated for each chunk, drastically reducing GC/allocator overhead.

### `O(1)` Memory Footprint for Row Pointers
Previously, chunk writing accumulated the entire `indptr` vector in RAM before writing to the backend. We have transitioned `indptr` to utilize `ExtendableDataset`. Row pointers are now dynamically offset and streamed directly to disk chunk-by-chunk. This guarantees that writing multi-gigabyte sparse matrices maintains an `O(1)` memory footprint.

### Zero-Copy Reading Maintained
The `read` operations for sparse matrices in `sprs.rs` strictly respect the integer pointer lengths defined by the underlying storage (e.g., HDF5 or Zarr). We deliberately avoided applying generic type casts during reads. By maintaining strict structural alignment, extracting massive sparse matrices directly into `sprs::CsMatI` continues to utilize zero-copy memory extraction.

### Native CSC Chunk Writing
Chunked writing is now fully supported for CSC-layout `sprs::CsMatI` matrices, sharing the same optimized streaming logic as CSR.

## 3. How to Use the Project Now

### Dependencies
You no longer need to depend on `nalgebra-sparse` to interact with sparse matrices in `anndata-rs`. You must use `sprs` directly.

### Creating and Storing Sparse Matrices
To store a sparse matrix into an `AnnData` object, instantiate an `sprs::CsMatI` and simply call `.into()`. The library automatically infers the layout:

```rust
use sprs::CsMatI;
use anndata::data::ArrayData;

// Create a CSR matrix using sprs
let csr_matrix: CsMatI<f64, u32, u64> = CsMatI::new(
    (3, 3), 
    vec![0, 1, 2, 3], 
    vec![0, 1, 2], 
    vec![1.0, 2.0, 3.0]
);

// Convert automatically to ArrayData
let data: ArrayData = csr_matrix.into();
// `data` is now strictly an ArrayData::CsrMatrix
```

### Retrieving Sparse Matrices
To retrieve a matrix back out of `ArrayData`, use the standard `TryFrom` trait specifying the expected `sprs` types. The library handles unpacking the dynamic wrappers:

```rust
use sprs::CsMatI;

// Assuming `data` is an ArrayData variant pulled from AnnData
let extracted_matrix: CsMatI<f64, u32, u64> = CsMatI::try_from(data)
    .expect("Failed to extract CsMatI");

assert!(extracted_matrix.is_csr());
```

### Note on Integration Tests
If you maintain workspace members (like `anndata-test-utils` or Python bindings), be sure to update their data ingestion paths to match `sprs::CsMatI` and remove references to `nalgebra_sparse::CsrMatrix`. All internal generic bounds referencing `CsrMatrix` have been updated to target `CsMatI`.
