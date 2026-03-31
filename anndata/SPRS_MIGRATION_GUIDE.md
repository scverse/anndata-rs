# Migration Guide: `nalgebra-sparse` to `sprs`

This document outlines the architectural changes, performance optimizations, and usage instructions following the complete migration of the `anndata-rs` sparse matrix backend from `nalgebra-sparse` to `sprs`.

## 1. Architectural Changes

### Unified Sparse Matrix Representation
Previously, `anndata-rs` maintained separate definitions and parallel structures for `DynCsrMatrix` and `DynCscMatrix`. 
Because the `sprs` library utilizes a unified `CsMatI` struct that tracks its layout (CSR vs. CSC) internally, we have removed `DynCsrMatrix` and `DynCscMatrix` from `src/data/array/sparse/dynamic.rs`. 

The single source of truth for dynamic sparse matrices is now `DynIndSparseMatrix`, which encapsulates matrix types supporting generic element types (`i8` to `f64`, `bool`, `String`) and specifically uses index types (`i16`, `i32`, `i64`, `u16`, `u32`, `u64`).

### Fixed Index Pointers (`indptr`) vs. Dynamic Indices
To align with backend capabilities and ensure robust handling of massive datasets (>2 billion non-zero elements), the library enforces `u64` for all index pointer (`indptr`) vectors in `sprs::CsMatI`. However, the column/row indices remain dynamic (`i32`, `i64`, `u32`, etc.).

### Scipy Compatibility
When serializing matrices to disk (e.g., HDF5 or Zarr), index arrays are automatically cast to `i32` or `i64` if possible. This maintains direct binary compatibility with Python's `scipy.sparse` modules.

### `ArrayData` Variants Preserved
To maintain a clean and ergonomic API for downstream consumers, the top-level `ArrayData` enum retains its distinct variants:
*   `ArrayData::CsrMatrix(DynIndSparseMatrix)`
*   `ArrayData::CscMatrix(DynIndSparseMatrix)`
*   `ArrayData::CsrNonCanonical(DynCsrNonCanonical)`

Note: There is no `CscNonCanonical` variant. Non-canonical operations are presently supported primarily for CSR layout arrays.

Conversions seamlessly route the internal layout of the `DynIndSparseMatrix` (via `get_sparse_layout()`) to the correct `ArrayData` variant.

### Non-Canonical & COO Parsing
The legacy `nalgebra_sparse::coo::CooMatrix` usage has been replaced with `sprs::TriMatI`. 
The `CsrNonCanonical` type natively consumes `sprs::TriMatI` for parsing non-canonical triplets with duplicate entries. Its `.canonicalize()` method directly yields an `sprs::CsMatI`. 

## 2. Performance and Memory Optimizations

### Zero-Allocation Chunked Writes
In `src/data/array/chunks.rs`, the `write_by_chunk` implementation for sparse matrices was rewritten to eliminate heap allocations within the inner loop. We now utilize persistent workspace buffers (`indptr_workspace` and `indices_workspace`) that are cleared and re-populated for each chunk, drastically reducing GC/allocator overhead.

### `O(1)` Memory Footprint for Row Pointers
Previously, chunk writing accumulated the entire `indptr` vector in RAM before writing to the backend. We have transitioned `indptr` to utilize `ExtendableDataset`. Row pointers are now dynamically offset and streamed directly to disk chunk-by-chunk. This guarantees that writing multi-gigabyte sparse matrices maintains an `O(1)` memory footprint.

### Fast-Path Monotonic Selections and `SmallVec`
When selecting slices of sparse matrices, the code now checks for monotonic, continuously increasing bounds. If detected, it bypasses costly sorting operations. Furthermore, the lookup mechanisms utilize `smallvec::SmallVec` to mitigate heap allocations for unique or low-duplicate indices.

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
// Note the explicit signature specifying the data type (f64), index type (u32), and indptr type (u64)
let csr_matrix: CsMatI<f64, u32, u64> = CsMatI::new(
    (3, 3), 
    vec![0, 1, 2, 3], 
    vec![0, 1, 2], 
    vec![1.0, 2.0, 3.0]
);

// Convert automatically to ArrayData
let data: ArrayData = csr_matrix.into();
// `data` is now strictly an ArrayData::CsrMatrix

// Create a CSC matrix
let csc_matrix: CsMatI<i32, i64, u64> = CsMatI::new_csc(
    (3, 3),
    vec![0, 1, 2, 3],
    vec![0, 1, 2],
    vec![10, 20, 30]
);
let csc_data: ArrayData = csc_matrix.into();
// `csc_data` is strictly an ArrayData::CscMatrix
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

### Note on Integration Tests and Python Bindings
If you maintain workspace members (like `anndata-test-utils` or Python bindings `pyanndata`), their data ingestion paths have been updated to match `sprs::CsMatI` and any legacy references to `nalgebra_sparse::CsrMatrix` have been eliminated. All internal generic bounds previously referencing `CsrMatrix` have been updated to target `CsMatI`.
