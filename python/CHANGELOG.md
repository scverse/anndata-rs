# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0](https://github.com/scverse/anndata-rs/releases/tag/anndata_rs-v0.6.0) - 2026-06-22

### Added

- 1GB zarr shard defaults ([#22](https://github.com/scverse/anndata-rs/pull/22))
- Implement default write configuration management and compression options in Python bindings
- add split_by
- support pyo3 0.23
- use zstd for compression
- infer backend from filename
- add zarr backend
- support native string array
- support numpy string array
- support noncanonical sparse matrix
- support CSC matrix
- Implement layers
- allow empty AnnDataSet
- implement obs_ix and var_ix

### Fixed

- fix to_memory with empty obs/var but non-empty obs_names/var_names
- fix categorical concat
- fix nullable arrays
- fix csc, csr matrices conversion issues
- fix zlib
- fix deadlock in displaying AnnDataSet
- fix StackAnnData indexing
- fix StackedDataFrameElem
- fix dataframe index io
- fix wrongly slicing
- fix a bug in set_obs and set_var
- fix AnnDataSet subsetting
- fix anndataset indexing
- fix varm
- fix column subsetting
- fix subsetting in AnnDataSet
- fix a bug in AnnDataSet chunking
- fix a bug in read_mtx
- fix subsetting
- ignore keys with incompatible dimensions during dataset creation
