# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0](https://github.com/scverse/anndata-rs/releases/tag/anndata_rs-v0.6.0) - 2026-06-22

### Added

- 1GB zarr shard defaults ([#22](https://github.com/scverse/anndata-rs/pull/22))

### Fixed

- fix tests
- fix tests
- fix to_memory with empty obs/var but non-empty obs_names/var_names
- fix tests
- fix tests
- fix tests
- fix categorical concat
- fix nullable arrays
- fix tests
- fix csc, csr matrices conversion issues
- fix zlib
- fix tests
- fix deadlock in displaying AnnDataSet
- fix StackAnnData indexing
- fix StackedDataFrameElem
- fix dataframe index io
- fix tests
- fix wrongly slicing
- fix a bug in set_obs and set_var
- fix AnnDataSet subsetting
- fix tests
- fix anndataset indexing
- fix varm
- fix column subsetting
- fix subsetting in AnnDataSet
- fix a bug in AnnDataSet chunking
- fix a bug in read_mtx
- fix subsetting

### Other

- fix deps for release
- format and clippy ([#23](https://github.com/scverse/anndata-rs/pull/23))
- Merge branch 'pyo3-0.27' into main
- Implement default write configuration management and compression options in Python bindings
- Allow any index type when creating sparse matrix ([#18](https://github.com/scverse/anndata-rs/pull/18))
- refactoring
- add split_by
- drop support for zarr backend
- modify ci
- minor fix
- upgrade pyo3 and polars
- support pyo3 0.23
- try zst
- compression algorithm except deflate doesn't work with strings
- minor
- use zstd for compression
- upgrade polars
- add concat test
- infer backend from filename
- add zarr backend
- bug fix
- refactoring
- upgrade pyo3
- upgrade dependencies
- update dependency bound
- upgrade pyo3
- minor
- bump polars version
- ignore keys with imcompatible dimensions during dataset creation
- support native string array
- support numpy string array
- bump dependency versions
- bump dependency versions
- support noncanonical sparse matrix
- support CSC matrix
- Implement layers, close #6
- allow empty AnnDataSet
- upgrade dependencies
- implement obs_ix and var_ix
- add ArrayIterator trait
- bug fixes
- allow empty AnnDataSet
- bug fixes
- add Python AnnDataSet
- reimplement pyanndata
- refactoring
- return reordering indices for AnnDataSet subsetting functions
- bump polars version
- minor
- allow indexing by names in AnnDataSet
- DataFrameElem refactoring
- minor
- allusing any sequence type to index
- indexing obs or var now returns polars Series objects
- use static build of C hdf5 library
- fit dict reading
- minor
- update pyo3 to 17.1
- upgrade dependencies
- add obs_ix and var_ix
- allow using names for indexing
- allow empty X in AnnDataSet
- allow add dataframes to anndata in Python
- add docs
- improve partial IO
- add docs
- minor
- allow reading full csr in AnnDataSet
- add no_check for read_anndataset
- allow making a new copy when subsetting
- refactoring
- improve type conversion
- add mappings
- refactoring
- add CI
- update
- use parking_lot
- add close function
- organize files
- work on AnnDataSet
- refactoring
- minor
- work on AnnDataSet
- refactoring
- refactoring
- add read_mtx
- add AnnDataSet
- add scalar
- refactoring
- change AnnData type
- refactoring
- add cache
- update
- update
- implment iterator
- bug fix
- udpate
- refacotring
- add test
- set obs and var
- add more functions
- refactoring
- refactoring
- data setter
- add dataframe support
- add obsm and varm
- add python bindings
