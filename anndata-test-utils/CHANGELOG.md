# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/scverse/anndata-rs/releases/tag/anndata-test-utils-v0.2.0) - 2026-06-22

### Added

- 1GB zarr shard defaults ([#22](https://github.com/scverse/anndata-rs/pull/22))
- add an option for storing absolute paths in AnnDataSet

### Fixed

- bring back zarr ([#21](https://github.com/scverse/anndata-rs/pull/21))
- fix to_memory with empty obs/var but non-empty obs_names/var_names
- fix non-string cat array reading
- improve sparse matrix reading
