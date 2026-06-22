# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0](https://github.com/scverse/anndata-rs/compare/pyanndata-v0.6.0...pyanndata-v0.7.0) - 2026-06-22

## Added

- Implement default write configuration management and compression options in Python bindings
- Allow any index type when creating sparse matrix ([#18](https://github.com/scverse/anndata-rs/pull/18))
- implement split_obs_by for AnnDataSet
- add an option for storing absolute paths in AnnDataSet
- support string array conversion
- add partial to `.to_memory()`
- allow partial copying and writing of AnnData objects
- add split_by
