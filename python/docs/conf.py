"""Sphinx configuration."""

from __future__ import annotations

from importlib.metadata import metadata

from scverse_doc.registry import intersphinx

project = "anndata-rs"
meta = metadata(project)
author = "Kai Zhang"
copyright = f"2022, {author}"
version = release = meta["version"]

extensions = ["scverse_doc"]
source_repository = "https://github.com/scverse/anndata-rs"
source_directory = "python/docs"

autodoc_member_order = "groupwise"

intersphinx_mapping = intersphinx() | dict(
    h5py=("https://docs.h5py.org/en/stable/", None),
    polars=("https://docs.pola.rs/api/python/stable/", None),
)
xref_overrides = {"Path": "pathlib.Path"}
