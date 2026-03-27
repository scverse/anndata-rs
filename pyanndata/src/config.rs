use anndata::backend::{Compression, get_default_write_config, set_default_write_config};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compression algorithm for data storage
#[pyclass(name = "Compression")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyCompression {
    inner: Compression,
}

#[pymethods]
impl PyCompression {
    /// Create a Gzip compression with the specified level (1-9)
    #[staticmethod]
    fn gzip(level: u8) -> PyResult<Self> {
        if level < 1 || level > 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Gzip compression level must be between 1 and 9",
            ));
        }
        Ok(PyCompression {
            inner: Compression::Gzip(level),
        })
    }

    /// Create a Zstandard compression with the specified level (1-22)
    #[staticmethod]
    fn zstd(level: u8) -> PyResult<Self> {
        if level < 1 || level > 22 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Zstd compression level must be between 1 and 22",
            ));
        }
        Ok(PyCompression {
            inner: Compression::Zst(level),
        })
    }

    fn __eq__(&self, other: &PyCompression) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Compression::Gzip(level) => format!("Compression.gzip({})", level),
            Compression::Zst(level) => format!("Compression.zstd({})", level),
        }
    }
}

/// Set the default write configuration for all subsequent write operations.
/// 
/// This configuration is stored in thread-local storage and applies to all
/// dataset write operations that don't explicitly specify a configuration.
/// 
/// Parameters
/// ----------
/// config : dict, optional
///     Dictionary with optional keys:
///     - "compression": Compression or None
///     - "block_size": list of int or None
/// 
/// Examples
/// --------
/// >>> import anndata_rs
/// >>> # Set Gzip compression with custom block size
/// >>> anndata_rs.set_write_options({
/// ...     "compression": anndata_rs.Compression.gzip(9),
/// ...     "block_size": [1024, 1024],
/// ... })
/// >>> # Set only compression
/// >>> anndata_rs.set_write_options({
/// ...     "compression": anndata_rs.Compression.zstd(10)
/// ... })
#[pyfunction]
#[pyo3(name = "set_write_options", signature = (config))]
pub fn py_set_default_write_config(config: Bound<'_, PyDict>) -> PyResult<()> {
    let mut new_config = get_default_write_config();

    if let Some(value) = config.get_item("compression")? {
        new_config.compression = value.extract::<Option<PyCompression>>()?.map(|c| c.inner);
    }

    if let Some(value) = config.get_item("block_size")? {
        new_config.block_size = value.extract::<Option<Vec<usize>>>()?.map(|s| s.into());
    }

    set_default_write_config(new_config);
    Ok(())
}

/// Get the current default write configuration.
/// 
/// Returns the thread-local default configuration used for dataset writes.
/// 
/// Returns
/// -------
/// dict
///     Dictionary with keys "compression" and "block_size".
/// 
/// Examples
/// --------
/// >>> import anndata_rs
/// >>> config = anndata_rs.get_write_options()
/// >>> print(config["compression"])
/// Compression.zstd(5)
/// >>> print(config["block_size"])
/// None
#[pyfunction]
#[pyo3(name = "get_write_options")]
pub fn py_get_default_write_config<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let config = get_default_write_config();

    let compression: Option<PyCompression> = config.compression.map(|c| PyCompression { inner: c });
    let block_size: Option<Vec<usize>> = config.block_size.map(|s| s.as_ref().to_vec());

    let dict = PyDict::new(py);
    dict.set_item("compression", compression)?;
    dict.set_item("block_size", block_size)?;

    Ok(dict)
}
