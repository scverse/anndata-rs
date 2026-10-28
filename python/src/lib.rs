use pyo3::pymodule;

#[pymodule]
mod anndata_rs {
    use pyo3::{Bound, PyResult};
    use pyo3::types::PyModule;
   
    #[pymodule_init]
    fn init(_: &Bound<'_, PyModule>) -> PyResult<()>  {
        pyo3_log::init();
        Ok(())
    }
    
    #[pymodule_export]
    use pyanndata::{AnnData,AnnDataSet,PyCompression};
    
    #[pymodule_export]
    use pyanndata::{read,read_dataset,read_mtx,concat,py_get_default_write_config,py_set_default_write_config};
 
    /*
    #[pymodule_export]
    use pyanndata::StackedAnnData;
    
    #[pymodule_export]
    use pyanndata::element::{PyElemCollection,PyAxisArrays,PyElem,PyMatrixElem,PyDataFrameElem,PyStackedMatrixElem,PyStackedAxisArrays,PyStackedDataFrame};

    #[pymodule_export]
    use pyanndata::read_csv;
    */
}
