use crate::data::instance::*;

use pyo3::prelude::*;
use anndata::data::{Shape, SelectInfo, SelectInfoElem};

pub fn to_select_info(ob: &Bound<'_, PyAny>, shape: &Shape) -> PyResult<SelectInfo> {
    let ndim = shape.ndim();
    if is_none_slice(ob)? {
        Ok(std::iter::repeat(SelectInfoElem::full()).take(ndim).collect())
    } else if ob.is_instance_of::<pyo3::types::PyTuple>() {
        ob.try_iter()?.zip(shape.as_ref())
            .map(|(x, len)| to_select_elem(&x?, *len))
            .collect()
    } else {
        todo!()
    }
}

/// Converts a Python object into a `SelectInfoElem` for array indexing.
///
/// This function handles multiple Python indexing types and converts them to a unified
/// `SelectInfoElem` representation:
///
/// - **PySlice**: Converts Python slice objects (e.g., `1:5:2`) into a `Slice` with
///   start, stop, and step values.
/// - **None**: Represents a full selection (equivalent to `:`).
/// - **PyInt**: Extracts a single integer index.
/// - **Boolean NumPy array**: Validates the array has dtype `bool` and matches the
///   expected length, then converts to indices via `boolean_mask_to_indices`.
/// - **Iterable of booleans**: Attempts to extract as a Python iterable of booleans.
///   If successful and length matches, converts to indices. If length is 0, returns
///   empty selection. If length mismatches, panics.
/// - **Iterable of integers**: Falls back to extracting as a vector of `usize` indices.
///
/// # Arguments
///
/// * `ob` - A Python object to be interpreted as an index or mask.
/// * `length` - The expected length of the dimension being indexed.
///
/// # Panics
///
/// Panics if a boolean mask's length does not match the expected `length`.
///
/// # Returns
///
/// A `SelectInfoElem` representing the selection operation.
pub fn to_select_elem(ob: &Bound<'_, PyAny>, length: usize) -> PyResult<SelectInfoElem> {
    let select = if let Ok(slice) = ob.cast::<pyo3::types::PySlice>() {
        let s = slice.indices(length.try_into()?)?;
        ndarray::Slice { 
            start: s.start,
            end: Some(s.stop),
            step: s.step,
        }.into()
    } else if is_none_slice(ob)? {
        SelectInfoElem::full()
    } else if ob.is_instance_of::<pyo3::types::PyInt>() {
        ob.extract::<usize>()?.into()
    } else if isinstance_of_arr(ob)? && ob.getattr("dtype")?.getattr("name")?.extract::<&str>()? == "bool" {
        let arr = ob
            .extract::<numpy::PyReadonlyArray1<bool>>()?;
        if arr.len()? == length {
            boolean_mask_to_indices(arr.as_array().into_iter().map(|x| *x)).into()
        } else {
            panic!("boolean mask dimension mismatched")
        }
    } else {
        let boolean_mask: PyResult<Vec<bool>> =
            ob.try_iter()?.map(|x| x.unwrap().extract()).collect();
        match boolean_mask {
            Ok(mask) => {
                if mask.len() == length {
                    boolean_mask_to_indices(mask.into_iter()).into()
                } else if mask.len() == 0 {
                    Vec::new().into()
                } else {
                    panic!("boolean mask dimension mismatched")
                }
            }
            _ => ob.try_iter()?.map(|x| x.unwrap().extract()).collect::<PyResult<Vec<usize>>>()?.into(),
        }
    };
    Ok(select)
}

fn boolean_mask_to_indices<I>(iter: I) -> Vec<usize>
where
    I: Iterator<Item = bool>,
{
    iter.enumerate()
        .filter_map(|(i, x)| if x { Some(i) } else { None })
        .collect()
}