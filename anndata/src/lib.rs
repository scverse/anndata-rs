mod anndata;
pub mod backend;
pub mod concat;
pub mod container;
pub mod data;
mod macros;
pub mod reader;
pub mod traits;

pub use crate::anndata::{AnnData, AnnDataSet, StackedAnnData};
pub use backend::Backend;
pub use container::{
    ArrayElem, AxisArrays, DataFrameElem, Elem, ElemCollection, StackedArrayElem,
    StackedAxisArrays, StackedDataFrame,
};
pub use data::{
    ArrayData, Data, HasShape, Readable, ReadableArray, Selectable, Writable, WritableArray,
};
pub use traits::{AnnDataOp, ArrayElemOp, AxisArraysOp, ElemCollectionOp};
