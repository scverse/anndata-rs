use core::panic;

use crate::backend::*;
use crate::data::ArrayConvert;
use crate::data::{
    data_traits::*,
    slice::{SelectInfoElem, Shape},
};

use anyhow::{Result, bail};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::ArrayD;
use num::FromPrimitive;
use sprs::{CsMatI, SpIndex};

#[derive(Debug, Clone, PartialEq)]
pub enum DynIndSparseMatrix {
    I16(DynSparseMatrix<i16>),
    I32(DynSparseMatrix<i32>),
    I64(DynSparseMatrix<i64>),
    U16(DynSparseMatrix<u16>),
    U32(DynSparseMatrix<u32>),
    U64(DynSparseMatrix<u64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynSparseMatrix<T>
where
    T: SpIndex,
{
    I8(CsMatI<i8, T, u64>),
    I16(CsMatI<i16, T, u64>),
    I32(CsMatI<i32, T, u64>),
    I64(CsMatI<i64, T, u64>),
    U8(CsMatI<u8, T, u64>),
    U16(CsMatI<u16, T, u64>),
    U32(CsMatI<u32, T, u64>),
    U64(CsMatI<u64, T, u64>),
    F32(CsMatI<f32, T, u64>),
    F64(CsMatI<f64, T, u64>),
    Bool(CsMatI<bool, T, u64>),
    String(CsMatI<String, T, u64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCsrMatrix {
    I8(CsrMatrix<i8>),
    I16(CsrMatrix<i16>),
    I32(CsrMatrix<i32>),
    I64(CsrMatrix<i64>),
    U8(CsrMatrix<u8>),
    U16(CsrMatrix<u16>),
    U32(CsrMatrix<u32>),
    U64(CsrMatrix<u64>),
    F32(CsrMatrix<f32>),
    F64(CsrMatrix<f64>),
    Bool(CsrMatrix<bool>),
    String(CsrMatrix<String>),
}

macro_rules! impl_dyncsr_traits {
    ($($scalar_ty:ty, $variant:ident),*) => {
        $(
            impl From<CsrMatrix<$scalar_ty>> for DynCsrMatrix {
                fn from(data: CsrMatrix<$scalar_ty>) -> Self {
                    DynCsrMatrix::$variant(data)
                }
            }
            impl TryFrom<DynCsrMatrix> for CsrMatrix<$scalar_ty> {
                type Error = anyhow::Error;
                fn try_from(data: DynCsrMatrix) -> Result<Self> {
                    match data {
                        DynCsrMatrix::$variant(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {} to {} CsrMatrix",
                            data.data_type(),
                            stringify!($scalar_ty)
                        ),
                    }
                }
            }
        )*
    };
}

impl_dyncsr_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl Element for DynCsrMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, metadata)
    }
}

impl Writable for DynCsrMatrix {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, write, location, name)
    }
}

impl Readable for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CsrMatrix::<$variant>::read(container).map(Into::into)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, shape)
    }
}

impl Selectable for DynCsrMatrix {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! fun {
            ($variant:ident, $data:expr) => {
                $data.select(info).into()
            };
        }
        crate::macros::dyn_map!(self, DynCsrMatrix, fun)
    }
}

impl Stackable for DynCsrMatrix {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrMatrix::U8(_) => Ok(DynCsrMatrix::U8(CsrMatrix::<u8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::U16(_) => Ok(DynCsrMatrix::U16(CsrMatrix::<u16>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::U32(_) => Ok(DynCsrMatrix::U32(CsrMatrix::<u32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::U64(_) => Ok(DynCsrMatrix::U64(CsrMatrix::<u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I8(_) => Ok(DynCsrMatrix::I8(CsrMatrix::<i8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I16(_) => Ok(DynCsrMatrix::I16(CsrMatrix::<i16>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I32(_) => Ok(DynCsrMatrix::I32(CsrMatrix::<i32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I64(_) => Ok(DynCsrMatrix::I64(CsrMatrix::<i64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::F32(_) => Ok(DynCsrMatrix::F32(CsrMatrix::<f32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::F64(_) => Ok(DynCsrMatrix::F64(CsrMatrix::<f64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::Bool(_) => Ok(DynCsrMatrix::Bool(CsrMatrix::<bool>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::String(_) => Ok(DynCsrMatrix::String(CsrMatrix::<String>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
        }
    }
}

impl ArrayArithmetic for DynCsrMatrix {
    fn sum(&self) -> f64 {
        match self {
            DynCsrMatrix::I8(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::I16(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::I32(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::I64(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::U8(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::U16(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::U32(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::U64(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::F32(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::F64(arr) => ArrayArithmetic::sum(arr),
            DynCsrMatrix::Bool(_) => panic!("Cannot compute sum for Bool csr matrix"),
            DynCsrMatrix::String(_) => panic!("Cannot compute sum for String csr matrix"),
        }
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        match self {
            DynCsrMatrix::I8(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::I16(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::I32(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::I64(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::U8(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::U16(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::U32(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::U64(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::F32(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::F64(arr) => ArrayArithmetic::sum_axis(arr, axis),
            DynCsrMatrix::Bool(_) => panic!("Cannot compute sum for Bool csr matrix"),
            DynCsrMatrix::String(_) => bail!("Cannot compute sum for String csr matrix"),
        }
    }

    fn min(&self) -> f64 {
        match self {
            DynCsrMatrix::I8(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::I16(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::I32(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::I64(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::U8(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::U16(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::U32(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::U64(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::F32(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::F64(arr) => ArrayArithmetic::min(arr),
            DynCsrMatrix::Bool(_) => panic!("Cannot compute min for Bool csr matrix"),
            DynCsrMatrix::String(_) => panic!("Cannot compute min for String csr matrix"),
        }
    }

    fn max(&self) -> f64 {
        match self {
            DynCsrMatrix::I8(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::I16(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::I32(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::I64(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::U8(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::U16(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::U32(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::U64(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::F32(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::F64(arr) => ArrayArithmetic::max(arr),
            DynCsrMatrix::Bool(_) => panic!("Cannot compute max for Bool csr matrix"),
            DynCsrMatrix::String(_) => panic!("Cannot compute max for String csr matrix"),
        }
    }
}

impl WritableArray for DynCsrMatrix {}
impl ReadableArray for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CsrMatrix(ty, tp) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CsrMatrix::<$variant>::read_select(container, info)?.into()
                };
            }
            Ok(crate::macros::dyn_match!(tp, ScalarType, fun))
        } else {
            bail!("the container does not contain a csr matrix");
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCscMatrix {
    I8(CscMatrix<i8>),
    I16(CscMatrix<i16>),
    I32(CscMatrix<i32>),
    I64(CscMatrix<i64>),
    U8(CscMatrix<u8>),
    U16(CscMatrix<u16>),
    U32(CscMatrix<u32>),
    U64(CscMatrix<u64>),
    F32(CscMatrix<f32>),
    F64(CscMatrix<f64>),
    Bool(CscMatrix<bool>),
    String(CscMatrix<String>),
}

macro_rules! impl_dyncsc_traits {
    ($($from_type:ty, $to_type:ident),*) => {
        $(
            impl From<CscMatrix<$from_type>> for DynCscMatrix {
                fn from(data: CscMatrix<$from_type>) -> Self {
                    DynCscMatrix::$to_type(data)
                }
            }
            impl TryFrom<DynCscMatrix> for CscMatrix<$from_type> {
                type Error = anyhow::Error;
                fn try_from(data: DynCscMatrix) -> Result<Self> {
                    match data {
                        DynCscMatrix::$to_type(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {:?} to {} CscMatrix",
                            data.data_type(),
                            stringify!($from_type)
                        ),
                    }
                }
            }
        )*
    };
}

impl_dyncsc_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl Element for DynCscMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, metadata)
    }
}

impl Writable for DynCscMatrix {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, write, location, name)
    }
}

impl Readable for DynCscMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CscMatrix::<$variant>::read(container).map(Into::into)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("cannot read csc matrix from non-group container"),
        }
    }
}

impl HasShape for DynCscMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, shape)
    }
}

impl Selectable for DynCscMatrix {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! select {
            ($variant:ident, $data:expr) => {
                $data.select(info).into()
            };
        }
        crate::macros::dyn_map!(self, DynCscMatrix, select)
    }
}

impl WritableArray for DynCscMatrix {}
impl ReadableArray for DynCscMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CscMatrix(ty, tp) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CscMatrix::<$variant>::read_select(container, info).map(Into::into)
                };
            }
            crate::macros::dyn_match!(tp, ScalarType, fun)
        } else {
            bail!("the container does not contain a csc matrix");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// ArrayConvert implementations
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_arrayconvert {
    ($($ty:ident, $fun:expr),*) => {
        $(paste::paste! {

            impl ArrayConvert<$ty<u32>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<u32>> {
                    match self {
                        [<Dyn $ty>]::U32(data) => Ok(data),
                        [<Dyn $ty>]::I8(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::I16(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::I32(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::I64(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::U8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U64(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::Bool(data) => $fun(data, |x| Ok(x.into())),
                        v => bail!("Cannot convert {} to {}<u32>", v.data_type(), stringify!($ty)),
                    }
                }
            }

            impl ArrayConvert<$ty<f32>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<f32>> {
                    match self {
                        [<Dyn $ty>]::F32(data) => Ok(data),
                        [<Dyn $ty>]::I8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I32(data) => $fun(data, |x| Ok(f32::from_i32(x).unwrap())),
                        [<Dyn $ty>]::I64(data) => $fun(data, |x| Ok(f32::from_i64(x).unwrap())),
                        [<Dyn $ty>]::U8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U32(data) => $fun(data, |x| Ok(f32::from_u32(x).unwrap())),
                        [<Dyn $ty>]::U64(data) => $fun(data, |x| Ok(f32::from_u64(x).unwrap())),
                        [<Dyn $ty>]::F64(data) => $fun(data, |x| Ok(f32::from_f64(x).unwrap())),
                        [<Dyn $ty>]::Bool(data) => $fun(data, |x| Ok(x.into())),
                        v => bail!("Cannot convert {} to {}<f32>", v.data_type(), stringify!($ty)),
                    }
                }
            }

            impl ArrayConvert<$ty<f64>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<f64>> {
                    match self {
                        [<Dyn $ty>]::F64(data) => Ok(data),
                        [<Dyn $ty>]::I8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I32(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I64(data) => $fun(data, |x| Ok(f64::from_i64(x).unwrap())),
                        [<Dyn $ty>]::U8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U32(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U64(data) => $fun(data, |x| Ok(f64::from_u64(x).unwrap())),
                        [<Dyn $ty>]::F32(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::Bool(data) => $fun(data, |x| Ok(x.into())),
                        v => bail!("Cannot convert {} to {}<f64>", v.data_type(), stringify!($ty)),
                    }
                }
            }

        })*
    };
}

impl_arrayconvert!(CsrMatrix, convert_csr_with, CscMatrix, convert_csc_with);

fn convert_csr_with<T, U, F>(csr: CsrMatrix<T>, f: F) -> Result<CsrMatrix<U>>
where
    F: Fn(T) -> Result<U>,
{
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(|x| f(x)).collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

fn convert_csc_with<T, U, F>(csc: CscMatrix<T>, f: F) -> Result<CscMatrix<U>>
where
    F: Fn(T) -> Result<U>,
{
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(|x| f(x)).collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

///// New DynIndSparseMatrix
///
///

impl<T: SpIndex + BackendData> Element for DynSparseMatrix<T> {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, metadata)
    }
}

impl<T: BackendData + SpIndex> Writable for DynSparseMatrix<T> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, write, location, name)
    }
}

impl<T: BackendData + SpIndex> Readable for DynSparseMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        CsMatI::<$type, T, u64>::read(container).map(DynSparseMatrix::$variant)
                    };
                }
                crate::macros::dyn_match_new!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("Can't read sparse matrix from non-group container!"),
        }
    }
}

impl<T: BackendData + SpIndex> HasShape for DynSparseMatrix<T> {
    fn shape(&self) -> Shape {
        match self {
            DynSparseMatrix::I8(data) => HasShape::shape(data),
            DynSparseMatrix::I16(data) => HasShape::shape(data),
            DynSparseMatrix::I32(data) => HasShape::shape(data),
            DynSparseMatrix::I64(data) => HasShape::shape(data),
            DynSparseMatrix::U8(data) => HasShape::shape(data),
            DynSparseMatrix::U16(data) => HasShape::shape(data),
            DynSparseMatrix::U32(data) => HasShape::shape(data),
            DynSparseMatrix::U64(data) => HasShape::shape(data),
            DynSparseMatrix::F32(data) => HasShape::shape(data),
            DynSparseMatrix::F64(data) => HasShape::shape(data),
            DynSparseMatrix::Bool(data) => HasShape::shape(data),
            DynSparseMatrix::String(data) => HasShape::shape(data),
        }
    }
}

impl<T: BackendData + SpIndex> Selectable for DynSparseMatrix<T> {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        crate::macros::dyn_sparse_map!(self, DynSparseMatrix, select, info)
    }
}

impl<T: BackendData + SpIndex> SparseMatrixLayout for DynSparseMatrix<T> {
    fn get_sparse_layout(&self) -> SparseMatrixLayoutE {
        crate::macros::dyn_map_fun!(self, DynSparseMatrix, get_sparse_layout)
    }
}

// no stackable yet implemented. TODO later

impl<T: BackendData + SpIndex> WritableArray for DynSparseMatrix<T> {}

impl<T: BackendData + SpIndex> ReadableArray for DynSparseMatrix<T> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($type: ident, $variant:ident) => {
                        CsMatI::<$type, T, u64>::read_select(container, info)
                            .map(DynSparseMatrix::$variant)
                    };
                }
                crate::macros::dyn_match_new!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("Unable to read sparse matrix from non-group container"),
        }
    }
}

// trait impls for dynsparsematrix

impl Element for DynIndSparseMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, metadata)
    }
}

impl Writable for DynIndSparseMatrix {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, write, location, name)
    }
}

impl Readable for DynIndSparseMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                let indices_dtype = group.open_dataset("indices")?.dtype()?;
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        DynSparseMatrix::<$type>::read(container).map(DynIndSparseMatrix::$variant)
                    };
                }

                crate::macros::dyn_index_match!(indices_dtype, ScalarType, fun)
            }
            _ => bail!("Can't read sparse amtrix from non-group container"),
        }
    }
}

impl HasShape for DynIndSparseMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, shape)
    }
}

impl Selectable for DynIndSparseMatrix {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        crate::macros::dyn_index_sparse_map!(self, DynIndSparseMatrix, select, info)
    }
}

impl WritableArray for DynIndSparseMatrix {}

impl ReadableArray for DynIndSparseMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        match container {
            DataContainer::Group(group) => {
                let indices_type = group.open_dataset("indices")?.dtype()?;
                macro_rules! fun {
                    ($type:ident, $variant:ident) => {
                        DynSparseMatrix::<$type>::read_select(container, info)
                            .map(DynIndSparseMatrix::$variant)
                    };
                }
                crate::macros::dyn_index_match!(indices_type, ScalarType, fun)
            }
            _ => bail!("Cant read sparse matrix from non group container"),
        }
    }
}

impl SparseMatrixLayout for DynIndSparseMatrix {
    fn get_sparse_layout(&self) -> SparseMatrixLayoutE {
        crate::macros::dyn_index_map_fun!(self, DynIndSparseMatrix, get_sparse_layout)
    }
}

impl From<DynCsrMatrix> for DynIndSparseMatrix {
    fn from(value: DynCsrMatrix) -> Self {
        match value {
            DynCsrMatrix::I8(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I8(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::I16(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I16(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::I32(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I32(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::I64(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I64(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::U8(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U8(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::U16(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U16(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::U32(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U32(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::U64(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U64(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::F32(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::F32(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::F64(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::F64(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::Bool(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::Bool(csr_matrix_conv_csmati(csr)))
            }
            DynCsrMatrix::String(csr) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::String(csr_matrix_conv_csmati(csr)))
            }
        }
    }
}

impl From<DynCscMatrix> for DynIndSparseMatrix {
    fn from(value: DynCscMatrix) -> Self {
        match value {
            DynCscMatrix::I8(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I8(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::I16(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I16(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::I32(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I32(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::I64(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::I64(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::U8(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U8(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::U16(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U16(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::U32(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U32(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::U64(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::U64(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::F32(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::F32(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::F64(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::F64(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::Bool(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::Bool(csc_matrix_conv_csmati(csc)))
            }
            DynCscMatrix::String(csc) => {
                DynIndSparseMatrix::U64(DynSparseMatrix::String(csc_matrix_conv_csmati(csc)))
            }
        }
    }
}

fn csr_matrix_conv_csmati<T: BackendData>(csr: CsrMatrix<T>) -> CsMatI<T, u64, u64> {
    let shape = (csr.nrows(), csr.ncols());
    let (rows, col, data) = csr.disassemble();
    let rows: Vec<u64> = vec_usize_to_u64(rows);
    let cols: Vec<u64> = vec_usize_to_u64(col);
    CsMatI::new(shape, rows, cols, data)
}

fn csc_matrix_conv_csmati<T: BackendData>(csc: CscMatrix<T>) -> CsMatI<T, u64, u64> {
    let shape = (csc.nrows(), csc.ncols());
    let (cols, rows, data) = csc.disassemble();
    let cols: Vec<u64> = vec_usize_to_u64(cols);
    let rows: Vec<u64> = vec_usize_to_u64(rows);
    CsMatI::new_csc(shape, cols, rows, data)
}

#[inline]
pub fn vec_usize_to_u64(v: Vec<usize>) -> Vec<u64> {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        use std::mem::ManuallyDrop;

        let mut v = ManuallyDrop::new(v);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut u64, v.len(), v.capacity())
    }

    #[cfg(not(target_pointer_width = "64"))]
    {
        v.into_iter().map(|x| x as u64).collect()
    }
}

macro_rules! impl_from_dynsparse_to_dynind {
    ($($type:ty, $variant:ident),*) => {
        $(
            impl From<DynSparseMatrix<$type>> for DynIndSparseMatrix {
                fn from(value: DynSparseMatrix<$type>) -> Self {
                    DynIndSparseMatrix::$variant(value)
                }
            }
        )*
    };
}

impl_from_dynsparse_to_dynind!(i16, I16, i32, I32, i64, I64, u16, U16, u32, U32, u64, U64);

macro_rules! impl_from_csmati_to_dynsparse {
    ($($value_type:ty, $value_variant:ident),*) => {
        $(
            impl<T: SpIndex + BackendData> From<CsMatI<$value_type, T, u64>> for DynSparseMatrix<T> {
                fn from(value: CsMatI<$value_type, T, u64>) -> Self {
                    DynSparseMatrix::$value_variant(value)
                }
            }
        )*
    };
}

impl_from_csmati_to_dynsparse!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

macro_rules! impl_tryfrom_dynsparse_to_csmati {
    ($($value_type:ty, $value_variant:ident),*) => {
        $(
            impl<T: SpIndex + BackendData> TryFrom<DynSparseMatrix<T>> for CsMatI<$value_type, T, u64> {
                type Error = anyhow::Error;

                fn try_from(value: DynSparseMatrix<T>) -> Result<Self> {
                    match value {
                        DynSparseMatrix::$value_variant(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {} to CsMatI<{}, {}, u64>",
                            value.data_type(),
                            stringify!($value_type),
                            std::any::type_name::<T>()
                        ),
                    }
                }
            }
        )*
    };
}

impl_tryfrom_dynsparse_to_csmati!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

macro_rules! impl_tryfrom_dynind_to_dynsparse {
    ($($index_type:ty, $index_variant:ident),*) => {
        $(
            impl TryFrom<DynIndSparseMatrix> for DynSparseMatrix<$index_type> {
                type Error = anyhow::Error;

                fn try_from(value: DynIndSparseMatrix) -> Result<Self> {
                    match value {
                        DynIndSparseMatrix::$index_variant(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert DynIndSparseMatrix with index type {:?} to DynSparseMatrix<{}>",
                            value.data_type(),
                            stringify!($index_type)
                        ),
                    }
                }
            }
        )*
    };
}

impl_tryfrom_dynind_to_dynsparse!(i16, I16, i32, I32, i64, I64, u16, U16, u32, U32, u64, U64);
