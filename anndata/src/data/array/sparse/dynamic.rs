use core::panic;

use crate::backend::*;
use crate::data::ArrayConvert;
use crate::data::{
    data_traits::*,
    slice::{SelectInfoElem, Shape},
};
use crate::macros::arith;

use anyhow::{Context, Result, bail};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::ArrayD;
use num::NumCast;

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
    ($($scalar_ty:ty => $variant:ident),*) => {
        $( impl_dyncsr_traits!($scalar_ty, $variant); )*
    };
    ($scalar_ty:ty, $variant:ident) => {
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
    };
}

impl_dyncsr_traits!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    f32 => F32, f64 => F64,
    bool => Bool, String => String
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
        let dtype = iter
            .peek()
            .unwrap()
            .data_type()
            .scalar_type()
            .expect("Dyn* types always have a scalar type");
        macro_rules! fun {
            ($ty:ty) => {
                CsrMatrix::<$ty>::vstack(iter.map(|x| x.try_into().unwrap())).map(Into::into)
            };
        }
        crate::macros::dyn_match!(dtype, ScalarType, fun)
    }
}

impl ArrayArithmetic for DynCsrMatrix {
    fn sum(&self) -> f64 {
        crate::macros::dyn_map!(self, DynCsrMatrix, arith, panic, sum)
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        crate::macros::dyn_map!(self, DynCsrMatrix, arith, bail, sum_axis, axis)
    }

    fn min(&self) -> f64 {
        crate::macros::dyn_map!(self, DynCsrMatrix, arith, panic, min)
    }

    fn max(&self) -> f64 {
        crate::macros::dyn_map!(self, DynCsrMatrix, arith, panic, max)
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
        if let DataType::CsrMatrix(ty) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CsrMatrix::<$variant>::read_select(container, info)?.into()
                };
            }
            Ok(crate::macros::dyn_match!(ty, ScalarType, fun))
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
    ($($from_type:ty => $to_type:ident),*) => {
        $( impl_dyncsc_traits!($from_type, $to_type); )*
    };
    ($from_type:ty, $to_type:ident) => {
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
    };
}

impl_dyncsc_traits!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    f32 => F32, f64 => F64,
    bool => Bool, String => String
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
        if let DataType::CscMatrix(ty) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CscMatrix::<$variant>::read_select(container, info).map(Into::into)
                };
            }
            crate::macros::dyn_match!(ty, ScalarType, fun)
        } else {
            bail!("the container does not contain a csc matrix");
        }
    }
}

//-----------------------------------------------------------------------------
// ArrayConvert implementations
//-----------------------------------------------------------------------------

/// Inner macros for `dyn_map!`, converting `$data` with `$fun` (`convert_csr_with` or `convert_csc_with`).
macro_rules! int_cast {
    (F32, $data:expr, $fun:expr) => {
        unreachable!()
    };
    (F64, $data:expr, $fun:expr) => {
        unreachable!()
    };
    (String, $data:expr, $fun:expr) => {
        unreachable!()
    };
    ($variant:ident, $data:expr, $fun:expr) => {
        $fun($data, |x| Ok(x.try_into()?))
    };
}

macro_rules! float_cast {
    (Bool, $data:expr, $fun:expr) => {
        $fun($data, |x| Ok(x.into()))
    };
    (String, $data:expr, $fun:expr) => {
        unreachable!()
    };
    ($variant:ident, $data:expr, $fun:expr) => {
        $fun($data, |x| NumCast::from(x).context("value out of range"))
    };
}

macro_rules! impl_arrayconvert {
    ($($ty:ident => $fun:expr),*) => {
        $( impl_arrayconvert!($ty, $fun); )*
    };
    ($ty:ident, $fun:expr) => {
        paste::paste! {
            impl ArrayConvert<$ty<u32>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<u32>> {
                    match self {
                        [<Dyn $ty>]::U32(data) => Ok(data),
                        v @ ([<Dyn $ty>]::F32(_) | [<Dyn $ty>]::F64(_) | [<Dyn $ty>]::String(_)) => {
                            bail!("Cannot convert {} to {}<u32>", v.data_type(), stringify!($ty))
                        }
                        other => crate::macros::dyn_map!(other, [<Dyn $ty>], int_cast, $fun),
                    }
                }
            }

            impl ArrayConvert<$ty<f32>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<f32>> {
                    match self {
                        [<Dyn $ty>]::F32(data) => Ok(data),
                        v @ [<Dyn $ty>]::String(_) => {
                            bail!("Cannot convert {} to {}<f32>", v.data_type(), stringify!($ty))
                        }
                        other => crate::macros::dyn_map!(other, [<Dyn $ty>], float_cast, $fun),
                    }
                }
            }

            impl ArrayConvert<$ty<f64>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<f64>> {
                    match self {
                        [<Dyn $ty>]::F64(data) => Ok(data),
                        v @ [<Dyn $ty>]::String(_) => {
                            bail!("Cannot convert {} to {}<f64>", v.data_type(), stringify!($ty))
                        }
                        other => crate::macros::dyn_map!(other, [<Dyn $ty>], float_cast, $fun),
                    }
                }
            }
        }
    };
}

impl_arrayconvert!(CsrMatrix => convert_csr_with, CscMatrix => convert_csc_with);

fn convert_csr_with<T, U, F>(csr: CsrMatrix<T>, f: F) -> Result<CsrMatrix<U>>
where
    F: Fn(T) -> Result<U>,
{
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(f).collect::<Result<_, _>>()?,
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
        values.into_iter().map(f).collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr<T>(vals: Vec<T>) -> CsrMatrix<T> {
        let n = vals.len();
        CsrMatrix::try_from_csr_data(1, n, vec![0, n], (0..n).collect(), vals).unwrap()
    }

    #[test]
    fn convert_and_arith() {
        let m = DynCsrMatrix::I64(csr(vec![1i64, 2]));
        let r: CsrMatrix<f32> = m.clone().try_convert().unwrap();
        assert_eq!(r.values(), &[1.0, 2.0]);
        let r: CsrMatrix<u32> = m.try_convert().unwrap();
        assert_eq!(r.values(), &[1, 2]);
        let r: Result<CsrMatrix<u32>> = DynCsrMatrix::I8(csr(vec![-1i8])).try_convert();
        assert!(r.is_err());
        let r: CsrMatrix<f64> = DynCsrMatrix::Bool(csr(vec![true])).try_convert().unwrap();
        assert_eq!(r.values(), &[1.0]);
        let s = DynCsrMatrix::String(csr(vec!["a".to_owned()]));
        assert!(s.sum_axis(0).is_err());
        assert!(DynCsrMatrix::U8(csr(vec![3u8])).sum_axis(1).is_ok());
    }
}
