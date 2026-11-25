use crate::{
    backend::*,
    data::{
        data_traits::*,
        slice::{SelectInfoElem, Shape},
    },
    macros::arith,
};

use anyhow::{Result, bail, ensure};
use ndarray::{Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn, arr0};
use paste::paste;
use polars::series::Series;

#[derive(Debug, Clone, PartialEq)]
pub enum DynScalar {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
}

/// macro to implement `From` trait for `DynScalar`
macro_rules! impl_from_dynscalar {
    ($($from:ident => $to:ident),*) => {
        $( impl_from_dynscalar!($from, $to); )*
    };
    ($from:ident, $to:ident) => {
        impl From<$from> for DynScalar {
            fn from(val: $from) -> Self {
                DynScalar::$to(val)
            }
        }

        impl Readable for $from {
            fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
                let dataset = container.as_dataset()?;
                match dataset.dtype()? {
                    ScalarType::$to => Ok(dataset.read_scalar()?),
                    _ => bail!("Cannot read $from"),
                }
            }
        }

        impl Element for $from {
            fn data_type(&self) -> DataType {
                DataType::Scalar(ScalarType::$to)
            }

            fn metadata(&self) -> MetaData {
                let encoding_type = if $from::DTYPE == ScalarType::String {
                    "string"
                } else {
                    "numeric-scalar"
                };
                MetaData::new(encoding_type, "0.2.0", None)
            }
        }

        impl Writable for $from {
            fn write<B: Backend, G: GroupOp<B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
                let dataset = location.new_scalar_dataset(name, self)?;
                let mut container = DataContainer::Dataset(dataset);
                self.metadata().save(&mut container)?;
                Ok(container)
            }
        }
    };
}

impl_from_dynscalar!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    f32 => F32, f64 => F64,
    bool => Bool, String => String
);

impl Element for DynScalar {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynScalar, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynScalar, metadata)
    }
}

impl Writable for DynScalar {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynScalar, write, location, name)
    }
}

impl Readable for DynScalar {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let dataset = container.as_dataset()?;

        macro_rules! fun {
            ($variant:ident) => {
                dataset.read_scalar::<$variant>()?.into()
            };
        }

        Ok(crate::macros::dyn_match!(dataset.dtype()?, ScalarType, fun))
    }
}

/// A dynamic-typed array.
#[derive(Debug, Clone, PartialEq)]
pub enum DynArray {
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    Bool(ArrayD<bool>),
    String(ArrayD<String>),
}

macro_rules! impl_dynarray_into_array{
    ($($variant:ident => $scalar_ty:ident),*) => {
        $( impl_dynarray_into_array!($variant, $scalar_ty); )*
    };
    ($variant:ident, $scalar_ty:ident) => {
        paste! {
            pub fn [<as_ $scalar_ty:lower>](&self) -> Result<&ArrayD<$scalar_ty>> {
                match self {
                    DynArray::$variant(x) => Ok(x),
                    v => bail!("Cannot convert {} to {}", v.data_type(), stringify!($scalar_ty)),
                }
            }
        }
    };
}

impl DynArray {
    pub fn ndim(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynArray, ndim)
    }

    pub fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynArray, shape)
            .to_vec()
            .into()
    }

    pub fn len(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynArray, len)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        crate::macros::dyn_map_fun!(self, DynArray, is_empty)
    }

    impl_dynarray_into_array!(
        I8 => i8, I16 => i16, I32 => i32, I64 => i64,
        U8 => u8, U16 => u16, U32 => u32, U64 => u64,
        F32 => f32, F64 => f64,
        Bool => bool, String => String
    );
}

macro_rules! impl_dynarray_traits{
    ($($scalar_ty:ty => $ident:ident),*) => {
        $( impl_dynarray_traits!($scalar_ty, $ident); )*
    };
    ($scalar_ty:ty, $ident:ident) => {
        impl<D: Dimension> From<Array<$scalar_ty, D>> for DynArray {
            fn from(data: Array<$scalar_ty, D>) -> Self {
                DynArray::$ident(data.into_dyn())
            }
        }

        impl<D: Dimension> TryFrom<DynArray> for Array<$scalar_ty, D> {
            type Error = anyhow::Error;
            fn try_from(arr: DynArray) -> Result<Self, Self::Error> {
                match arr {
                    DynArray::$ident(x) => Ok(x.into_dimensionality::<D>()?),
                    v => bail!("Cannot convert {} to {}", v.data_type(), stringify!($scalar_ty)),
                }
            }
        }
    };
}

impl_dynarray_traits!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    f32 => F32, f64 => F64,
    bool => Bool, String => String
);

impl From<DynArray> for Series {
    fn from(val: DynArray) -> Self {
        macro_rules! fun {
            (String, $x:expr) => {
                $x.iter().map(|x| x.as_str()).collect()
            };
            ($variant:ident, $x:expr) => {
                $x.iter().collect()
            };
        }
        crate::macros::dyn_map!(val, DynArray, fun)
    }
}

impl Element for DynArray {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynArray, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynArray, metadata)
    }
}

impl Writable for DynArray {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, Self, write, location, name)
    }
}

impl Readable for DynArray {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        container.as_dataset()?.read_dyn_array()
    }
}

impl HasShape for DynArray {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynArray, shape)
            .to_vec()
            .into()
    }
}

impl Indexable for DynArray {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                $exp.get(index).map(|x| x.clone().into())
            };
        }
        crate::macros::dyn_map!(self, DynArray, fun)
    }
}

impl Selectable for DynArray {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                Selectable::select($exp, info).into()
            };
        }
        crate::macros::dyn_map!(self, DynArray, fun)
    }
}

impl Stackable for DynArray {
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
                ArrayD::<$ty>::vstack(iter.map(|x| x.try_into().unwrap())).map(Into::into)
            };
        }
        crate::macros::dyn_match!(dtype, ScalarType, fun)
    }
}

impl WritableArray for DynArray {}
impl ReadableArray for DynArray {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        container.as_dataset()?.read_dyn_array_slice(info)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCowArray<'a> {
    I8(CowArray<'a, i8, IxDyn>),
    I16(CowArray<'a, i16, IxDyn>),
    I32(CowArray<'a, i32, IxDyn>),
    I64(CowArray<'a, i64, IxDyn>),
    U8(CowArray<'a, u8, IxDyn>),
    U16(CowArray<'a, u16, IxDyn>),
    U32(CowArray<'a, u32, IxDyn>),
    U64(CowArray<'a, u64, IxDyn>),
    F32(CowArray<'a, f32, IxDyn>),
    F64(CowArray<'a, f64, IxDyn>),
    Bool(CowArray<'a, bool, IxDyn>),
    String(CowArray<'a, String, IxDyn>),
}

impl From<DynScalar> for DynCowArray<'_> {
    fn from(scalar: DynScalar) -> Self {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                DynCowArray::$variant(arr0($exp).into_dyn().into())
            };
        }
        crate::macros::dyn_map!(scalar, DynScalar, fun)
    }
}

impl DynCowArray<'_> {
    pub fn ndim(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynCowArray, ndim)
    }

    pub fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCowArray, shape)
            .to_vec()
            .into()
    }

    pub fn len(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynCowArray, len)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        crate::macros::dyn_map_fun!(self, DynCowArray, is_empty)
    }
}

macro_rules! impl_dyn_cowarray_convert {
    ($($from_type:ty => $to_type:ident),+) => {
        $( impl_dyn_cowarray_convert!($from_type, $to_type); )*
    };
    ($from_type:ty, $to_type:ident) => {
        impl<D: Dimension> From<Array<$from_type, D>> for DynCowArray<'_> {
            fn from(data: Array<$from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn().into())
            }
        }

        impl<'a, D: Dimension> From<ArrayView<'a, $from_type, D>> for DynCowArray<'a> {
            fn from(data: ArrayView<'a, $from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn().into())
            }
        }

        impl<'a, D: Dimension> From<CowArray<'a, $from_type, D>> for DynCowArray<'a> {
            fn from(data: CowArray<'a, $from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn())
            }
        }

        impl<D: Dimension> TryFrom<DynCowArray<'_>> for Array<$from_type, D> {
            type Error = anyhow::Error;
            fn try_from(v: DynCowArray) -> Result<Self, Self::Error> {
                match v {
                    DynCowArray::$to_type(data) => {
                        let arr: ArrayD<$from_type> = data.into_owned();
                        if let Some(n) = D::NDIM {
                            ensure!(
                                arr.ndim() == n,
                                format!("Dimension mismatch: {} (in) != {} (out)", arr.ndim(), n)
                            );
                        }
                        Ok(arr.into_dimensionality::<D>()?)
                    }
                    _ => bail!("Cannot convert to {} ArrayD", stringify!($from_type)),
                }
            }
        }
    };
}

impl_dyn_cowarray_convert!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    f32 => F32, f64 => F64,
    bool => Bool, String => String
);

/// `ArrayConvert` trait for converting dynamic arrays to concrete arrays.
/// The `try_convert` method performs the conversion and returns the result.
pub trait ArrayConvert<T> {
    fn try_convert(self) -> Result<T>;
}

/// Integer source variants converted with `try_into` (lossless ones go through `From`).
macro_rules! int_cast {
    (F32, $data:expr) => {
        unreachable!()
    };
    (F64, $data:expr) => {
        unreachable!()
    };
    (String, $data:expr) => {
        unreachable!()
    };
    ($variant:ident, $data:expr) => {{
        let data = $data;
        let shape = data.raw_dim();
        // logical iteration order == standard layout, so `from_shape_vec` restores the array
        let vec = data
            .into_iter()
            .map(|x| x.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Array::from_shape_vec(shape, vec)?.into_dimensionality()?)
    }};
}

macro_rules! impl_int_array_convert {
    ($($ty:ty $(=> $variant:ident)?),*) => { $(
        impl<D: Dimension> ArrayConvert<Array<$ty, D>> for DynArray {
            fn try_convert(self) -> Result<Array<$ty, D>> {
                match self {
                    $(DynArray::$variant(data) => Ok(data.into_dimensionality()?),)?
                    DynArray::F32(_) | DynArray::F64(_) | DynArray::String(_) => {
                        bail!("Cannot convert to {} Array", stringify!($ty))
                    }
                    other => crate::macros::dyn_map!(other, DynArray, int_cast),
                }
            }
        }
    )* };
}

impl_int_array_convert!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    usize
);

impl<D: Dimension> ArrayConvert<Array<f32, D>> for DynArray {
    fn try_convert(self) -> Result<Array<f32, D>> {
        match self {
            DynArray::F32(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert {self:?} to f32 Array"),
        }
    }
}

impl<D: Dimension> ArrayConvert<Array<f64, D>> for DynArray {
    fn try_convert(self) -> Result<Array<f64, D>> {
        macro_rules! fun {
            (F64, $data:expr) => {
                Ok($data.into_dimensionality()?)
            };
            (String, $data:expr) => {
                bail!("Cannot convert {} to f64 Array", $data.data_type())
            };
            // i64 and u64 have no lossless conversion to f64
            (I64, $data:expr) => {
                Ok($data.mapv(|x| x as f64).into_dimensionality()?)
            };
            (U64, $data:expr) => {
                Ok($data.mapv(|x| x as f64).into_dimensionality()?)
            };
            ($variant:ident, $data:expr) => {
                Ok($data.mapv(|x| x.into()).into_dimensionality()?)
            };
        }
        crate::macros::dyn_map!(self, DynArray, fun)
    }
}

impl<D: Dimension> ArrayConvert<Array<bool, D>> for DynArray {
    fn try_convert(self) -> Result<Array<bool, D>> {
        match self {
            DynArray::Bool(data) => Ok(data.into_dimensionality()?),
            _ => bail!("Cannot convert {self:?} to bool Array"),
        }
    }
}

impl<D: Dimension> ArrayConvert<Array<String, D>> for DynArray {
    fn try_convert(self) -> Result<Array<String, D>> {
        match self {
            DynArray::String(data) => Ok(data.into_dimensionality()?),
            other => {
                macro_rules! fun {
                    ($variant:ident, $data:expr) => {
                        Ok($data.mapv(|x| x.to_string()).into_dimensionality()?)
                    };
                }
                crate::macros::dyn_map!(other, DynArray, fun)
            }
        }
    }
}

impl ArrayArithmetic for DynArray {
    fn sum(&self) -> f64 {
        crate::macros::dyn_map!(self, DynArray, arith, panic, sum)
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayD<f64>> {
        crate::macros::dyn_map!(self, DynArray, arith, bail, sum_axis, axis)
    }

    fn min(&self) -> f64 {
        crate::macros::dyn_map!(self, DynArray, arith, panic, min)
    }

    fn max(&self) -> f64 {
        crate::macros::dyn_map!(self, DynArray, arith, panic, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn int_convert_out_of_range_errors() {
        let arr = DynArray::I64(array![[1i64, -1]].into_dyn());
        let res: Result<Array2<u8>> = arr.try_convert();
        let err = res.unwrap_err();
        assert!(
            err.downcast_ref::<std::num::TryFromIntError>().is_some(),
            "expected out-of-range error, got: {err:?}"
        );
    }

    #[test]
    fn int_convert_keeps_values_and_shape() {
        let src = array![[1i8, 2, 3], [4, 5, 6]];
        // transposed: non-standard layout must still come out in logical order
        let arr = DynArray::I8(src.t().to_owned().into_dyn());
        let res: Array2<u64> = arr.try_convert().unwrap();
        assert_eq!(res, src.t().mapv(|x| x as u64));
        let arr = DynArray::Bool(array![true, false].into_dyn());
        let res: Array1<usize> = arr.try_convert().unwrap();
        assert_eq!(res, array![1, 0]);
    }
}
