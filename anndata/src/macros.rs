macro_rules! dyn_match {
    ($scalar:expr, $enum:ident, $inner_macro:ident) => {
        match $scalar {
            $enum::I8 => $inner_macro!(i8),
            $enum::I16 => $inner_macro!(i16),
            $enum::I32 => $inner_macro!(i32),
            $enum::I64 => $inner_macro!(i64),
            $enum::U8 => $inner_macro!(u8),
            $enum::U16 => $inner_macro!(u16),
            $enum::U32 => $inner_macro!(u32),
            $enum::U64 => $inner_macro!(u64),
            $enum::F32 => $inner_macro!(f32),
            $enum::F64 => $inner_macro!(f64),
            $enum::Bool => $inner_macro!(bool),
            $enum::String => $inner_macro!(String),
        }
    };
}

/// Extra arguments are passed on to the inner macro after the variant and value.
macro_rules! dyn_map {
    ($scalar:expr, $enum:ident, $inner_macro:ident $(, $arg:tt)*) => {
        match $scalar {
            $enum::I8(_val) => $inner_macro!(I8, _val $(, $arg)*),
            $enum::I16(_val) => $inner_macro!(I16, _val $(, $arg)*),
            $enum::I32(_val) => $inner_macro!(I32, _val $(, $arg)*),
            $enum::I64(_val) => $inner_macro!(I64, _val $(, $arg)*),
            $enum::U8(_val) => $inner_macro!(U8, _val $(, $arg)*),
            $enum::U16(_val) => $inner_macro!(U16, _val $(, $arg)*),
            $enum::U32(_val) => $inner_macro!(U32, _val $(, $arg)*),
            $enum::U64(_val) => $inner_macro!(U64, _val $(, $arg)*),
            $enum::F32(_val) => $inner_macro!(F32, _val $(, $arg)*),
            $enum::F64(_val) => $inner_macro!(F64, _val $(, $arg)*),
            $enum::Bool(_val) => $inner_macro!(Bool, _val $(, $arg)*),
            $enum::String(_val) => $inner_macro!(String, _val $(, $arg)*),
        }
    };
}

/// Inner macro for `dyn_map!` calling an `ArrayArithmetic` method,
/// or `$fail!` (`panic` or `bail`) for non-numeric variants.
/// Usage: `dyn_map!(self, DynArray, arith, bail, sum_axis, axis)`
macro_rules! arith {
    (Bool, $val:expr, $fail:ident, $method:ident $(, $arg:expr)*) => {
        $fail!("Cannot compute {} for {}", stringify!($method), $val.data_type())
    };
    (String, $val:expr, $fail:ident, $method:ident $(, $arg:expr)*) => {
        $fail!("Cannot compute {} for {}", stringify!($method), $val.data_type())
    };
    ($variant:ident, $val:expr, $fail:ident, $method:ident $(, $arg:expr)*) => {
        ArrayArithmetic::$method($val $(, $arg)*)
    };
}

macro_rules! dyn_map_fun {
    ($scalar:expr, $enum:ident, $fun:ident $(, $arg:expr)*) => {
        match $scalar {
            $enum::I8(_val) => _val.$fun($($arg),*),
            $enum::I16(_val) => _val.$fun($($arg),*),
            $enum::I32(_val) => _val.$fun($($arg),*),
            $enum::I64(_val) => _val.$fun($($arg),*),
            $enum::U8(_val) => _val.$fun($($arg),*),
            $enum::U16(_val) => _val.$fun($($arg),*),
            $enum::U32(_val) => _val.$fun($($arg),*),
            $enum::U64(_val) => _val.$fun($($arg),*),
            $enum::F32(_val) => _val.$fun($($arg),*),
            $enum::F64(_val) => _val.$fun($($arg),*),
            $enum::Bool(_val) => _val.$fun($($arg),*),
            $enum::String(_val) => _val.$fun($($arg),*),
        }
    };
}

pub(crate) use {arith, dyn_map, dyn_map_fun, dyn_match};
