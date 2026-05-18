use anndata::{
    backend::*,
    data::{DynArray, DynCowArray, SelectInfoBounds, SelectInfoElem, SelectInfoElemBounds, Shape},
};

use anyhow::{Context, Result, bail};
use ndarray::{Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn, SliceInfoElem};
use std::{
    borrow::Cow,
    num::NonZeroU64,
    ops::{Deref, Index},
    path::{Path, PathBuf},
};
use std::{sync::Arc, vec};
use zarrs::array::{
    ZARR_NAN_F32, ZARR_NAN_F64,
    codec::{SubchunkWriteOrder, bytes_to_bytes::zstd::ZstdCodec},
    data_type::{
        BoolDataType, Float32DataType, Float64DataType, Int8DataType, Int16DataType, Int32DataType,
        Int64DataType, StringDataType, UInt8DataType, UInt16DataType, UInt32DataType,
        UInt64DataType,
    },
};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::{array::ElementOwned, storage::ReadableWritableListableStorageTraits};
use zarrs::{
    array::{
        ArrayShardedReadableExt, ArraySubset, Element, FillValue, codec::ShardingCodecBuilder,
        data_type,
    },
    storage::StorePrefix,
};

/// The Zarr backend.
pub struct Zarr;

#[derive(Clone)]
pub struct ZarrStore {
    inner: Arc<dyn ReadableWritableListableStorageTraits>,
    path: PathBuf,
}

impl Deref for ZarrStore {
    type Target = Arc<dyn ReadableWritableListableStorageTraits>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct ZarrGroup {
    group: Group<dyn ReadableWritableListableStorageTraits>,
    store: ZarrStore,
}

pub struct ZarrDataset {
    dataset: zarrs::array::Array<dyn ReadableWritableListableStorageTraits>,
    cache: zarrs::array::ArrayShardedReadableExtCache,
    store: ZarrStore,
}

impl Backend for Zarr {
    const NAME: &'static str = "zarr";

    type Store = ZarrStore;

    type Group = ZarrGroup;

    /// datasets contain arrays.
    type Dataset = ZarrDataset;

    fn new<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        if path.as_ref().try_exists()? {
            let metadata = std::fs::metadata(&path)?;
            if metadata.is_file() {
                std::fs::remove_file(&path)?;
            } else {
                std::fs::remove_dir_all(&path)?;
            }
        }

        let inner = Arc::new(FilesystemStore::new(path.as_ref())?);
        zarrs::group::GroupBuilder::new()
            .build(inner.clone(), "/")?
            .store_metadata()?;
        Ok(ZarrStore {
            path: path.as_ref().to_path_buf(),
            inner,
        })
    }

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore {
            path: path.as_ref().to_path_buf(),
            inner: Arc::new(FilesystemStore::new(path)?),
        })
    }

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore {
            path: path.as_ref().to_path_buf(),
            inner: Arc::new(FilesystemStore::new(path)?),
        })
    }
}

impl StoreOp<Zarr> for ZarrStore {
    /// Returns the file path.
    fn filename(&self) -> PathBuf {
        self.path.clone()
    }

    /// Close the file.
    fn close(self) -> Result<()> {
        drop(self);
        Ok(())
    }
}

impl GroupOp<Zarr> for ZarrStore {
    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>> {
        let result = self.list_dir(&StorePrefix::root())?;
        Ok(result
            .prefixes()
            .iter()
            .map(|x| x.as_str().trim_end_matches("/").to_string())
            .collect())
    }

    /// Create a new group.
    fn new_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = canoincalize_path(name);
        let group = zarrs::group::GroupBuilder::new().build(self.inner.clone(), &path)?;
        group.store_metadata()?;
        Ok(ZarrGroup {
            group,
            store: self.clone(),
        })
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let group = zarrs::group::Group::open(self.inner.clone(), &canoincalize_path(name))?;
        Ok(ZarrGroup {
            group,
            store: self.clone(),
        })
    }

    /// Create an empty dataset holding an array value.
    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let path = canoincalize_path(name);
        let array = new_empty_dataset_helper::<T, _>(self.inner.clone(), &path, shape, config)?;
        array.store_metadata()?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let array = zarrs::array::Array::open(self.inner.clone(), &canoincalize_path(name))?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.clone(),
        })
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        self.inner.erase_prefix(&str_to_prefix(name))?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        let path = format!("/{name}");
        Ok(zarrs::node::node_exists(
            &self.inner,
            &path.as_str().try_into()?,
        )?)
    }
}

impl GroupOp<Zarr> for ZarrGroup {
    fn list(&self) -> Result<Vec<String>> {
        let current_path = str_to_prefix(self.group.path().as_str());
        let result = self
            .store
            .list_dir(&current_path.as_str().try_into()?)?
            .prefixes()
            .iter()
            .map(|x| {
                x.as_str()
                    .strip_prefix(current_path.as_str())
                    .unwrap()
                    .strip_suffix("/")
                    .unwrap()
                    .to_owned()
            })
            .collect();
        Ok(result)
    }

    /// Create a new group.
    fn new_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = self.group.path().as_path().join(name);
        let group = zarrs::group::GroupBuilder::new()
            .build(self.store.inner.clone(), path.to_str().unwrap())?;
        group.store_metadata()?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = self.group.path().as_path().join(name);
        let group = zarrs::group::Group::open(self.store.inner.clone(), path.to_str().unwrap())?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    /// Create an empty dataset holding an array value.
    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let path = self.group.path().as_path().join(name);
        let array = new_empty_dataset_helper::<T, _>(
            self.store.inner.clone(),
            path.to_str().unwrap(),
            shape,
            config,
        )?;
        array.store_metadata()?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.store.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let path = self.group.path().as_path().join(name);
        let array = zarrs::array::Array::open(self.store.inner.clone(), path.to_str().unwrap())?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.store.clone(),
        })
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        self.store.erase_prefix(&str_to_prefix(&path))?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        let path = self
            .group
            .path()
            .as_path()
            .join(name)
            .as_os_str()
            .to_str()
            .unwrap()
            .try_into()?;
        Ok(zarrs::node::node_exists(&self.store.inner, &path)?)
    }
}

impl AttributeOp<Zarr> for ZarrGroup {
    /// Returns the Root.
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(self.store.clone())
    }

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf {
        self.group.path().as_path().to_path_buf()
    }

    /// Write an attribute at a given location.
    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        self.group
            .attributes_mut()
            .insert(name.to_string(), value.clone());
        self.group.store_metadata()?;
        Ok(())
    }

    fn get_json_attr(&self, name: &str) -> Result<Value> {
        Ok(self
            .group
            .attributes()
            .get(name)
            .with_context(|| format!("Attribute {name} not found"))?
            .clone())
    }
}

impl AttributeOp<Zarr> for ZarrDataset {
    /// Returns the Root.
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(self.store.clone())
    }

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf {
        self.dataset.path().as_path().to_path_buf()
    }

    /// Write an attribute at a given location.
    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        self.dataset
            .attributes_mut()
            .insert(name.to_string(), value.clone());
        self.dataset.store_metadata()?;
        Ok(())
    }

    fn get_json_attr(&self, name: &str) -> Result<Value> {
        Ok(self
            .dataset
            .attributes()
            .get(name)
            .with_context(|| format!("Attribute {name} not found"))?
            .clone())
    }
}

impl DatasetOp<Zarr> for ZarrDataset {
    fn dtype(&self) -> Result<ScalarType> {
        if self.dataset.data_type().is::<UInt8DataType>() {
            Ok(ScalarType::U8)
        } else if self.dataset.data_type().is::<UInt16DataType>() {
            Ok(ScalarType::U16)
        } else if self.dataset.data_type().is::<UInt32DataType>() {
            Ok(ScalarType::U32)
        } else if self.dataset.data_type().is::<UInt64DataType>() {
            Ok(ScalarType::U64)
        } else if self.dataset.data_type().is::<Int8DataType>() {
            Ok(ScalarType::I8)
        } else if self.dataset.data_type().is::<Int16DataType>() {
            Ok(ScalarType::I16)
        } else if self.dataset.data_type().is::<Int32DataType>() {
            Ok(ScalarType::I32)
        } else if self.dataset.data_type().is::<Int64DataType>() {
            Ok(ScalarType::I64)
        } else if self.dataset.data_type().is::<Float32DataType>() {
            Ok(ScalarType::F32)
        } else if self.dataset.data_type().is::<Float64DataType>() {
            Ok(ScalarType::F64)
        } else if self.dataset.data_type().is::<BoolDataType>() {
            Ok(ScalarType::Bool)
        } else if self.dataset.data_type().is::<StringDataType>() {
            Ok(ScalarType::String)
        } else {
            bail!("Unsupported type: {:?}", self.dataset.data_type())
        }
    }

    fn shape(&self) -> Shape {
        self.dataset.shape().iter().map(|x| *x as usize).collect()
    }

    fn reshape(&mut self, shape: &Shape) -> Result<()> {
        self.dataset
            .set_shape(shape.as_ref().iter().map(|x| *x as u64).collect())?;
        self.dataset.store_metadata()?;
        // The intenion of the caching API is no mutation after creation:
        // https://ossci.zulipchat.com/#narrow/channel/423692-Zarr/topic/zarrs.20.60ArrayShardedReadableExtCache.60.20.2B.20.60set_shape.60/with/595519775
        self.cache.clear();
        Ok(())
    }

    /// TODO: current implementation reads the entire array and then selects the slice.
    fn read_array_slice<T: BackendData, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn read_arr<T, S, D>(dataset: &ZarrDataset, selection: &[S]) -> Result<Array<T, D>>
        where
            T: ElementOwned + BackendData,
            S: AsRef<SelectInfoElem>,
            D: Dimension,
        {
            let sel = SelectInfoBounds::new(&selection, &dataset.shape());
            if let Some(subset) = to_array_subset(sel) {
                let arr = dataset
                    .dataset
                    .retrieve_array_subset_sharded_opt::<ndarray::ArrayD<T>>(
                        &dataset.cache,
                        &subset,
                        &zarrs::array::CodecOptions::default(),
                    )?
                    .into_dimensionality::<D>()?;
                Ok(arr)
            } else {
                // Read the entire array and then select the slice.
                let arr = dataset
                    .dataset
                    .retrieve_array_subset_sharded_opt::<ndarray::ArrayD<T>>(
                        &dataset.cache,
                        &dataset.dataset.subset_all(),
                        &zarrs::array::CodecOptions::default(),
                    )?
                    .into_dimensionality::<D>()?;
                Ok(select(arr.view(), selection))
            }
        }

        let array: DynArray = match T::DTYPE {
            ScalarType::U8 => read_arr::<u8, _, D>(self, selection)?.into(),
            ScalarType::U16 => read_arr::<u16, _, D>(self, selection)?.into(),
            ScalarType::U32 => read_arr::<u32, _, D>(self, selection)?.into(),
            ScalarType::U64 => read_arr::<u64, _, D>(self, selection)?.into(),
            ScalarType::I8 => read_arr::<i8, _, D>(self, selection)?.into(),
            ScalarType::I16 => read_arr::<i16, _, D>(self, selection)?.into(),
            ScalarType::I32 => read_arr::<i32, _, D>(self, selection)?.into(),
            ScalarType::I64 => read_arr::<i64, _, D>(self, selection)?.into(),
            ScalarType::F32 => read_arr::<f32, _, D>(self, selection)?.into(),
            ScalarType::F64 => read_arr::<f64, _, D>(self, selection)?.into(),
            ScalarType::Bool => read_arr::<bool, _, D>(self, selection)?.into(),
            ScalarType::String => read_arr::<String, _, D>(self, selection)?.into(),
        };
        Ok(BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
    }

    fn write_array_slice<S, T, D>(&self, arr: CowArray<'_, T, D>, selection: &[S]) -> Result<()>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn write_array_impl<T, S>(
            container: &ZarrDataset,
            arr: CowArray<'_, T, IxDyn>,
            selection: &[S],
        ) -> Result<()>
        where
            T: Element + 'static,
            S: AsRef<SelectInfoElem>,
        {
            let selection = SelectInfoBounds::new(&selection, &container.shape());
            let starts: Vec<_> = selection
                .iter()
                .flat_map(|x| {
                    if let SelectInfoElemBounds::Slice(slice) = x {
                        if slice.step == 1 {
                            Some(slice.start as u64)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if starts.len() == selection.ndim() {
                container.cache.clear();
                container.dataset.store_array_subset(
                    &ArraySubset::new_with_start_shape(
                        starts,
                        arr.shape().iter().map(|x| *x as u64).collect(),
                    )?,
                    arr.to_owned(),
                )?;
            } else {
                panic!("Not implemented");
            }
            Ok(())
        }

        match BackendData::into_dyn_arr(arr.into_dyn()) {
            DynCowArray::U8(x) => write_array_impl(self, x, selection),
            DynCowArray::U16(x) => write_array_impl(self, x, selection),
            DynCowArray::U32(x) => write_array_impl(self, x, selection),
            DynCowArray::U64(x) => write_array_impl(self, x, selection),
            DynCowArray::I8(x) => write_array_impl(self, x, selection),
            DynCowArray::I16(x) => write_array_impl(self, x, selection),
            DynCowArray::I32(x) => write_array_impl(self, x, selection),
            DynCowArray::I64(x) => write_array_impl(self, x, selection),
            DynCowArray::F32(x) => write_array_impl(self, x, selection),
            DynCowArray::F64(x) => write_array_impl(self, x, selection),
            DynCowArray::Bool(x) => write_array_impl(self, x, selection),
            DynCowArray::String(x) => write_array_impl(self, x, selection),
        }
    }
}

fn select<'a, S, T, D>(arr: ArrayView<'a, T, D>, info: &[S]) -> Array<T, D>
where
    S: AsRef<SelectInfoElem>,
    T: Clone,
    D: Dimension,
{
    let arr = arr.into_dyn();
    let slices = info
        .as_ref()
        .iter()
        .map(|x| match x.as_ref() {
            SelectInfoElem::Slice(slice) => Some(SliceInfoElem::from(*slice)),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    if let Some(slices) = slices {
        arr.slice(slices.as_slice()).into_owned()
    } else {
        let shape = arr.shape();
        let select: Vec<_> = info
            .as_ref()
            .iter()
            .zip(shape)
            .map(|(x, n)| SelectInfoElemBounds::new(x.as_ref(), *n))
            .collect();
        let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
        ArrayD::from_shape_fn(new_shape, |idx| {
            let new_idx: Vec<_> = (0..idx.ndim()).map(|i| select[i].index(idx[i])).collect();
            arr.index(new_idx.as_slice()).clone()
        })
    }
    .into_dimensionality::<D>()
    .unwrap()
}

fn str_to_prefix(s: &str) -> StorePrefix {
    if s.is_empty() {
        StorePrefix::root()
    } else {
        let s = s.trim_matches('/').to_string();
        StorePrefix::new((s + "/").as_str()).unwrap()
    }
}

fn canoincalize_path<'a>(path: &'a str) -> Cow<'a, str> {
    if path.starts_with("/") {
        path.into()
    } else {
        format!("/{path}").into()
    }
}

fn to_array_subset(info: SelectInfoBounds) -> Option<ArraySubset> {
    let ranges = info
        .iter()
        .map(|x| {
            if let SelectInfoElemBounds::Slice(slice) = x {
                if slice.step == 1 {
                    Some(slice.start as u64..slice.end as u64)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ArraySubset::new_with_ranges(&ranges))
}

/// a direct port of https://github.com/zarr-developers/zarr-python/blob/cdb5846c33fdc217c4ac743a5cdb3e5c54b1868c/src/zarr/core/chunk_grids.py#L831-L865
/// for a fixed shard shape
fn compute_shard_shape(chunk_size: &[u64], element_size: u64, shape: &[u64]) -> Vec<u64> {
    const TARGET_SHARD_SIZE: u64 = 1_000_000_000;
    let bytes_per_chunk = chunk_size.iter().product::<u64>() * element_size;
    if bytes_per_chunk > TARGET_SHARD_SIZE {
        return chunk_size.to_vec();
    }
    let num_axes = chunk_size.len() as u32;
    let mut chunks_per_shard = 1u64;
    while (bytes_per_chunk * (chunks_per_shard + 1).pow(num_axes)) <= TARGET_SHARD_SIZE
        && chunk_size
            .iter()
            .zip(shape.iter())
            .all(|(c, s)| (c * (chunks_per_shard + 1)) <= *s)
    {
        chunks_per_shard += 1;
    }
    chunk_size.iter().map(|c| c * chunks_per_shard).collect()
}

fn new_empty_dataset_helper<T: BackendData, S: ?Sized>(
    store: Arc<S>,
    path: &str,
    shape: &Shape,
    config: WriteConfig,
) -> Result<zarrs::array::Array<S>> {
    let (datatype, fill) = match T::DTYPE {
        ScalarType::U8 => (data_type::uint8(), FillValue::from(0u8)),
        ScalarType::U16 => (data_type::uint16(), FillValue::from(0u16)),
        ScalarType::U32 => (data_type::uint32(), FillValue::from(0u32)),
        ScalarType::U64 => (data_type::uint64(), FillValue::from(0u64)),
        ScalarType::I8 => (data_type::int8(), FillValue::from(0i8)),
        ScalarType::I16 => (data_type::int16(), FillValue::from(0i16)),
        ScalarType::I32 => (data_type::int32(), FillValue::from(0i32)),
        ScalarType::I64 => (data_type::int64(), FillValue::from(0i64)),
        ScalarType::F32 => (data_type::float32(), FillValue::from(ZARR_NAN_F32)),
        ScalarType::F64 => (data_type::float64(), FillValue::from(ZARR_NAN_F64)),
        ScalarType::Bool => (data_type::bool(), FillValue::from(false)),
        ScalarType::String => (data_type::string(), FillValue::from("")),
    };

    let shape = shape.as_ref();
    let chunk_size: Vec<u64> = match config.block_size {
        Some(s) => s
            .as_ref()
            .iter()
            .map(|x| (*x).max(1) as u64)
            .collect::<Vec<_>>(),
        _ => {
            if shape.len() == 1 {
                vec![shape[0].clamp(1, 16384) as u64]
            } else {
                shape.iter().map(|&x| x.clamp(1, 128) as u64).collect()
            }
        }
    };

    let mut use_sharding = true;
    if datatype == data_type::string() || shape.is_empty() {
        // Strings are not sharded, they are stored as a single chunk.
        // Scalars are also not sharded
        use_sharding = false;
    }

    let array = if let Some(fixed_size) = datatype.fixed_size()
        && use_sharding
    {
        let element_size = u64::try_from(fixed_size).expect("element size does not fit into u64");
        let shard_shape = compute_shard_shape(
            &chunk_size,
            element_size,
            shape
                .iter()
                .map(|e| u64::try_from(*e).unwrap())
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let mut sharding_codec_builder = ShardingCodecBuilder::new(
            chunk_size
                .iter()
                .map(|e| NonZeroU64::try_from(*e))
                .collect::<Result<Vec<NonZeroU64>, _>>()?,
            &datatype,
        );
        sharding_codec_builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(7, false))]);
        // For 1D arrays, use lexicographic ordering to improve performance for contiguous reads.
        // TODO: morton ordering for 2D, possibly: https://github.com/zarrs/zarrs/pull/364
        let mut sharding_codec = sharding_codec_builder.build();
        if shape.len() == 1 {
            sharding_codec = sharding_codec.with_subchunk_write_order(SubchunkWriteOrder::C)
        }
        zarrs::array::ArrayBuilder::new(
            shape.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            shard_shape.as_slice(),
            datatype,
            fill,
        )
        .array_to_bytes_codec(Arc::new(sharding_codec))
        .build(store, path)?
    } else {
        zarrs::array::ArrayBuilder::new(
            shape.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            chunk_size.as_slice(),
            datatype,
            fill,
        )
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(7, false))])
        .build(store, path)?
    };

    Ok(array)
}

/// test module
#[cfg(test)]
mod tests {
    use super::*;
    use anndata::s;
    use ndarray::{Array2, Axis, Ix2, array, concatenate};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use zarrs::array::ArrayShardedExt;

    pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        func(path)
    }

    fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        with_tmp_dir(|dir| func(dir.join("temp")))
    }

    #[test]
    fn test_basic_ops() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            store.open_group("/")?;

            store.new_scalar_dataset("data", &4)?;
            store.open_dataset("data")?;

            let group = store.new_group("group")?;
            assert!(store.exists("group")?);

            let subgroup = group.new_group("group")?;
            assert!(group.exists("group")?);

            let subsubgroup = subgroup.new_group("group")?;
            assert!(subgroup.exists("group")?);

            let data = subsubgroup.new_scalar_dataset("group", &4)?;
            assert!(subsubgroup.exists("group")?);
            subsubgroup.open_dataset("group")?;

            {
                let store = Zarr::open(&path)?;
                DataContainer::open(&store, "group")?;
            }

            assert_eq!(group.path(), PathBuf::from("/group"));
            assert_eq!(subgroup.path(), PathBuf::from("/group/group"));
            assert_eq!(subsubgroup.path(), PathBuf::from("/group/group/group"));
            assert_eq!(data.path(), PathBuf::from("/group/group/group/group"));
            Ok(())
        })
    }

    #[test]
    fn test_write_empty() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            let group = store.new_group("group")?;
            let config = WriteConfig {
                ..Default::default()
            };

            let empty: Array2<i64> = array![[]];
            let dataset = group.new_array_dataset("test", empty.view().into(), config)?;
            assert_eq!(empty, dataset.read_array::<i64, Ix2>()?);
            Ok(())
        })
    }

    #[test]
    fn test_write_slice() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(path)?;
            let config = WriteConfig {
                block_size: Some(vec![2, 2].as_slice().into()),
                ..Default::default()
            };

            let group = store.new_group("group")?;
            let mut dataset =
                group.new_empty_dataset::<i32>("test", &[20, 50].as_slice().into(), config)?;

            // Repeated writes force cache clearance
            let arr = Array::random((10, 10), Uniform::new(0, 100).unwrap());
            dataset.write_array_slice(arr.view().into(), s![5..15, 10..20].as_ref())?;
            assert_eq!(
                arr,
                dataset.read_array_slice::<i32, _, _>(s![5..15, 10..20].as_ref())?
            );
            let arr = Array::random((10, 10), Uniform::new(0, 100).unwrap());
            dataset.write_array_slice(arr.view().into(), s![5..15, 10..20].as_ref())?;
            assert_eq!(
                arr,
                dataset.read_array_slice::<i32, _, _>(s![5..15, 10..20].as_ref())?
            );

            // Repeatitive writes
            let arr = Array::random((20, 50), Uniform::new(0, 100).unwrap());
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;

            // Out-of-bounds writes should fail
            //assert!(dataset.write_array_slice(&arr, s![20..40, ..].as_ref()).is_err());

            // Reshape and write
            dataset.reshape(&[40, 50].as_slice().into())?;
            dataset.write_array_slice(arr.view().into(), s![20..40, ..].as_ref())?;

            // Read back is OK
            let merged = concatenate(Axis(0), &[arr.view(), arr.view()])?;
            assert_eq!(merged, dataset.read_array::<i32, _>()?);

            // Shrinking is OK
            dataset.reshape(&[20, 50].as_slice().into())?;
            assert_eq!(arr, dataset.read_array::<i32, _>()?);

            dataset.reshape(&[50, 50].as_slice().into())?;
            assert_eq!(
                [50, 50],
                store
                    .open_group("group")?
                    .open_dataset("test")?
                    .shape()
                    .as_ref(),
            );

            assert_eq!(vec!["group"], store.list()?);
            assert_eq!(vec!["test"], group.list()?);

            assert!(store.exists("group")?);
            assert!(group.exists("test")?);

            store.delete("group")?;
            assert!(!store.exists("group")?);
            assert!(!group.exists("test")?);

            Ok(())
        })
    }

    #[test]
    fn test_sharding_even() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(path)?;
            let config = WriteConfig {
                block_size: Some(vec![2, 2].as_slice().into()),
                ..Default::default()
            };

            let group = store.new_group("group")?;
            let dataset =
                group.new_empty_dataset::<i32>("test", &[50, 50].as_slice().into(), config)?;
            assert!(dataset.dataset.is_sharded());

            // At (50, 50), the shard shape is the same as the shard size because the chunking is (2, 2), which fits perfectly into the shard,
            // and each shard at (50, 50) is under a GB, so the shard shape will match the array shape.
            // Thus the chunk grid shape will be (1, 1) i.e., the ceiling of the array shape divided by shard shape.
            // zarrs considers the chunk grid to be the grid of outer chunks i.e., shards.
            let arr = Array::from_shape_vec(vec![50, 50], vec![0; 50 * 50])
                .unwrap()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;
            assert_eq!(dataset.dataset.chunk_grid_shape().to_vec(), vec![1, 1]);
            Ok(())
        })
    }

    #[test]
    fn test_sharding_uneven() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(path)?;
            let config = WriteConfig {
                block_size: Some(vec![2, 2].as_slice().into()),
                ..Default::default()
            };

            let group = store.new_group("group")?;
            let dataset =
                group.new_empty_dataset::<i32>("test", &[55, 55].as_slice().into(), config)?;
            assert!(dataset.dataset.is_sharded());
            // At (55, 55) shape with (2, 2) chunks, the the shard shape is (54, 54).
            // And each shard at (54, 54) will not match the array shape because it does not divide evenly i.e., there is a remainder on the edge.
            // Thus we have on extra shard along each axis and the chunk grid shape is thus (2, 2) i.e., the ceiling of the array shape divided by shard shape.
            // zarrs considers the chunk grid to be the grid of outer chunks i.e., shards.
            let arr = Array::from_shape_vec(vec![55, 55], vec![0; 55 * 55])
                .unwrap()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;
            assert_eq!(dataset.dataset.chunk_grid_shape().to_vec(), vec![2, 2]);
            Ok(())
        })
    }
}
