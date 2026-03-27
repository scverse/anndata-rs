use std::collections::HashMap;

use anyhow::bail;
use ndarray::Ix1;
use sprs::{CsMat, CsMatI, SpIndex};

use crate::{
    HasShape, Readable, ReadableArray, Selectable, Writable, WritableArray,
    backend::{AttributeOp, BackendData, DataType, DatasetOp, GroupOp},
    data::{Element, MetaData, SelectInfo, SelectInfoElem, Shape, SparseMatrixLayout},
};

impl<N, T: SpIndex> HasShape for CsMatI<N, T, u64> {
    fn shape(&self) -> Shape {
        let rows = self.rows();
        let cols = self.cols();
        vec![rows, cols].into()
    }
}

impl<N: BackendData, T: BackendData + SpIndex> Element for CsMatI<N, T, u64> {
    fn data_type(&self) -> crate::backend::DataType {
        if self.is_csr() {
            DataType::CsrMatrix(N::DTYPE, T::DTYPE)
        } else {
            DataType::CscMatrix(N::DTYPE, T::DTYPE)
        }
    }

    fn metadata(&self) -> crate::data::MetaData {
        let mut metadata = HashMap::new();

        metadata.insert("shape".to_string(), HasShape::shape(&self).into());
        let mtx_type = if self.is_csc() {
            "csc_matrix"
        } else {
            "csr_matrix"
        };
        MetaData::new(mtx_type, "0.1.0", Some(metadata))
    }
}

impl<N: Clone, T: Clone + SpIndex> Selectable for CsMatI<N, T, u64> {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<crate::data::SelectInfoElem>,
    {
        todo!()
    }

    fn select_axis<S>(&self, axis: usize, slice: S) -> Self
    where
        S: AsRef<crate::data::SelectInfoElem>,
        Self: Sized,
    {
        let full = crate::data::SelectInfoElem::full();
        let selection = slice.as_ref().set_axis(axis, 2, &full);
        self.select(selection.as_slice())
    }
}

impl<N: BackendData, T: BackendData + SpIndex> Writable for CsMatI<N, T, u64> {
    fn write<B: crate::Backend, G: crate::backend::GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> anyhow::Result<crate::backend::DataContainer<B>> {
        let mut group = location.new_group(name)?;
        let shape = self.shape();

        self.metadata().save(&mut group);
        group.new_array_dataset("data", self.data().into(), Default::default());

        //let num_cols = shape.1;
        let indptr = self.indptr();
        group.new_array_dataset(
            "indptr",
            indptr.as_slice().unwrap().into(),
            Default::default(),
        );
        let min_idx = self.indices();
        group.new_array_dataset("indices", min_idx.into(), Default::default());
        Ok(crate::backend::DataContainer::Group(group))
    }
}

impl<N: BackendData, T: BackendData + SpIndex> Readable for CsMatI<N, T, u64> {
    fn read<B: crate::Backend>(container: &crate::backend::DataContainer<B>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let data_type = container.encoding_type()?;

        match (data_type) {
            DataType::CsrMatrix(_, _) => {
                let group = container.as_group()?;
                let shape: Vec<u64> = group.get_attr("shape")?;
                let data: Vec<N> = group
                    .open_dataset("data")?
                    .read_array::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;

                let indptr: Vec<u64> = group
                    .open_dataset("indptr")?
                    .read_array::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;
                let indices: Vec<T> = group
                    .open_dataset("indices")?
                    .read_array::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;
                CsMatI::try_new(
                    (shape[0] as usize, shape[1] as usize),
                    indptr,
                    indices,
                    data,
                )
                .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csr matrix {}", e))
            }
            DataType::CscMatrix(_, _) => {
                let group = container.as_group()?;
                let shape: Vec<u64> = group.get_attr("shape")?;
                let data: Vec<N> = group
                    .open_dataset("data")?
                    .read_array::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;

                let indptr: Vec<u64> = group
                    .open_dataset("indptr")?
                    .read_array_cast::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;
                let indices: Vec<T> = group
                    .open_dataset("indices")?
                    .read_array::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;
                CsMatI::try_new_csc(
                    (shape[0] as usize, shape[1] as usize),
                    indptr,
                    indices,
                    data,
                )
                .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csc matrix {}", e))
            }
            _ => bail!("Cannot read CSMatI from container of type {:?}", data_type),
        }
    }
}

impl<N: BackendData, T: BackendData + SpIndex> ReadableArray for CsMatI<N, T, u64> {
    fn get_shape<B: crate::Backend>(
        container: &crate::backend::DataContainer<B>,
    ) -> anyhow::Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(
        container: &crate::backend::DataContainer<B>,
        info: &[S],
    ) -> anyhow::Result<Self>
    where
        B: crate::Backend,
        S: AsRef<crate::data::SelectInfoElem>,
        Self: Sized,
    {
        let data_type = container.encoding_type()?;

        match data_type {
            DataType::CsrMatrix(_, _) => {}
            DataType::CscMatrix(_, _) => {}
            _ => bail!("Cannot read sparse matrix from group."),
        };

        if info.as_ref().len() != 2 {
            panic!("index must have length 2");
        }

        if info.iter().all(|s| s.as_ref().is_full()) {
            return Self::read(container);
        }

        let s = match data_type {
            DataType::CsrMatrix(_, _) => info[0].as_ref(),
            DataType::CscMatrix(_, _) => info[1].as_ref(),
            _ => bail!("This should definitely not happen here! E1093"),
        };

        if let SelectInfoElem::Index(_) = s {
            return Ok(Self::read(container)?.select(info));
        }

        let slc = match s {
            SelectInfoElem::Slice(s) => s,
            _ => bail!("This should definitely not happen here! E1094"),
        };

        let group = container.as_group()?;
        let indptr_slice = if let Some(end) = slc.end {
            SelectInfoElem::from(slc.start..end + 1)
        } else {
            SelectInfoElem::from(slc.start..)
        };

        let mut indptr: Vec<u64> = group
            .open_dataset("indptr")?
            .read_array_slice_cast(&[indptr_slice])?
            .to_vec();
        let lo = indptr[0] as usize;
        let end = indptr[indptr.len() - 1] as usize;
        let slice = SelectInfoElem::from(lo..end);
        let data: Vec<N> = group
            .open_dataset("data")?
            .read_array_slice(&[&slice])?
            .to_vec();
        let indices: Vec<T> = group
            .open_dataset("indices")?
            .read_array_slice(&[&slice])?
            .to_vec();
        let lo = indptr[0];
        indptr.iter_mut().for_each(|x| *x -= lo);

        match data_type {
            DataType::CsrMatrix(_, _) => CsMatI::try_new(
                (indptr.len() - 1, Self::get_shape(container)?[1]),
                indptr,
                indices,
                data,
            )
            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csr matrix {}", e)),
            DataType::CscMatrix(_, _) => CsMatI::try_new_csc(
                (Self::get_shape(container)?[0], indptr.len() - 1),
                indptr,
                indices,
                data,
            )
            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csc matrix {}", e)),
            _ => bail!(
                "cannot read sparse matrix from container with data type {:?}",
                data_type
            ),
        }
    }
}

impl<N: BackendData, T: BackendData + SpIndex> WritableArray for &CsMatI<N, T, u64> {}
impl<N: BackendData, T: BackendData + SpIndex> WritableArray for CsMatI<N, T, u64> {}

impl<N: BackendData, T: BackendData + SpIndex> SparseMatrixLayout for CsMatI<N, T, u64> {
    fn get_sparse_layout(&self) -> crate::data::SparseMatrixLayoutE {
        if self.is_csc() {
            crate::data::SparseMatrixLayoutE::CSC
        } else if self.is_csr() {
            crate::data::SparseMatrixLayoutE::CSR
        } else {
            crate::data::SparseMatrixLayoutE::NONE
        }
    }
}
