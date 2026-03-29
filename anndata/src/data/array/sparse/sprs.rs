use std::collections::HashMap;

use anyhow::bail;
use ndarray::Ix1;
use num::ToPrimitive;
use sprs::{CsMatI, SpIndex};

use crate::{
    HasShape, Readable, ReadableArray, Selectable, Writable, WritableArray,
    backend::{AttributeOp, BackendData, DataType, DatasetOp, GroupOp},
    data::{Element, MetaData, Shape, Stackable, SelectInfoElemBounds, SelectInfoElem},
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

        metadata.insert("shape".to_string(), HasShape::shape(self).into());
        let mtx_type = if self.is_csc() {
            "csc_matrix"
        } else {
            "csr_matrix"
        };
        MetaData::new(mtx_type, "0.1.0", Some(metadata))
    }
}

impl<N: BackendData, T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive>
    Selectable for CsMatI<N, T, u64>
{
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<crate::data::SelectInfoElem>,
    {
        let is_csr = self.is_csr();
        let rows = self.rows();
        let cols = self.cols();
        
        let row_bounds = SelectInfoElemBounds::new(info[0].as_ref(), rows);
        let full = SelectInfoElem::full();
        let col_bounds = if info.len() > 1 {
            SelectInfoElemBounds::new(info[1].as_ref(), cols)
        } else {
            SelectInfoElemBounds::new(&full, cols)
        };

        if is_csr {
            // CSR Selection
            let indptr_obj = self.indptr();
            let indptr_slice = indptr_obj.as_slice().unwrap();
            let mut new_indptr = Vec::with_capacity(row_bounds.len() + 1);
            let mut temp_indices = Vec::new();
            let mut temp_data = Vec::new();
            new_indptr.push(0);

            // Check if minor axis selection is monotonically increasing
            let is_monotonic = col_bounds.iter().zip(col_bounds.iter().skip(1)).all(|(a, b)| a < b);

            // Use SmallVec to avoid heap allocations for unique indices
            let mut col_lookup: HashMap<usize, smallvec::SmallVec<[usize; 1]>> = HashMap::new();
            for (i, x) in col_bounds.iter().enumerate() {
                col_lookup.entry(x).or_default().push(i);
            }
            
            let mut row_workspace: Vec<(usize, N)> = Vec::new();

            for i in 0..row_bounds.len() {
                let row_idx = row_bounds.index(i);
                let start = indptr_slice[row_idx] as usize;
                let end = indptr_slice[row_idx + 1] as usize;
                
                if is_monotonic {
                    // Fast path: skip sorting
                    for j in start..end {
                        let col_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_col_indices) = col_lookup.get(&col_idx) {
                            for &new_col_idx in new_col_indices {
                                temp_indices.push(SpIndex::from_usize(new_col_idx));
                                temp_data.push(self.data()[j].clone());
                            }
                        }
                    }
                } else {
                    // Slow path: out-of-order or duplicate selection requires sorting
                    row_workspace.clear();
                    for j in start..end {
                        let col_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_col_indices) = col_lookup.get(&col_idx) {
                            for &new_col_idx in new_col_indices {
                                row_workspace.push((new_col_idx, self.data()[j].clone()));
                            }
                        }
                    }
                    row_workspace.sort_by_key(|x| x.0);
                    for (new_col_idx, val) in &row_workspace {
                        temp_indices.push(SpIndex::from_usize(*new_col_idx));
                        temp_data.push(val.clone());
                    }
                }
                new_indptr.push(temp_indices.len() as u64);
            }
            CsMatI::try_new((row_bounds.len(), col_bounds.len()), new_indptr, temp_indices, temp_data).unwrap()
        } else {
            // CSC Selection
            let indptr_obj = self.indptr();
            let indptr_slice = indptr_obj.as_slice().unwrap();
            let mut new_indptr = Vec::with_capacity(col_bounds.len() + 1);
            let mut temp_indices = Vec::new();
            let mut temp_data = Vec::new();
            new_indptr.push(0);

            // Check if minor axis selection is monotonically increasing
            let is_monotonic = row_bounds.iter().zip(row_bounds.iter().skip(1)).all(|(a, b)| a < b);

            // Use SmallVec to avoid heap allocations for unique indices
            let mut row_lookup: HashMap<usize, smallvec::SmallVec<[usize; 1]>> = HashMap::new();
            for (i, x) in row_bounds.iter().enumerate() {
                row_lookup.entry(x).or_default().push(i);
            }
            
            let mut col_workspace: Vec<(usize, N)> = Vec::new();

            for i in 0..col_bounds.len() {
                let col_idx = col_bounds.index(i);
                let start = indptr_slice[col_idx] as usize;
                let end = indptr_slice[col_idx + 1] as usize;
                
                if is_monotonic {
                    // Fast path: skip sorting
                    for j in start..end {
                        let row_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_row_indices) = row_lookup.get(&row_idx) {
                            for &new_row_idx in new_row_indices {
                                temp_indices.push(SpIndex::from_usize(new_row_idx));
                                temp_data.push(self.data()[j].clone());
                            }
                        }
                    }
                } else {
                    // Slow path: out-of-order or duplicate selection requires sorting
                    col_workspace.clear();
                    for j in start..end {
                        let row_idx = self.indices()[j].to_usize().unwrap();
                        if let Some(new_row_indices) = row_lookup.get(&row_idx) {
                            for &new_row_idx in new_row_indices {
                                col_workspace.push((new_row_idx, self.data()[j].clone()));
                            }
                        }
                    }
                    col_workspace.sort_by_key(|x| x.0);
                    for (new_row_idx, val) in &col_workspace {
                        temp_indices.push(SpIndex::from_usize(*new_row_idx));
                        temp_data.push(val.clone());
                    }
                }
                new_indptr.push(temp_indices.len() as u64);
            }
            CsMatI::try_new_csc((row_bounds.len(), col_bounds.len()), new_indptr, temp_indices, temp_data).unwrap()
        }
    }
}

impl<N: BackendData + std::fmt::Debug, T: BackendData + SpIndex + num::Integer + num::FromPrimitive>
    Stackable for CsMatI<N, T, u64>
{
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> anyhow::Result<Self> {
        let mut iter = iter.peekable();
        let first = iter.peek().ok_or(anyhow::anyhow!("Empty iterator"))?;
        let is_csr = first.is_csr();
        let cols = first.cols();

        if !is_csr {
            bail!("vstack is only implemented for CSR matrices");
        }

        let mut new_rows = 0;
        let mut total_nnz = 0;
        let mut matrices = Vec::new();

        for m in iter {
            if !m.is_csr() {
                bail!("Cannot vstack matrices with different layouts (CSR vs CSC)");
            }
            if m.cols() != cols {
                bail!("Cannot vstack matrices with different number of columns");
            }
            new_rows += m.rows();
            total_nnz += m.nnz();
            matrices.push(m);
        }

        let mut new_indptr = Vec::with_capacity(new_rows + 1);
        let mut new_indices = Vec::with_capacity(total_nnz);
        let mut new_data = Vec::with_capacity(total_nnz);

        new_indptr.push(0);
        let mut current_nnz: u64 = 0;

        for m in matrices {
            let indptr = m.indptr();
            let indptr_slice = indptr.as_slice().unwrap();
            for &p in &indptr_slice[1..] {
                new_indptr.push(current_nnz + p);
            }
            current_nnz += m.nnz() as u64;
            new_indices.extend_from_slice(m.indices());
            new_data.extend_from_slice(m.data());
        }

        Ok(CsMatI::new((new_rows, cols), new_indptr, new_indices, new_data))
    }
}

impl<N: BackendData, T: BackendData + SpIndex> Writable for CsMatI<N, T, u64> {
    fn write<B: crate::Backend, G: crate::backend::GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> anyhow::Result<crate::backend::DataContainer<B>> {
        let mut group = location.new_group(name)?;

        self.metadata().save(&mut group)?;
        group.new_array_dataset("data", self.data().into(), Default::default())?;

        let indptr = self.indptr();
        group.new_array_dataset(
            "indptr",
            indptr.as_slice().unwrap().into(),
            Default::default(),
        )?;
        let min_idx = self.indices();
        group.new_array_dataset("indices", min_idx.into(), Default::default())?;
        Ok(crate::backend::DataContainer::Group(group))
    }
}

impl<N: BackendData, T: BackendData + SpIndex + num::FromPrimitive> Readable for CsMatI<N, T, u64> {
    fn read<B: crate::Backend>(container: &crate::backend::DataContainer<B>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let data_type = container.encoding_type()?;

        match data_type {
            DataType::CsrMatrix(_, _) => {
                let group = container.as_group()?;
                let shape: Vec<u64> = group.get_attr("shape")?;
                let data: Vec<N> = group
                    .open_dataset("data")?
                    .read_array::<_, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;

                // Read indptr as u64 and indices as i64 to be robust against what builders write
                let indptr: Vec<u64> = group
                    .open_dataset("indptr")?
                    .read_array_cast::<u64, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;
                let indices: Vec<T> = group
                    .open_dataset("indices")?
                    .read_array_cast::<i64, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0
                    .into_iter()
                    .map(|x| T::from_i64(x).unwrap())
                    .collect();
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
                    .read_array_cast::<u64, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0;
                let indices: Vec<T> = group
                    .open_dataset("indices")?
                    .read_array_cast::<i64, Ix1>()?
                    .into_raw_vec_and_offset()
                    .0
                    .into_iter()
                    .map(|x| T::from_i64(x).unwrap())
                    .collect();
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

impl<N: BackendData, T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive>
    ReadableArray for CsMatI<N, T, u64>
{
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
            DataType::CsrMatrix(_, _) | DataType::CscMatrix(_, _) => {
                let info = info.as_ref();
                let is_full = info.iter().all(|x| x.as_ref().is_full());
                if is_full {
                    return Self::read(container);
                }

                // TODO: optimized read_select
                return Ok(Self::read(container)?.select(info));
            }
            _ => bail!("Cannot read sparse matrix from group."),
        }
    }
}

impl<N: BackendData, T: BackendData + SpIndex> WritableArray for &CsMatI<N, T, u64> {}
impl<N: BackendData, T: BackendData + SpIndex> WritableArray for CsMatI<N, T, u64> {}

impl<N: BackendData, T: BackendData + SpIndex> crate::data::data_traits::Indexable
    for CsMatI<N, T, u64>
{
    fn get(&self, index: &[usize]) -> Option<crate::data::DynScalar> {
        self.get(index[0], index[1]).map(|v| v.into_dyn())
    }
}

impl<N: BackendData + ToPrimitive, T: BackendData + SpIndex> crate::data::data_traits::ArrayArithmetic
    for CsMatI<N, T, u64>
{
    fn sum(&self) -> f64 {
        self.data().iter().map(|x| x.to_f64().unwrap()).sum()
    }

    fn sum_axis(&self, axis: usize) -> anyhow::Result<ndarray::ArrayD<f64>> {
        let rows = self.rows();
        let cols = self.cols();
        let indptr_obj = self.indptr();
        let indptr_slice = indptr_obj.as_slice().unwrap();
        if self.is_csr() {
            if axis == 0 {
                let mut col_sums = vec![0.0; cols];
                for i in 0..rows {
                    let start = indptr_slice[i] as usize;
                    let end = indptr_slice[i + 1] as usize;
                    for (&col, val) in self.indices()[start..end].iter().zip(&self.data()[start..end]) {
                        col_sums[col.to_usize().unwrap()] += val.to_f64().unwrap();
                    }
                }
                Ok(ndarray::Array1::from_vec(col_sums).into_dyn())
            } else if axis == 1 {
                let row_sums: Vec<f64> = (0..rows)
                    .map(|i| {
                        let start = indptr_slice[i] as usize;
                        let end = indptr_slice[i + 1] as usize;
                        self.data()[start..end]
                            .iter()
                            .map(|x| x.to_f64().unwrap())
                            .sum()
                    })
                    .collect();
                Ok(ndarray::Array1::from_vec(row_sums).into_dyn())
            } else {
                bail!("Axis {} out of bounds for 2D matrix", axis);
            }
        } else {
            if axis == 0 {
                let col_sums: Vec<f64> = (0..cols)
                    .map(|i| {
                        let start = indptr_slice[i] as usize;
                        let end = indptr_slice[i + 1] as usize;
                        self.data()[start..end]
                            .iter()
                            .map(|x| x.to_f64().unwrap())
                            .sum()
                    })
                    .collect();
                Ok(ndarray::Array1::from_vec(col_sums).into_dyn())
            } else if axis == 1 {
                let mut row_sums = vec![0.0; rows];
                for i in 0..cols {
                    let start = indptr_slice[i] as usize;
                    let end = indptr_slice[i + 1] as usize;
                    for (&row, val) in self.indices()[start..end].iter().zip(&self.data()[start..end]) {
                        row_sums[row.to_usize().unwrap()] += val.to_f64().unwrap();
                    }
                }
                Ok(ndarray::Array1::from_vec(row_sums).into_dyn())
            } else {
                bail!("Axis {} out of bounds for 2D matrix", axis);
            }
        }
    }

    fn min(&self) -> f64 {
        self.data()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.data()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl<N, T: SpIndex> crate::data::data_traits::SparseMatrixLayout for CsMatI<N, T, u64> {
    fn get_sparse_layout(&self) -> crate::data::SparseMatrixLayoutE {
        if self.is_csr() {
            crate::data::SparseMatrixLayoutE::CSR
        } else {
            crate::data::SparseMatrixLayoutE::CSC
        }
    }
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn test_vstack() {
        let m1: CsMatI<f64, u32, u64> = CsMatI::new(
            (2, 3),
            vec![0, 1, 2],
            vec![0, 1],
            vec![1.0, 2.0]
        );
        let m2: CsMatI<f64, u32, u64> = CsMatI::new(
            (1, 3),
            vec![0, 1],
            vec![2],
            vec![3.0]
        );

        let stacked = CsMatI::vstack(vec![m1.clone(), m2.clone()].into_iter()).unwrap();
        assert_eq!(stacked.rows(), 3);
        assert_eq!(stacked.cols(), 3);
        assert_eq!(stacked.nnz(), 3);
        assert_eq!(stacked.indptr().as_slice().unwrap(), &[0, 1, 2, 3]);
        assert_eq!(stacked.indices(), &[0, 1, 2]);
        assert_eq!(stacked.data(), &[1.0, 2.0, 3.0]);

        // Verify values
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(stacked.get(i, j), m1.get(i, j));
            }
        }
        for j in 0..3 {
            assert_eq!(stacked.get(2, j), m2.get(0, j));
        }
    }
}
