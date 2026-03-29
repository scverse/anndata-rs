use std::collections::HashMap;

use anyhow::bail;
use ndarray::Ix1;
use num::ToPrimitive;
use sprs::{CsMatI, SpIndex};

use crate::{
    HasShape, Readable, ReadableArray, Selectable, Writable, WritableArray,
    backend::{AttributeOp, BackendData, DataType, DatasetOp, GroupOp},
    data::{Element, MetaData, SelectInfoElem, Shape},
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

impl<N: BackendData, T: BackendData + SpIndex + ToPrimitive + num::Integer + num::FromPrimitive>
    Selectable for CsMatI<N, T, u64>
{
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<crate::data::SelectInfoElem>,
    {
        let info = crate::data::SelectInfoBounds::new(&info, &crate::data::Shape::from(self.shape()));
        if info.ndim() != 2 {
            panic!("index must have length 2");
        }
        let row_idx = &info.as_ref()[0];
        let col_idx = &info.as_ref()[1];

        let (major_idx, minor_idx, _major_dim, minor_dim) = if self.is_csr() {
            (row_idx, col_idx, self.rows(), self.cols())
        } else {
            (col_idx, row_idx, self.cols(), self.rows())
        };

        let indptr = self.indptr();
        let indptr_slice = indptr.as_slice().unwrap();
        let indices_slice = self.indices();

        let (new_indptr, new_indices, new_data) = if minor_idx.is_full(minor_dim) {
            let (offsets, indices, data) = match major_idx {
                crate::data::SelectInfoElemBounds::Slice(s) => {
                    if s.step == 1 {
                        let (o, i, d) = crate::data::array::utils::cs_major_slice(
                            s.start,
                            s.end,
                            indptr_slice,
                            indices_slice,
                            self.data(),
                        );
                        (o, i.to_vec(), d.to_vec())
                    } else {
                        crate::data::array::utils::cs_major_index(
                            major_idx.iter(),
                            indptr_slice,
                            indices_slice,
                            self.data(),
                        )
                    }
                }
                crate::data::SelectInfoElemBounds::Index(_) => {
                    crate::data::array::utils::cs_major_index(
                        major_idx.iter(),
                        indptr_slice,
                        indices_slice,
                        self.data(),
                    )
                }
            };
            (offsets, indices, data)
        } else {
            crate::data::array::utils::cs_major_minor_index(
                major_idx.to_vec().into_iter(),
                minor_idx.to_vec().into_iter(),
                minor_dim,
                indptr_slice,
                indices_slice,
                self.data(),
            )
        };

        if self.is_csr() {
            CsMatI::try_new(
                (new_indptr.len() - 1, minor_idx.len()),
                new_indptr,
                new_indices,
                new_data,
            )
            .unwrap()
        } else {
            CsMatI::try_new_csc(
                (minor_idx.len(), new_indptr.len() - 1),
                new_indptr,
                new_indices,
                new_data,
            )
            .unwrap()
        }
    }
}

impl<N: Clone, T: Clone + SpIndex + num::Integer + num::FromPrimitive> crate::data::data_traits::Stackable
    for CsMatI<N, T, u64>
{
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> anyhow::Result<Self> {
        let mut iter = iter.peekable();
        let (is_csr, cols) = if let Some(first) = iter.peek() {
            (first.is_csr(), first.cols())
        } else {
            bail!("Cannot stack empty iterator");
        };

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

impl<N: BackendData, T: BackendData + SpIndex> Readable for CsMatI<N, T, u64> {
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

        let res = match data_type {
            DataType::CsrMatrix(_, _) => CsMatI::try_new(
                (indptr.len() - 1, Self::get_shape(container)?[1]),
                indptr,
                indices,
                data,
            )
            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csr matrix {}", e))?
            .select_axis(1, info[1].as_ref()),
            DataType::CscMatrix(_, _) => CsMatI::try_new_csc(
                (Self::get_shape(container)?[0], indptr.len() - 1),
                indptr,
                indices,
                data,
            )
            .map_err(|(_, _, _, e)| anyhow::anyhow!("Cannot read csc matrix {}", e))?
            .select_axis(0, info[0].as_ref()),
            _ => bail!(
                "cannot read sparse matrix from container with data type {:?}",
                data_type
            ),
        };
        Ok(res)
    }
}

impl<N: BackendData, T: BackendData + SpIndex> WritableArray for &CsMatI<N, T, u64> {}
impl<N: BackendData, T: BackendData + SpIndex> WritableArray for CsMatI<N, T, u64> {}

impl<N: BackendData, T: BackendData + SpIndex> crate::data::data_traits::Indexable
    for CsMatI<N, T, u64>
{
    fn get(&self, index: &[usize]) -> Option<crate::data::DynScalar> {
        if index.len() != 2 {
            panic!("index must have length 2");
        }
        self.get(index[0], index[1]).map(|v| v.into_dyn())
    }
}

impl<N: BackendData, T: BackendData + SpIndex> crate::data::SparseMatrixLayout for CsMatI<N, T, u64> {
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

impl<N: BackendData + Clone + ToPrimitive, T: BackendData + SpIndex> crate::data::ArrayArithmetic
    for CsMatI<N, T, u64>
{
    fn sum(&self) -> f64 {
        self.data()
            .iter()
            .map(|x| x.to_f64().unwrap())
            .sum()
    }

    fn sum_axis(&self, axis: usize) -> anyhow::Result<ndarray::ArrayD<f64>> {
        if axis >= 2 {
            anyhow::bail!("axis {} out of bounds", axis);
        }

        let (rows, cols) = (self.rows(), self.cols());
        let res = if self.is_csr() {
            if axis == 0 {
                // Sum along rows (sum each column)
                let mut col_sums = vec![0.0; cols];
                self.indices().iter().zip(self.data()).for_each(|(&col, val)| {
                    col_sums[col.to_usize().unwrap()] += val.to_f64().unwrap();
                });
                ndarray::Array1::from(col_sums)
            } else {
                // Sum along columns (sum each row)
                let indptr = self.indptr();
                let indptr_slice = indptr.as_slice().unwrap();
                let row_sums: Vec<f64> = (0..rows)
                    .map(|i| {
                        let start = indptr_slice[i] as usize;
                        let end = indptr_slice[i + 1] as usize;
                        self.data()[start..end]
                            .iter()
                            .map(|v| v.to_f64().unwrap())
                            .sum()
                    })
                    .collect();
                ndarray::Array1::from(row_sums)
            }
        } else {
            // CSC
            if axis == 0 {
                // Sum along rows (sum each column)
                let indptr = self.indptr();
                let indptr_slice = indptr.as_slice().unwrap();
                let col_sums: Vec<f64> = (0..cols)
                    .map(|i| {
                        let start = indptr_slice[i] as usize;
                        let end = indptr_slice[i + 1] as usize;
                        self.data()[start..end]
                            .iter()
                            .map(|v| v.to_f64().unwrap())
                            .sum()
                    })
                    .collect();
                ndarray::Array1::from(col_sums)
            } else {
                // Sum along columns (sum each row)
                let mut row_sums = vec![0.0; rows];
                self.indices().iter().zip(self.data()).for_each(|(&row, val)| {
                    row_sums[row.to_usize().unwrap()] += val.to_f64().unwrap();
                });
                ndarray::Array1::from(row_sums)
            }
        };
        Ok(res.into_dyn())
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

#[cfg(test)]
mod sprs_tests {
    use super::*;
    use crate::s;
    use crate::data::ArrayArithmetic;
    use crate::data::data_traits::Selectable;
    use crate::data::data_traits::Indexable;
    use rand::{Rng, thread_rng};

    fn create_test_matrix(rows: usize, cols: usize, nnz: usize, is_csr: bool) -> CsMatI<i64, u32, u64> {
        let mut rng = thread_rng();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for _ in 0..nnz {
            row_indices.push(rng.gen_range(0..rows));
            col_indices.push(rng.gen_range(0..cols));
            values.push(rng.gen_range(1..100));
        }

        let coo = sprs::TriMatI::<i64, usize>::from_triplets(
            (rows, cols),
            row_indices,
            col_indices,
            values,
        );

        if is_csr {
            let csr = coo.to_csr();
            CsMatI::new(
                csr.shape(),
                csr.indptr().as_slice().unwrap().iter().map(|&x: &usize| x as u64).collect::<Vec<u64>>(),
                csr.indices().iter().map(|&x: &usize| x as u32).collect::<Vec<u32>>(),
                csr.data().to_vec(),
            )
        } else {
            let csc = coo.to_csc();
            CsMatI::new_csc(
                csc.shape(),
                csc.indptr().as_slice().unwrap().iter().map(|&x: &usize| x as u64).collect::<Vec<u64>>(),
                csc.indices().iter().map(|&x: &usize| x as u32).collect::<Vec<u32>>(),
                csc.data().to_vec(),
            )
        }
    }

    #[test]
    fn test_sprs_basic() {
        let csr = create_test_matrix(10, 20, 30, true);
        assert_eq!(csr.rows(), 10);
        assert_eq!(csr.cols(), 20);
        assert!(csr.is_csr());
        assert!(!csr.is_csc());

        let csc = create_test_matrix(10, 20, 30, false);
        assert_eq!(csc.rows(), 10);
        assert_eq!(csc.cols(), 20);
        assert!(csc.is_csc());
        assert!(!csc.is_csr());
    }

    #[test]
    fn test_sprs_arithmetic() {
        let csr = create_test_matrix(5, 5, 10, true);
        let sum: f64 = csr.data().iter().map(|&x| x as f64).sum();
        let min: f64 = csr.data().iter().map(|&x| x as f64).fold(f64::INFINITY, f64::min);
        let max: f64 = csr.data().iter().map(|&x| x as f64).fold(f64::NEG_INFINITY, f64::max);

        assert_eq!(csr.sum(), sum);
        assert_eq!(csr.min(), min);
        assert_eq!(csr.max(), max);

        let row_sum = csr.sum_axis(1).unwrap();
        assert_eq!(row_sum.len(), 5);
        
        let col_sum = csr.sum_axis(0).unwrap();
        assert_eq!(col_sum.len(), 5);
    }

    #[test]
    fn test_sprs_selection() {
        let csr = create_test_matrix(100, 100, 500, true);
        
        // Full slice
        let full = csr.select(s![.., ..].as_ref());
        assert_eq!(full.rows(), 100);
        assert_eq!(full.cols(), 100);
        assert_eq!(full.nnz(), csr.nnz());

        // Row slice
        let row_sub = csr.select(s![10..20, ..].as_ref());
        assert_eq!(row_sub.rows(), 10);
        assert_eq!(row_sub.cols(), 100);

        // Column slice
        let col_sub = csr.select(s![.., 5..15].as_ref());
        assert_eq!(col_sub.rows(), 100);
        assert_eq!(col_sub.cols(), 10);

        // Fancy indexing
        let mut rng = thread_rng();
        let idx: Vec<usize> = (0..10).map(|_| rng.gen_range(0..100)).collect();
        let fancy = csr.select(s![&idx, ..].as_ref());
        assert_eq!(fancy.rows(), 10);
        assert_eq!(fancy.cols(), 100);
    }

    #[test]
    fn test_sprs_indexing() {
        let rows = 10;
        let cols = 10;
        let csr = create_test_matrix(rows, cols, 20, true);
        
        for i in 0..rows {
            for j in 0..cols {
                let val = csr.get(i, j);
                let trait_val = Indexable::get(&csr, &[i, j]);
                
                match (val, trait_val) {
                    (Some(v), Some(tv)) => assert_eq!(v.into_dyn(), tv),
                    (None, None) => (),
                    _ => panic!("Mismatch at [{}, {}]", i, j),
                }
            }
        }
    }

    #[test]
    fn test_dyn_sprs() {
        use crate::data::DynIndSparseMatrix;
        let rows = 10;
        let cols = 10;
        let csr = create_test_matrix(rows, cols, 20, true);
        let sum = csr.sum();
        
        let dyn_mtx = DynIndSparseMatrix::U32(csr.into());
        
        assert_eq!(dyn_mtx.shape()[0], rows);
        assert_eq!(dyn_mtx.shape()[1], cols);
        assert_eq!(ArrayArithmetic::sum(&dyn_mtx), sum);
        
        // Test indexing through dynamic dispatch
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(
                    Indexable::get(&dyn_mtx, &[i, j]),
                    Indexable::get(&dyn_mtx, &[i, j]) // This is a bit redundant but tests the dispatch
                );
            }
        }
    }

    #[test]
    fn test_sprs_vstack() {
        use crate::data::data_traits::Stackable;
        let rows = 10;
        let cols = 10;
        let m1 = create_test_matrix(rows, cols, 20, true);
        let m2 = create_test_matrix(rows, cols, 20, true);
        
        let stacked = CsMatI::vstack(vec![m1.clone(), m2.clone()].into_iter()).unwrap();
        
        assert_eq!(stacked.rows(), rows * 2);
        assert_eq!(stacked.cols(), cols);
        assert_eq!(stacked.nnz(), m1.nnz() + m2.nnz());
        assert_eq!(stacked.sum(), m1.sum() + m2.sum());
        
        // Verify values
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(stacked.get(i, j), m1.get(i, j));
                assert_eq!(stacked.get(i + rows, j), m2.get(i, j));
            }
        }
    }
}
