use crate::backend::{Backend, DataContainer, DataType, GroupOp, iter_containers};
use crate::data::{Data, Readable, Writable};

use anyhow::Result;
use std::collections::HashMap;
use std::ops::Deref;

use super::{Element, MetaData};

#[derive(Debug, Clone, PartialEq)]
pub struct Mapping(HashMap<String, Data>);

impl From<Mapping> for HashMap<String, Data> {
    fn from(val: Mapping) -> Self {
        val.0
    }
}

impl From<HashMap<String, Data>> for Mapping {
    fn from(data: HashMap<String, Data>) -> Self {
        Self(data)
    }
}

impl Deref for Mapping {
    type Target = HashMap<String, Data>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Element for Mapping {
    fn data_type(&self) -> DataType {
        DataType::Mapping
    }

    fn metadata(&self) -> MetaData {
        crate::data::MAPPING_ENCODING
    }
}

impl Readable for Mapping {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let data: Result<_> = iter_containers::<B>(container.as_group()?)
            .map(|(k, v)| Ok((k.to_owned(), Data::read(&v)?)))
            .collect();
        Ok(Mapping(data?))
    }
}

impl Writable for Mapping {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        self.metadata().save(&mut group)?;
        self.0
            .iter()
            .try_for_each(|(k, v)| v.write(&group, k).map(|_| ()))?;
        Ok(DataContainer::Group(group))
    }
}
