use super::backend::Backend;
use super::types::TaggedNdArray;
use crate::path::Path;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Parameters<B: Backend>(pub HashMap<Path, TaggedNdArray<B>>);

// Needed so Backend doesn't have to implement Default
impl<B: Backend> Default for Parameters<B> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<B: Backend> From<HashMap<Path, TaggedNdArray<B>>> for Parameters<B> {
    fn from(map: HashMap<Path, TaggedNdArray<B>>) -> Self {
        Parameters(map)
    }
}

impl<const N: usize, B: Backend> From<[(Path, TaggedNdArray<B>); N]> for Parameters<B> {
    fn from(arr: [(Path, TaggedNdArray<B>); N]) -> Self {
        Parameters(HashMap::from(arr))
    }
}

impl<'a, B: Backend> IntoIterator for &'a Parameters<B> {
    type Item = &'a Path;
    type IntoIter = std::collections::hash_map::Keys<'a, Path, TaggedNdArray<B>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.keys()
    }
}

impl<B: Backend> Parameters<B> {
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, Path, TaggedNdArray<B>> {
        self.0.keys()
    }
}
