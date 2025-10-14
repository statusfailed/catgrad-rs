//use super::backend::Backend;
//use super::types::TaggedNdArray;

use super::{Interpreter, Value};
use crate::path::Path;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Parameters<I: Interpreter>(pub HashMap<Path, Value<I>>);

// Needed so Backend doesn't have to implement Default
impl<I: Interpreter> Default for Parameters<I> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<I: Interpreter> From<HashMap<Path, Value<I>>> for Parameters<I> {
    fn from(map: HashMap<Path, Value<I>>) -> Self {
        Parameters(map)
    }
}

impl<const N: usize, I: Interpreter> From<[(Path, Value<I>); N]> for Parameters<I> {
    fn from(arr: [(Path, Value<I>); N]) -> Self {
        Parameters(HashMap::from(arr))
    }
}

impl<'a, I: Interpreter> IntoIterator for &'a Parameters<I> {
    type Item = &'a Path;
    type IntoIter = std::collections::hash_map::Keys<'a, Path, Value<I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.keys()
    }
}

impl<I: Interpreter> Parameters<I> {
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, Path, Value<I>> {
        self.0.keys()
    }
}
