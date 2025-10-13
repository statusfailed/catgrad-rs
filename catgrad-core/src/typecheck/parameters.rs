use crate::path::Path;
use std::collections::HashMap;

type Type = super::interpreter::Value;

#[derive(PartialEq, Clone, Debug, Default)]
pub struct Parameters(pub HashMap<Path, Type>);

impl From<HashMap<Path, Type>> for Parameters {
    fn from(map: HashMap<Path, Type>) -> Self {
        Parameters(map)
    }
}

impl<const N: usize> From<[(Path, Type); N]> for Parameters {
    fn from(arr: [(Path, Type); N]) -> Self {
        Parameters(HashMap::from(arr))
    }
}

impl<'a> IntoIterator for &'a Parameters {
    type Item = &'a Path;
    type IntoIter = std::collections::hash_map::Keys<'a, Path, Type>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.keys()
    }
}

impl Parameters {
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, Path, Type> {
        self.0.keys()
    }
}
