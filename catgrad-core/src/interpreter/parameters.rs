use super::{Backend, Interpreter, TaggedTensor, Value};
use crate::abstract_interpreter::parameters;
use crate::path::Path;
use std::collections::HashMap;

/// Dictionary of Value parameters, keyed by Path, parametrised by backend B.
pub type Parameters<B> = parameters::Parameters<Interpreter<B>>;

// Convenience helper for converting from a hashmap of *tensors* instead of general values
impl<B: Backend> From<HashMap<Path, TaggedTensor<B>>> for Parameters<B> {
    fn from(x: HashMap<Path, TaggedTensor<B>>) -> Self {
        let x = x.into_iter().map(|(k, v)| (k, Value::Tensor(v))).collect();
        parameters::Parameters(x)
    }
}
