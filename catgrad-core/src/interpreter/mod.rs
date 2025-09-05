pub mod types;
pub use types::*;

pub mod run;
pub use run::*;

pub mod backend;

pub mod shape_op;
pub mod tensor_op;

pub use crate::category::core::Shape;
pub use backend::{Backend, BackendError};

#[cfg(all(test, feature = "ndarray-backend"))]
mod tests;

// Create a tensor
pub fn tensor<B: Backend, T: IntoTagged<B, 1>>(
    backend: &B,
    shape: Shape,
    data: &[T],
) -> Result<Value<B>, BackendError> {
    if shape.size() != data.len() {
        return Err(BackendError::ShapeError);
    }
    let tagged = TaggedNdArray::from_slice(backend, data, shape)?;
    Ok(Value::NdArray(tagged))
}
