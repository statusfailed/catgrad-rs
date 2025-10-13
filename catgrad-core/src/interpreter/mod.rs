pub mod types;
pub use types::*;

pub mod interpreter;
pub use interpreter::*;

pub mod backend;
pub mod parameters;
pub mod tensor_op;

pub use crate::category::core::Shape;
pub use backend::{Backend, BackendError};
pub use parameters::Parameters;

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
    Ok(Value::Tensor(tagged))
}
