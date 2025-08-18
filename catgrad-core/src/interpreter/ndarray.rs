//use crate::category::core::{Constant, Dtype, ScalarOp, Shape};
use crate::category::core::Shape;

#[derive(PartialEq, Debug, Clone)]
pub struct NdArray {
    pub buf: Vec<u8>,
    pub shape: Shape,
    pub strides: Vec<isize>,
    pub offset: usize,
}

#[derive(PartialEq, Debug, Clone)]
pub enum NdArrayError {
    TypeError,
    ShapeMismatch,
}

// memory manager.
pub struct NdArrayManager;

/*
impl NdArray {
    /// Create scalar from constant
    pub fn from_constant(_constant: Constant) -> Self {
        todo!()
    }

    /// Apply scalar operation element-wise (m → n)
    pub fn map(
        inputs: Vec<&Self>,
        op: ScalarOp,
        arity: usize,
        coarity: usize,
    ) -> Result<Vec<Self>, NdArrayError> {
        todo!()
    }

    /// Reduce along a dimension
    pub fn reduce(&self, _op: ScalarOp, _axis: i8) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Matrix multiplication
    pub fn matmul(&self, _other: &Self) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Reshape
    pub fn reshape(&self, _new_shape: Shape) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Stack multiple arrays along a new dimension
    pub fn stack(_arrays: Vec<&Self>) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Split array along the first dimension
    pub fn split(&self) -> Result<Vec<Self>, NdArrayError> {
        todo!()
    }

    /// Index into array using another array as indices
    pub fn index(&self, _indices: &Self) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Broadcast array by appending a shape
    pub fn broadcast(&self, _target_shape: Shape) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Copy array
    pub fn copy(&self) -> Self {
        todo!()
    }
}
*/
