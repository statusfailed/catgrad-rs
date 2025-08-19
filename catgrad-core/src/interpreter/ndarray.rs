use crate::category::core::{Constant, ScalarOp, Shape};

#[derive(PartialEq, Debug, Clone)]
pub enum TaggedArray {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct NdArray {
    pub buf: TaggedArray,
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

impl NdArray {
    /// Create scalar from constant
    pub fn from_constant(_constant: Constant) -> Self {
        todo!()
    }

    /// Check if array is stored contiguously in memory
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let mut expected_stride = 1_isize;

        // Check strides from right to left (last dimension should have stride 1)
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }

        true
    }

    /// Apply scalar operation element-wise (m → n)
    pub fn map(mut _inputs: Vec<Self>, _op: &ScalarOp) -> Result<Vec<Self>, NdArrayError> {
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

    // Split array along the first dimension (no copies; slices original)
    pub fn split(&self) -> Result<Vec<Self>, NdArrayError> {
        todo!()
    }

    // Index into array using another array as indices
    // Result buffer is distinct from input.
    pub fn index(&self, _indices: &Self) -> Result<Self, NdArrayError> {
        todo!()
    }

    /// Broadcast array by appending a shape.
    /// Does not change underlying buffer, just strides/offset.
    pub fn broadcast(self, _shape_extension: Shape) -> Result<Self, NdArrayError> {
        todo!("port Shape, NdArrayType, etc. from catgrad main")
    }
}

impl TaggedArray {}
