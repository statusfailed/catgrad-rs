use crate::core::object::*;

use num_traits::Zero;

/// N-dimensional arrays of elements T with a given shape.
#[derive(PartialEq, Debug, Clone)]
pub struct NdArray<T> {
    pub data: Vec<T>, // raw data of the array
    pub shape: Shape, // shape information (erasable?)
}

impl<T> NdArray<T> {
    pub fn is_empty(&self) -> bool {
        self.shape.size() == 0
    }

    pub fn len(&self) -> usize {
        self.shape.size()
    }
}

impl<T: Clone + Zero> NdArray<T> {
    pub fn from_shape(shape: Shape) -> Self {
        // TODO: don't really need to initialize to zero; is there a better way here? bytemuck?
        NdArray {
            data: vec![T::zero(); shape.size()],
            shape,
        }
    }
}

/// A disjoint union of typed arrays
/// intuition: the "values" assigned to each node in a [`Term`].
#[derive(PartialEq, Debug, Clone)]
pub enum TaggedNdArray {
    F32(NdArray<f32>),
    I32(NdArray<i32>),
}

impl TaggedNdArray {
    pub fn is_empty(&self) -> bool {
        match self {
            TaggedNdArray::F32(vec) => vec.is_empty(),
            TaggedNdArray::I32(vec) => vec.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TaggedNdArray::F32(vec) => vec.len(),
            TaggedNdArray::I32(vec) => vec.len(),
        }
    }

    pub fn from_type(t: &NdArrayType) -> Self {
        match t.dtype {
            Dtype::F32 => TaggedNdArray::F32(NdArray::from_shape(t.shape.clone())),
            Dtype::I32 => TaggedNdArray::I32(NdArray::from_shape(t.shape.clone())),
        }
    }
}

impl From<NdArray<f32>> for TaggedNdArray {
    fn from(value: NdArray<f32>) -> Self {
        TaggedNdArray::F32(value)
    }
}

impl From<NdArray<i32>> for TaggedNdArray {
    fn from(value: NdArray<i32>) -> Self {
        TaggedNdArray::I32(value)
    }
}
