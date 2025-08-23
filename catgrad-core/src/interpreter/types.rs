//! Catgrad reference interpreter

use crate::category::core::{NdArrayType, Shape};
use crate::ssa::SSA;

use crate::category::bidirectional::*;

//use super::ndarray::TaggedNdArray;
use super::backend::*;

#[derive(PartialEq, Debug, Clone)]
pub struct ApplyError {
    pub kind: ApplyErrorKind,
    pub ssa: SSA<Object, Operation>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum ApplyErrorKind {
    TypeError,
    MissingOperation(Path),  // Operation declaration not found in ops
    MissingDefinition(Path), // Operation definition not found in env
}

// Actual values produced by the interpreter
#[derive(PartialEq, Debug, Clone)]
pub enum Value<B: Backend> {
    /// A concrete natural number
    Nat(usize),

    /// A concrete dtype
    Dtype(Dtype),

    /// A concrete shape (list of natural numbers)
    Shape(Shape),

    /// A concrete NdArrayType (dtype + shape)
    Type(NdArrayType),

    /// A tensor with actual data
    NdArray(TaggedNdArray<B>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum TaggedNdArray<B: Backend> {
    F32(<B as Backend>::NdArray<f32>),
    U32(<B as Backend>::NdArray<u32>),
}

impl<B: Backend> TaggedNdArray<B> {
    pub fn shape(&self) -> Shape {
        match self {
            Self::F32(x) => x.shape(),
            Self::U32(x) => x.shape(),
        }
    }

    pub fn scalar<T: IntoTagged<B>>(x: T) -> Self {
        let arr: B::NdArray<T> = <B as Backend>::scalar(x);
        T::into_tagged(arr)
    }

    pub fn from_slice<T: IntoTagged<B>>(backend: &B, data: &[T], shape: Shape) -> Self {
        let arr: B::NdArray<T> = backend.ndarray_from_slice(data, shape);
        T::into_tagged(arr)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reduce some boilerplate for TaggedNdArray

pub trait IntoTagged<B: Backend>: Clone + PartialEq + std::fmt::Debug + DType {
    fn into_tagged(arr: B::NdArray<Self>) -> TaggedNdArray<B>;
}

impl<B: Backend> IntoTagged<B> for f32 {
    fn into_tagged(arr: B::NdArray<Self>) -> TaggedNdArray<B> {
        TaggedNdArray::F32(arr)
    }
}

impl<B: Backend> IntoTagged<B> for u32 {
    fn into_tagged(arr: B::NdArray<Self>) -> TaggedNdArray<B> {
        TaggedNdArray::U32(arr)
    }
}
