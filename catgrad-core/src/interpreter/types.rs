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
    ArityError,
    MissingOperation(Path),  // Operation declaration not found in ops
    MissingDefinition(Path), // Operation definition not found in env
}

// Actual values produced by the interpreter #[derive(PartialEq, Debug, Clone)]
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

////////////////////////////////////////////////////////////////////////////////
// Multiple tagged ndarrays

// TODO: make this sealed
pub trait HasDtype: Copy + Send + Sync + std::fmt::Debug + PartialEq {}
impl HasDtype for f32 {}
impl HasDtype for u32 {}

/// A collection of N NdArrays of the same dtype
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TaggedNdArrayTuple<B: Backend, const N: usize> {
    F32([B::NdArray<f32>; N]),
    U32([B::NdArray<u32>; N]),
}

////////////////////////////////////////////////////////////////////////////////

pub trait IntoTagged<B: Backend, const N: usize>:
    Clone + PartialEq + std::fmt::Debug + HasDtype
{
    fn into_tagged(arr: [B::NdArray<Self>; N]) -> TaggedNdArrayTuple<B, N>;
}

impl<B: Backend, const N: usize> IntoTagged<B, N> for f32 {
    fn into_tagged(arrs: [B::NdArray<Self>; N]) -> TaggedNdArrayTuple<B, N> {
        TaggedNdArrayTuple::F32(arrs)
    }
}

impl<B: Backend, const N: usize> IntoTagged<B, N> for u32 {
    fn into_tagged(arrs: [B::NdArray<Self>; N]) -> TaggedNdArrayTuple<B, N> {
        TaggedNdArrayTuple::U32(arrs)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Single tagged array
// TODO: this can easily generalise to N; is that necessary?

pub type TaggedNdArray<B> = TaggedNdArrayTuple<B, 1>;

impl<B: Backend> TaggedNdArray<B> {
    pub fn shape(&self) -> Shape {
        match self {
            Self::F32(x) => x[0].shape(),
            Self::U32(x) => x[0].shape(),
        }
    }

    pub fn scalar<T: IntoTagged<B, 1>>(backend: &B, x: T) -> Self {
        let arr: B::NdArray<T> = backend.scalar(x);
        T::into_tagged([arr])
    }

    pub fn from_slice<T: IntoTagged<B, 1>>(backend: &B, data: &[T], shape: Shape) -> Self {
        let arr: B::NdArray<T> = backend.ndarray_from_slice(data, shape);
        T::into_tagged([arr])
    }
}
