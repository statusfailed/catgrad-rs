//! Catgrad reference interpreter

use crate::category::core::{Dtype, Object, Operation};
use crate::ssa::SSA;

use super::backend::*;
use crate::category::core::{NdArrayType, Shape};

use crate::definition::Def;
use crate::path::Path;

pub(crate) type CoreSSA = SSA<Object, Def<Path, Operation>>;

#[derive(Debug, Clone)]
pub struct ApplyError {
    pub kind: ApplyErrorKind,
    pub ssa: CoreSSA,
}

#[derive(Debug, Clone)]
pub enum ApplyErrorKind {
    TypeError,
    ArityError,
    NatOverflow,             // Nat was not representable as a u32
    MissingOperation(Path),  // Operation declaration not found in ops
    MissingDefinition(Path), // Operation definition not found in env
    BackendError(BackendError),
}

impl From<BackendError> for ApplyErrorKind {
    fn from(error: BackendError) -> Self {
        ApplyErrorKind::BackendError(error)
    }
}

// Actual values produced by the interpreter #[derive(Debug, Clone)]
#[derive(Debug, Clone)]
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
pub trait HasDtype: Copy + Send + Sync + std::fmt::Debug {}
impl HasDtype for f32 {}
impl HasDtype for u32 {}

/// A collection of N NdArrays of the same dtype
#[derive(Copy, Clone, Debug)]
pub enum TaggedNdArrayTuple<B: Backend, const N: usize> {
    F32([B::NdArray<f32>; N]),
    U32([B::NdArray<u32>; N]),
}

////////////////////////////////////////////////////////////////////////////////

pub trait IntoTagged<B: Backend, const N: usize>: Clone + std::fmt::Debug + HasDtype {
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

    pub fn dtype(&self) -> Dtype {
        match self {
            Self::F32(_) => Dtype::F32,
            Self::U32(_) => Dtype::U32,
        }
    }

    pub fn scalar<T: IntoTagged<B, 1>>(backend: &B, x: T) -> Self {
        let arr: B::NdArray<T> = backend.scalar(x);
        T::into_tagged([arr])
    }

    pub fn from_slice<T: IntoTagged<B, 1>>(
        backend: &B,
        data: &[T],
        shape: Shape,
    ) -> Result<Self, BackendError> {
        let arr = backend.ndarray_from_slice(data, shape)?;
        Ok(T::into_tagged([arr]))
    }
}
