//! Catgrad reference interpreter

use super::backend::*;
use super::interpreter::Interpreter;
use crate::abstract_interpreter;
use crate::category::core::{Dtype, Shape};

pub type Value<B> = abstract_interpreter::Value<Interpreter<B>>;
pub type ResultValues<B> = abstract_interpreter::EvalResultValues<Interpreter<B>>;

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
