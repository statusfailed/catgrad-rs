//! Catgrad reference interpreter

use super::backend::*;
use super::interpreter::Interpreter;
use crate::abstract_interpreter;
use crate::category::core::{Dtype, Shape};

pub type Value<B> = abstract_interpreter::Value<Interpreter<B>>;
pub type ResultValues<B> = abstract_interpreter::ResultValues<Interpreter<B>>;
pub type Parameters<B> = abstract_interpreter::parameters::Parameters<Interpreter<B>>;

////////////////////////////////////////////////////////////////////////////////
// Multiple tagged ndarrays

#[derive(Clone, Debug)]
pub enum TaggedVec {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

/// A collection of n tensors of the same dtype
#[derive(Copy, Clone, Debug)]
pub enum TaggedTensorTuple<B: Backend, const N: usize> {
    F32([B::BackendTensor; N]),
    U32([B::BackendTensor; N]),
}

////////////////////////////////////////////////////////////////////////////////

pub trait IntoTagged<B: Backend, const N: usize>:
    Clone + std::fmt::Debug + Copy + Sync + Send
{
    fn into_tagged(arr: [B::BackendTensor; N]) -> TaggedTensorTuple<B, N>;

    fn ndarray_from_vec(
        backend: &B,
        data: Vec<Self>,
        shape: Shape,
    ) -> Result<TaggedTensor<B>, BackendError>;
}

impl<B: Backend, const N: usize> IntoTagged<B, N> for f32 {
    fn into_tagged(arrs: [B::BackendTensor; N]) -> TaggedTensorTuple<B, N> {
        TaggedTensorTuple::F32(arrs)
    }

    fn ndarray_from_vec(
        backend: &B,
        data: Vec<Self>,
        shape: Shape,
    ) -> Result<TaggedTensor<B>, BackendError> {
        backend.ndarray_from_vec_f32(data, shape)
    }
}

impl<B: Backend, const N: usize> IntoTagged<B, N> for u32 {
    fn into_tagged(arrs: [B::BackendTensor; N]) -> TaggedTensorTuple<B, N> {
        TaggedTensorTuple::U32(arrs)
    }

    fn ndarray_from_vec(
        backend: &B,
        data: Vec<Self>,
        shape: Shape,
    ) -> Result<TaggedTensor<B>, BackendError> {
        backend.ndarray_from_vec_u32(data, shape)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Single tagged array
// TODO: this can easily generalise to N; is that necessary?

pub type TaggedTensor<B> = TaggedTensorTuple<B, 1>;

impl<B: Backend> TaggedTensor<B> {
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

    pub fn from_vec<T: IntoTagged<B, 1>>(
        backend: &B,
        data: Vec<T>,
        shape: Shape,
    ) -> Result<Self, BackendError> {
        T::ndarray_from_vec(backend, data, shape)
    }
}
