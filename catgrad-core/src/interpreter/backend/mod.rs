use super::types::*;
use crate::category::core::Shape;
use std::fmt::Debug;

#[cfg(feature = "ndarray-backend")]
pub mod ndarray;

pub trait Backend: Send + Sync + Clone + Debug {
    // HKT-esque via associated generic types.
    type NdArray<D: HasDtype>: NdArray<D, Backend = Self>;

    fn scalar<D: HasDtype>(d: D) -> Self::NdArray<D>;
    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D>;
    fn ndarray_from_slice<D: HasDtype>(&self, data: &[D], shape: Shape) -> Self::NdArray<D>;

    fn matmul_f32(lhs: Self::NdArray<f32>, rhs: Self::NdArray<f32>) -> Self::NdArray<f32>;
    fn matmul_u32(lhs: Self::NdArray<u32>, rhs: Self::NdArray<u32>) -> Self::NdArray<u32>;

    fn add_f32(x: Self::NdArray<f32>, y: Self::NdArray<f32>) -> Self::NdArray<f32>;
    fn add_u32(x: Self::NdArray<u32>, y: Self::NdArray<u32>) -> Self::NdArray<u32>;
}

pub trait NdArray<D: HasDtype>: Send + Sync + Clone + Debug + PartialEq {
    type Backend: Backend;
    fn shape(&self) -> Shape;
}
