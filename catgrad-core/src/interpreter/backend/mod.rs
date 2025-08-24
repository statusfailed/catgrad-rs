use crate::category::core::Shape;
use std::fmt::Debug;

#[cfg(feature = "ndarray-backend")]
pub mod ndarray;

pub trait DType: Copy + Send + Sync + Debug + PartialEq + std::ops::Add<Output = Self> {}
impl DType for f32 {}
impl DType for u32 {}

pub trait Backend: Send + Sync + Clone + Debug {
    // HKT-esque via associated generic types.
    type NdArray<D: DType>: NdArray<D, Backend = Self>;

    fn scalar<D: DType>(d: D) -> Self::NdArray<D>;
    fn zeros<D: DType + Default>(&self, shape: Shape) -> Self::NdArray<D>;
    fn ndarray_from_slice<D: DType>(&self, data: &[D], shape: Shape) -> Self::NdArray<D>;

    fn matmul_f32(lhs: Self::NdArray<f32>, rhs: Self::NdArray<f32>) -> Self::NdArray<f32>;
    fn matmul_u32(lhs: Self::NdArray<u32>, rhs: Self::NdArray<u32>) -> Self::NdArray<u32>;
}

pub trait NdArray<D: DType>: Send + Sync + Clone + Debug + PartialEq {
    type Backend: Backend;
    fn shape(&self) -> Shape;
    fn add(self, rhs: Self) -> Self;
}
