use super::types::*;
use crate::category::core::Shape;
use std::fmt::Debug;

#[cfg(feature = "ndarray-backend")]
pub mod ndarray;

pub trait Backend: Send + Sync + Clone + Debug {
    // HKT-esque via associated generic types.
    type NdArray<D: DType>: NdArray<D, Backend = Self>;

    fn scalar<D: DType>(d: D) -> Self::NdArray<D>;
    fn zeros<D: DType + Default>(&self, shape: Shape) -> Self::NdArray<D>;
    fn ndarray_from_slice<D: DType>(&self, data: &[D], shape: Shape) -> Self::NdArray<D>;

    // A matmul for every DType
    fn matmul(args: TaggedNdArrays<Self, 2>) -> TaggedNdArrays<Self, 1>;
    fn add(args: TaggedNdArrays<Self, 2>) -> TaggedNdArrays<Self, 1>;
}

pub trait NdArray<D: DType>: Send + Sync + Clone + Debug + PartialEq {
    type Backend: Backend;
    fn shape(&self) -> Shape;
}
