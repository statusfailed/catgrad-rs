use crate::category::core::Shape;
use std::fmt::Debug;

pub trait DType: Copy + Send + Sync + Debug + PartialEq + std::ops::Add<Output = Self> {}
impl DType for f32 {}
impl DType for u32 {}

pub trait Backend: Send + Sync + Clone + Debug {
    // HKT-esque via associated generic types.
    type NdArray<D: DType>: NdArray<D, Backend = Self>;

    fn scalar<D: DType>(d: D) -> Self::NdArray<D>;
    fn zeros<D: DType + Default>(&self, shape: Shape) -> Self::NdArray<D>;
    fn ndarray_from_slice<D: DType>(&self, data: &[D], shape: Shape) -> Self::NdArray<D>;
}

pub trait NdArray<D: DType>: Send + Sync + Clone + Debug + PartialEq {
    type Backend: Backend;
    fn shape(&self) -> Shape;
    fn add(self, rhs: Self) -> Self;
}

// The ONLY generic parameter is here.
/*
pub struct Runtime<B: Backend> {
    pub be: B,
}
*/

////////////////////////////////////////////////////////////////////////////////
// ndarray backend implementation

use ndarray::{ArrayD, IxDyn};

#[derive(Clone, Debug, PartialEq)]
pub struct NdArrayBackend;

impl Backend for NdArrayBackend {
    type NdArray<D: DType> = ArrayD<D>;

    fn scalar<D: DType>(d: D) -> Self::NdArray<D> {
        ArrayD::from_elem(IxDyn(&[]), d)
    }

    fn zeros<D: DType + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        let dims: Vec<usize> = shape.0;
        ArrayD::from_elem(IxDyn(&dims), D::default())
    }

    fn ndarray_from_slice<D: DType>(&self, data: &[D], shape: Shape) -> Self::NdArray<D> {
        let dims: Vec<usize> = shape.0;
        ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).unwrap()
    }
}

impl<D: DType> NdArray<D> for ArrayD<D> {
    type Backend = NdArrayBackend;

    fn shape(&self) -> Shape {
        Shape(self.shape().to_vec())
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
