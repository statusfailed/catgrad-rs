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

    fn matmul_f32(lhs: Self::NdArray<f32>, rhs: Self::NdArray<f32>) -> Self::NdArray<f32>;
    fn matmul_u32(lhs: Self::NdArray<u32>, rhs: Self::NdArray<u32>) -> Self::NdArray<u32>;
}

pub trait NdArray<D: DType>: Send + Sync + Clone + Debug + PartialEq {
    type Backend: Backend;
    fn shape(&self) -> Shape;
    fn add(self, rhs: Self) -> Self;
}

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

    fn matmul_f32(lhs: Self::NdArray<f32>, rhs: Self::NdArray<f32>) -> Self::NdArray<f32> {
        // For now, only handle rank 2 case
        assert_eq!(lhs.ndim(), 2, "matmul: self must be rank 2");
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be rank 2");

        // Convert ArrayD to Array2 for dot product
        // NOTE: ndarray needs to know we have 2d arrays statically to use .dot():
        // https://stackoverflow.com/questions/79035190/
        let self_2d = lhs.into_dimensionality::<ndarray::Ix2>().unwrap();
        let rhs_2d = rhs.into_dimensionality::<ndarray::Ix2>().unwrap();

        // Perform matrix multiplication and convert back to ArrayD
        self_2d.dot(&rhs_2d).into_dyn()
    }

    fn matmul_u32(lhs: Self::NdArray<u32>, rhs: Self::NdArray<u32>) -> Self::NdArray<u32> {
        // For now, only handle rank 2 case
        assert_eq!(lhs.ndim(), 2, "matmul: self must be rank 2");
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be rank 2");

        // Convert ArrayD to Array2 for dot product
        // NOTE: ndarray needs to know we have 2d arrays statically to use .dot():
        // https://stackoverflow.com/questions/79035190/
        let self_2d = lhs.into_dimensionality::<ndarray::Ix2>().unwrap();
        let rhs_2d = rhs.into_dimensionality::<ndarray::Ix2>().unwrap();

        // Perform matrix multiplication and convert back to ArrayD
        self_2d.dot(&rhs_2d).into_dyn()
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
