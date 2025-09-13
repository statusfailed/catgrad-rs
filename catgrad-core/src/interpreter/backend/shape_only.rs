/// A backend which only does shape computations, not tensor ones.
use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError};

#[derive(Clone, Debug)]
pub struct ShapeOnlyBackend;

#[derive(Clone, Debug)]
pub struct ShapeOnly(Shape);

impl<D: HasDtype> crate::interpreter::backend::NdArray<D> for ShapeOnly {
    type Backend = ShapeOnlyBackend;

    fn shape(&self) -> Shape {
        self.0.clone()
    }
}

impl Backend for ShapeOnlyBackend {
    type NdArray<D: HasDtype> = ShapeOnly;

    fn scalar<D: HasDtype>(&self, _d: D) -> Self::NdArray<D> {
        ShapeOnly(Shape(vec![]))
    }

    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        ShapeOnly(shape)
    }

    fn ndarray_from_slice<D: HasDtype>(
        &self,
        _data: &[D],
        shape: Shape,
    ) -> Result<Self::NdArray<D>, BackendError> {
        Ok(ShapeOnly(shape))
    }

    fn cast(&self, x: TaggedNdArray<Self>, _target_dtype: Dtype) -> TaggedNdArray<Self> {
        x
    }

    fn matmul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::matmul_shape(x, y)]),
            U32([x, y]) => U32([Self::matmul_shape(x, y)]),
        }
    }

    fn add(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn sub(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn mul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn div(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn pow(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn neg(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        x
    }

    fn broadcast(&self, x: TaggedNdArray<Self>, shape_prefix: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::broadcast_with_prefix(arr, shape_prefix)]),
            U32([arr]) => U32([Self::broadcast_with_prefix(arr, shape_prefix)]),
        }
    }

    fn reshape(&self, _x: TaggedNdArray<Self>, new_shape: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        let arr = ShapeOnly(new_shape);
        match _x {
            F32(_) => F32([arr]),
            U32(_) => U32([arr]),
        }
    }

    fn max(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn sum(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn compare(&self, x: TaggedNdArrayTuple<Self, 2>) -> bool {
        use TaggedNdArrayTuple::*;
        match x {
            F32([a, b]) => a.0 == b.0,
            U32([a, b]) => a.0 == b.0,
        }
    }
}

impl ShapeOnlyBackend {
    fn matmul_shape(lhs: ShapeOnly, rhs: ShapeOnly) -> ShapeOnly {
        let lhs_shape = &lhs.0.0;
        let rhs_shape = &rhs.0.0;

        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            panic!("matmul: both operands must have at least 2 dimensions");
        }

        let lhs_batch = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch = &rhs_shape[..rhs_shape.len() - 2];
        let lhs_m = lhs_shape[lhs_shape.len() - 2];
        let lhs_k = lhs_shape[lhs_shape.len() - 1];
        let rhs_k = rhs_shape[rhs_shape.len() - 2];
        let rhs_n = rhs_shape[rhs_shape.len() - 1];

        if lhs_k != rhs_k {
            panic!("matmul: inner dimensions must match");
        }

        if lhs_batch != rhs_batch {
            panic!("matmul: batch dimensions must match");
        }

        let mut result_shape = lhs_batch.to_vec();
        result_shape.push(lhs_m);
        result_shape.push(rhs_n);

        ShapeOnly(Shape(result_shape))
    }

    fn exact_shape_match(x: ShapeOnly, y: ShapeOnly) -> ShapeOnly {
        if x.0 != y.0 {
            panic!("Shape mismatch: {:?} vs {:?}", x.0, y.0);
        }
        x
    }

    fn broadcast_with_prefix(arr: ShapeOnly, shape_prefix: Shape) -> ShapeOnly {
        let mut result_shape = shape_prefix.0;
        result_shape.extend_from_slice(&arr.0.0);
        ShapeOnly(Shape(result_shape))
    }

    fn reduce_last_dim(arr: ShapeOnly) -> ShapeOnly {
        let shape = &arr.0.0;
        if shape.is_empty() {
            return arr;
        }

        let mut result_shape = shape[..shape.len() - 1].to_vec();
        if result_shape.is_empty() {
            result_shape.push(1);
        }

        ShapeOnly(Shape(result_shape))
    }
}
