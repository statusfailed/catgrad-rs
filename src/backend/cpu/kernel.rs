//! Array kernels for CPU
use super::ndarray::*;
use core::fmt::Debug;
//use matrixmultiply::sgemm;

pub trait MatMul: Sized {
    fn matmul(a: &NdArray<Self>, b: &NdArray<Self>, c: &mut NdArray<Self>);
}

impl MatMul for i32 {
    fn matmul(_a: &NdArray<Self>, _b: &NdArray<Self>, _c: &mut NdArray<Self>) {
        todo!("unsupported") // don't implement yet.
    }
}

impl MatMul for f32 {
    fn matmul(_a: &NdArray<Self>, _b: &NdArray<Self>, _c: &mut NdArray<Self>) {
        todo!("implement using sgemm")
    }
}

/// Batch matrix multiply (compose) `f : N×A×B` with `g : N×B×C`, writing the result to `h :
/// N×A×C`.
pub fn batch_matmul<T: MatMul + Debug>(_f: &NdArray<T>, _g: &NdArray<T>, _h: &mut NdArray<T>) {
    // TODO: assert that shapes of f, g, h are compatible with batched matmul operation
    // TODO: loop over batch dimension calling matmul
    //todo!("implement batch_matmul")
}
