//! Tensor operation implementations for the interpreter

use super::backend::*;
use super::{ApplyError, ApplyErrorKind, TaggedNdArray, TaggedNdArrays, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::core::{Dtype, ScalarOp, TensorOp};
use crate::ssa::SSA;

/// Apply a Tensor operation
pub(crate) fn apply_tensor_op<B: Backend>(
    tensor_op: &TensorOp,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match tensor_op {
        TensorOp::Map(ScalarOp::Add) => run_op(args, ssa, B::add),
        TensorOp::Map(scalar_op) => todo!("unimplemented scalar op {:?}", scalar_op),
        TensorOp::Reduce(_scalar_op, _axis) => todo!("implement tensor reduce"),
        TensorOp::Constant(_constant) => todo!("implement tensor constant"),
        TensorOp::Stack => todo!("implement tensor stack"),
        TensorOp::Split => todo!("implement tensor split"),
        TensorOp::Reshape => todo!("implement tensor reshape"),
        TensorOp::MatMul => run_op(args, ssa, B::matmul),
        TensorOp::Index => todo!("implement tensor index"),
        TensorOp::Broadcast => todo!("implement tensor broadcast"),
        TensorOp::Copy => todo!("implement tensor copy"),
    }
}

/// Run an M → 1 op taking M NdArray values of the same dtype, producing an NdArray.
fn run_op<B: Backend, F, const M: usize>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    f: F,
) -> Result<Vec<Value<B>>, Box<ApplyError>>
where
    F: Fn(TaggedNdArrays<B, M>) -> TaggedNdArrays<B, 1>,
{
    Ok(vec![Value::NdArray(f(try_into_tagged_ndarrays::<B, M>(
        args, ssa,
    )?))])
}

/// Convert a Vec<Value<B>> into TaggedNdArrays<B, N> with compile-time length checking
pub(crate) fn try_into_tagged_ndarrays<B: Backend, const N: usize>(
    values: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<TaggedNdArrays<B, N>, Box<ApplyError>> {
    let n = values.len();

    // clippy is WRONG! what if someone changes the type of n, then what, huh!?
    // i will die on this hill
    #[allow(clippy::absurd_extreme_comparisons)]
    if n != N || n <= 0 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let mut f32_arrays = Vec::new();
    let mut u32_arrays = Vec::new();
    let mut dtype_kind: Option<Dtype> = None;

    for value in values {
        match value {
            Value::NdArray(TaggedNdArray::F32(arr)) => {
                if let Some(ref existing) = dtype_kind {
                    if *existing != Dtype::F32 {
                        return Err(Box::new(ApplyError {
                            kind: ApplyErrorKind::TypeError,
                            ssa: ssa.clone(),
                        }));
                    }
                } else {
                    dtype_kind = Some(Dtype::F32);
                }
                f32_arrays.push(arr[0].clone());
            }
            Value::NdArray(TaggedNdArray::U32(arr)) => {
                if let Some(ref existing) = dtype_kind {
                    if *existing != Dtype::U32 {
                        return Err(Box::new(ApplyError {
                            kind: ApplyErrorKind::TypeError,
                            ssa: ssa.clone(),
                        }));
                    }
                } else {
                    dtype_kind = Some(Dtype::U32);
                }
                u32_arrays.push(arr[0].clone());
            }
            _ => {
                return Err(Box::new(ApplyError {
                    kind: ApplyErrorKind::TypeError,
                    ssa: ssa.clone(),
                }));
            }
        }
    }

    let result = match dtype_kind {
        Some(Dtype::F32) => f32_arrays.try_into().ok().map(TaggedNdArrays::F32),
        Some(Dtype::U32) => u32_arrays.try_into().ok().map(TaggedNdArrays::U32),
        _ => None,
    };

    result.ok_or_else(|| {
        Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        })
    })
}
