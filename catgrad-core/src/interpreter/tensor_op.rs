//! Tensor operation implementations for the interpreter

use super::backend::*;
use super::{ApplyError, ApplyErrorKind, TaggedNdArray, TaggedNdArrayTuple, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::core::{Dtype, ScalarOp, TensorOp};
use crate::ssa::SSA;

/// Apply a Tensor operation
pub(crate) fn apply_tensor_op<B: Backend>(
    backend: &B,
    tensor_op: &TensorOp,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match tensor_op {
        TensorOp::MatMul => binop(backend, args, ssa, B::matmul_f32, B::matmul_u32),
        TensorOp::Constant(_constant) => todo!(),
        TensorOp::Sum => todo!(),
        TensorOp::Max => todo!(),
        TensorOp::Argmax => todo!(),
        TensorOp::Broadcast => todo!(),
        TensorOp::Reshape => todo!(),
        TensorOp::Map(ScalarOp::Add) => binop(backend, args, ssa, B::add_f32, B::add_u32),
        TensorOp::Map(scalar_op) => todo!("unimplemented scalar op {:?}", scalar_op),
        TensorOp::Stack => todo!(),
        TensorOp::Split => todo!(),
        TensorOp::Index => todo!(),
        TensorOp::Copy => todo!(),
    }
}

#[allow(type_alias_bounds)]
type Binop<B: Backend, T> = fn(&B, B::NdArray<T>, B::NdArray<T>) -> B::NdArray<T>;

fn binop<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    case_f32: Binop<B, f32>,
    case_u32: Binop<B, u32>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match try_into_tagged_ndarrays::<B, 2>(args, ssa)? {
        TaggedNdArrayTuple::F32([x, y]) => {
            Ok(vec![Value::NdArray(TaggedNdArrayTuple::F32([case_f32(
                backend, x, y,
            )]))])
        }
        TaggedNdArrayTuple::U32([x, y]) => {
            Ok(vec![Value::NdArray(TaggedNdArrayTuple::U32([case_u32(
                backend, x, y,
            )]))])
        }
    }
}

// TODO: binop has some boilerplate- have to unpack/repack a lot. can we fix that using run_op &
// into() on taggedndarraytuple?
#[allow(dead_code)]
/// Run an M → 1 op taking M NdArray values of the same dtype, producing an NdArray.
fn run_op<B: Backend, F, const M: usize>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    f: F,
) -> Result<Vec<Value<B>>, Box<ApplyError>>
where
    F: Fn(TaggedNdArrayTuple<B, M>) -> TaggedNdArrayTuple<B, 1>,
{
    Ok(vec![Value::NdArray(f(try_into_tagged_ndarrays::<B, M>(
        args, ssa,
    )?))])
}

/// Convert a Vec<Value<B>> into TaggedNdArrays<B, N> with compile-time length checking
pub(crate) fn try_into_tagged_ndarrays<B: Backend, const N: usize>(
    values: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<TaggedNdArrayTuple<B, N>, Box<ApplyError>> {
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
        Some(Dtype::F32) => f32_arrays.try_into().ok().map(TaggedNdArrayTuple::F32),
        Some(Dtype::U32) => u32_arrays.try_into().ok().map(TaggedNdArrayTuple::U32),
        _ => None,
    };

    result.ok_or_else(|| {
        Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        })
    })
}
