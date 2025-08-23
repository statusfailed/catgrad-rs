//! Tensor operation implementations for the interpreter

use super::backend::*;
use super::{ApplyError, ApplyErrorKind, TaggedNdArray, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::core::{ScalarOp, TensorOp};
use crate::ssa::SSA;

/// Apply a Tensor operation
pub(crate) fn apply_tensor_op<B: Backend>(
    tensor_op: &TensorOp,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match tensor_op {
        TensorOp::Map(ScalarOp::Add) => binop(args, ssa, |x, y| x.add(y), |x, y| x.add(y)),
        TensorOp::Map(scalar_op) => todo!("unimplemented scalar op {:?}", scalar_op),
        TensorOp::Reduce(_scalar_op, _axis) => todo!("implement tensor reduce"),
        TensorOp::Constant(_constant) => todo!("implement tensor constant"),
        TensorOp::Stack => todo!("implement tensor stack"),
        TensorOp::Split => todo!("implement tensor split"),
        TensorOp::Reshape => todo!("implement tensor reshape"),
        TensorOp::MatMul => binop(
            args,
            ssa,
            <B as Backend>::matmul_f32,
            <B as Backend>::matmul_u32,
        ),
        TensorOp::Index => todo!("implement tensor index"),
        TensorOp::Broadcast => todo!("implement tensor broadcast"),
        TensorOp::Copy => todo!("implement tensor copy"),
    }
}

// helper for handling binop cases
fn binop<B: Backend>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    f32_op: impl Fn(B::NdArray<f32>, B::NdArray<f32>) -> B::NdArray<f32>,
    u32_op: impl Fn(B::NdArray<u32>, B::NdArray<u32>) -> B::NdArray<u32>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    }

    // TODO: get rid of clones!
    let (x, y) = match (args[0].clone(), args[1].clone()) {
        (Value::NdArray(x), Value::NdArray(y)) => (x, y),
        _ => {
            return Err(Box::new(ApplyError {
                kind: ApplyErrorKind::TypeError,
                ssa: ssa.clone(),
            }));
        }
    };

    let result = match (x, y) {
        (TaggedNdArray::F32(x_arr), TaggedNdArray::F32(y_arr)) => {
            TaggedNdArray::F32(f32_op(x_arr, y_arr))
        }
        (TaggedNdArray::U32(x_arr), TaggedNdArray::U32(y_arr)) => {
            TaggedNdArray::U32(u32_op(x_arr, y_arr))
        }
        _ => {
            return Err(Box::new(ApplyError {
                kind: ApplyErrorKind::TypeError,
                ssa: ssa.clone(),
            }));
        }
    };

    Ok(vec![Value::NdArray(result)])
}
