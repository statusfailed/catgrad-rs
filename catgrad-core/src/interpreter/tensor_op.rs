//! Tensor operation implementations for the interpreter

use super::{ApplyError, ApplyErrorKind, NdArray, TaggedNdArray, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::core::{ScalarOp, TensorOp};
use crate::ssa::SSA;

/// Apply a Tensor operation
pub(crate) fn apply_tensor_op(
    tensor_op: &TensorOp,
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    match tensor_op {
        TensorOp::Map(ScalarOp::Add) => binop(args, ssa, |x, y| x + y, |x, y| x + y),
        TensorOp::Map(ScalarOp::Mul) => binop(args, ssa, |x, y| x * y, |x, y| x * y),
        TensorOp::Map(ScalarOp::Div) => binop(args, ssa, |x, y| x / y, |x, y| x / y),
        TensorOp::Map(scalar_op) => todo!("unimplemented scalar op {:?}", scalar_op),
        TensorOp::Reduce(_scalar_op, _axis) => todo!("implement tensor reduce"),
        TensorOp::Constant(_constant) => todo!("implement tensor constant"),
        TensorOp::Stack => todo!("implement tensor stack"),
        TensorOp::Split => todo!("implement tensor split"),
        TensorOp::Reshape => todo!("implement tensor reshape"),
        TensorOp::MatMul => todo!("implement tensor matmul"),
        TensorOp::Index => todo!("implement tensor index"),
        TensorOp::Broadcast => todo!("implement tensor broadcast"),
        TensorOp::Copy => todo!("implement tensor copy"),
    }
}

// helper for handling binop cases
fn binop(
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
    f32_op: impl Fn(&NdArray<f32>, &NdArray<f32>) -> NdArray<f32>,
    u32_op: impl Fn(&NdArray<u32>, &NdArray<u32>) -> NdArray<u32>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
            args,
        }));
    }

    let (x, y) = match (&args[0], &args[1]) {
        (Value::NdArray(x), Value::NdArray(y)) => (x, y),
        _ => {
            return Err(Box::new(ApplyError {
                kind: ApplyErrorKind::TypeError,
                ssa: ssa.clone(),
                args,
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
                args,
            }));
        }
    };

    Ok(vec![Value::NdArray(result)])
}
