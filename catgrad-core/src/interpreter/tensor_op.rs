//! Tensor operation implementations for the interpreter

use super::{ApplyError, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::core::TensorOp;
use crate::ssa::SSA;

/// Apply a Tensor operation
pub(crate) fn apply_tensor_op(
    tensor_op: &TensorOp,
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    match tensor_op {
        TensorOp::Map(_scalar_op) => todo!(),
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
