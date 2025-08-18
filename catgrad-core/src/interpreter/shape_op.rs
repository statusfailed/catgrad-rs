//! Shape operation implementations for the interpreter

use super::{ApplyError, ApplyErrorKind, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::{core, shape};
use crate::ssa::SSA;

/// Apply a Type operation
pub fn apply_type_op(
    type_op: &shape::TypeOp,
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    match type_op {
        shape::TypeOp::Pack => apply_pack(args, ssa),
        shape::TypeOp::Unpack => apply_unpack(args, ssa),
        shape::TypeOp::Shape => apply_shape(args, ssa),
    }
}

/// Apply a Nat operation
pub fn apply_nat_op(
    nat_op: &shape::NatOp,
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    match nat_op {
        shape::NatOp::Constant(n) => apply_nat_constant(*n, args, ssa),
        shape::NatOp::Mul => apply_nat_mul(args, ssa),
        shape::NatOp::Add => apply_nat_add(args, ssa),
    }
}

/// Apply a Dtype constant operation
pub fn apply_dtype_constant(
    dtype: &core::Dtype,
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    if !args.is_empty() {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
            args,
        }));
    }

    Ok(vec![Value::Dtype(dtype.clone())])
}

// Type operation implementations
fn apply_pack(
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    todo!("Pack: Nat^k → Type")
}

fn apply_unpack(
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    todo!("Unpack: Type → Nat^k")
}

fn apply_shape(
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    todo!("Shape: Tensor → Shape")
}

// Nat operation implementations
fn apply_nat_constant(
    _n: usize,
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    todo!("NatConstant: [] → Nat")
}

fn apply_nat_mul(
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    todo!("NatMul: Nat^n → Nat")
}

fn apply_nat_add(
    _args: Vec<Value>,
    _ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    todo!("NatAdd: Nat^n → Nat")
}
