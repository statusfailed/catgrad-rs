//! Shape operation implementations for the interpreter

use super::{ApplyError, ApplyErrorKind, Value};
use crate::category::bidirectional::{Object, Operation};
use crate::category::{core, core::Shape, shape};
use crate::ssa::SSA;

/// Apply a Type operation
pub(crate) fn apply_type_op(
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
pub(crate) fn apply_nat_op(
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
pub(crate) fn apply_dtype_constant(
    dtype: &core::Dtype,
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    expect_arity(&args, 0, ssa)?;
    Ok(vec![Value::Dtype(dtype.clone())])
}

// Type operation implementations
pub(crate) fn apply_pack(
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    let mut shape = Vec::new();
    for arg in &args {
        match arg {
            Value::Nat(n) => shape.push(*n),
            _ => return type_error(ssa, args),
        }
    }
    Ok(vec![Value::Shape(Shape(shape))])
}

pub(crate) fn apply_unpack(
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    expect_arity(&args, 1, ssa)?;
    match &args[0] {
        Value::Shape(shape) => {
            let mut result = Vec::new();
            for dim in shape.0.iter() {
                result.push(Value::Nat(*dim));
            }
            Ok(result)
        }
        _ => type_error(ssa, args),
    }
}

pub(crate) fn apply_shape(
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    expect_arity(&args, 1, ssa)?;
    match &args[0] {
        Value::NdArray(tensor) => Ok(vec![Value::Shape(tensor.shape())]),
        _ => type_error(ssa, args),
    }
}

// Nat operation implementations
pub(crate) fn apply_nat_constant(
    n: usize,
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    expect_arity(&args, 0, ssa)?;
    Ok(vec![Value::Nat(n)])
}

pub(crate) fn apply_nat_mul(
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    if args.is_empty() {
        return Ok(vec![Value::Nat(1)]);
    }
    let mut result = 1;
    for arg in &args {
        match arg {
            Value::Nat(n) => result *= n,
            _ => return type_error(ssa, args),
        }
    }
    Ok(vec![Value::Nat(result)])
}

pub(crate) fn apply_nat_add(
    args: Vec<Value>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    if args.is_empty() {
        return Ok(vec![Value::Nat(0)]);
    }
    let mut result = 0;
    for arg in &args {
        match arg {
            Value::Nat(n) => result += n,
            _ => return type_error(ssa, args),
        }
    }
    Ok(vec![Value::Nat(result)])
}

////////////////////////////////////////////////////////////////////////////////
// utilities

pub fn type_error(
    ssa: &SSA<Object, Operation>,
    args: Vec<Value>,
) -> Result<Vec<Value>, Box<ApplyError>> {
    Err(Box::new(ApplyError {
        kind: ApplyErrorKind::TypeError,
        ssa: ssa.clone(),
        args,
    }))
}

pub fn expect_arity(
    args: &[Value],
    expected: usize,
    ssa: &SSA<Object, Operation>,
) -> Result<(), Box<ApplyError>> {
    if args.len() != expected {
        type_error(ssa, args.to_vec())?;
    }
    Ok(())
}
