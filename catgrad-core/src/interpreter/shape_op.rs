//! Shape operation implementations for the interpreter

use super::backend::Backend;
use super::{ApplyError, ApplyErrorKind, Value};
use crate::category::lang::{Object, Operation};
use crate::category::{core, core::Shape};
use crate::ssa::SSA;

/// Apply a Type operation
pub(crate) fn apply_type_op<B: Backend>(
    type_op: &core::TypeOp,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match type_op {
        core::TypeOp::Pack => apply_pack(args, ssa),
        core::TypeOp::Unpack => apply_unpack(args, ssa),
        core::TypeOp::Shape => apply_shape(args, ssa),
    }
}

/// Apply a Nat operation
pub(crate) fn apply_nat_op<B: Backend>(
    nat_op: &core::NatOp,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match nat_op {
        core::NatOp::Constant(n) => apply_nat_constant(*n, args, ssa),
        core::NatOp::Mul => apply_nat_mul(args, ssa),
        core::NatOp::Add => apply_nat_add(args, ssa),
    }
}

/// Apply a Dtype constant operation
pub(crate) fn apply_dtype_constant<B: Backend>(
    dtype: &core::Dtype,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    expect_arity(&args, 0, ssa)?;
    Ok(vec![Value::Dtype(dtype.clone())])
}

// Type operation implementations
pub(crate) fn apply_pack<B: Backend>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    let mut shape = Vec::new();
    for arg in &args {
        match arg {
            Value::Nat(n) => shape.push(*n),
            _ => return type_error(ssa),
        }
    }
    Ok(vec![Value::Shape(Shape(shape))])
}

pub(crate) fn apply_unpack<B: Backend>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    expect_arity(&args, 1, ssa)?;
    match &args[0] {
        Value::Shape(shape) => {
            let mut result = Vec::new();
            for dim in shape.0.iter() {
                result.push(Value::Nat(*dim));
            }
            Ok(result)
        }
        _ => type_error(ssa),
    }
}

pub(crate) fn apply_shape<B: Backend>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    expect_arity(&args, 1, ssa)?;
    match &args[0] {
        Value::NdArray(tensor) => Ok(vec![Value::Shape(tensor.shape())]),
        _ => type_error(ssa),
    }
}

// Nat operation implementations
pub(crate) fn apply_nat_constant<B: Backend>(
    n: usize,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    expect_arity(&args, 0, ssa)?;
    Ok(vec![Value::Nat(n)])
}

pub(crate) fn apply_nat_mul<B: Backend>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.is_empty() {
        return Ok(vec![Value::Nat(1)]);
    }
    let mut result = 1;
    for arg in &args {
        match arg {
            Value::Nat(n) => result *= n,
            _ => return type_error(ssa),
        }
    }
    Ok(vec![Value::Nat(result)])
}

pub(crate) fn apply_nat_add<B: Backend>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.is_empty() {
        return Ok(vec![Value::Nat(0)]);
    }
    let mut result = 0;
    for arg in &args {
        match arg {
            Value::Nat(n) => result += n,
            _ => return type_error(ssa),
        }
    }
    Ok(vec![Value::Nat(result)])
}

////////////////////////////////////////////////////////////////////////////////
// utilities

pub fn type_error<B: Backend>(
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    Err(Box::new(ApplyError {
        kind: ApplyErrorKind::TypeError,
        ssa: ssa.clone(),
    }))
}

pub fn expect_arity<B: Backend>(
    args: &[Value<B>],
    expected: usize,
    ssa: &SSA<Object, Operation>,
) -> Result<(), Box<ApplyError>> {
    if args.len() != expected {
        type_error::<B>(ssa)?;
    }
    Ok(())
}
