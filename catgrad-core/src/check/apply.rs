use super::types::*;
use crate::category::{bidirectional, shape::*};
use crate::ssa::SSA;

// Get a value for each resulting NodeId.
pub fn s_apply(
    op: &Operation,
    args: &[Value],
    ssa: &SSA<bidirectional::Object, bidirectional::Operation>,
) -> ApplyResult {
    // Unwrap each optional value-
    match &op {
        Operation::Type(op) => type_op(op, args),
        Operation::DtypeConstant(d) => Ok(vec![Value::Dtype(DtypeExpr::Constant(d.clone()))]),
        Operation::Nat(op) => nat_op(op, args),
        Operation::Tensor(op) => tensor_op(op, args),
        Operation::Copy => apply_copy(args, ssa),
    }
}

fn apply_copy(
    args: &[Value],
    ssa: &SSA<bidirectional::Object, bidirectional::Operation>,
) -> ApplyResult {
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    Ok(vec![args[0].clone(); ssa.targets.len()])
}

////////////////////////////////////////
// TypeOp application + helpers

fn type_op(op: &TypeOp, args: &[Value]) -> ApplyResult {
    match op {
        TypeOp::Pack => type_pack(args),
        TypeOp::Unpack => type_unpack(args),
        TypeOp::Shape => type_shape(args),
    }
}

fn type_shape(args: &[Value]) -> ApplyResult {
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    match &args[0] {
        Value::Tensor(TypeExpr::Var(v)) => Ok(vec![Value::Shape(ShapeExpr::OfType(*v))]),
        Value::Tensor(TypeExpr::NdArrayType(s)) => Ok(vec![Value::Shape(s.shape.clone())]),
        _ => Err(ApplyError::TypeError),
    }
}

fn type_pack(args: &[Value]) -> ApplyResult {
    // type_pack should have n nat args
    // Creates an NdArrayType from a dtype and individual nat dimensions
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let mut shape = Vec::new();
    for arg in args {
        match arg {
            Value::Nat(n) => shape.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    Ok(vec![Value::Shape(ShapeExpr::Shape(shape))])
}

fn type_unpack(args: &[Value]) -> ApplyResult {
    // type_unpack should have exactly 1 NdArrayType arg
    // Returns dtype + individual nat dimensions
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    match &args[0] {
        Value::Shape(s) => {
            match &s {
                ShapeExpr::Var(_) => Err(ApplyError::TypeError),
                ShapeExpr::OfType(_) => Err(ApplyError::TypeError),
                ShapeExpr::Shape(nat_exprs) => {
                    let mut result = Vec::new();

                    // Return each dimension as individual nats
                    for dim in nat_exprs {
                        result.push(Value::Nat(dim.clone()));
                    }

                    Ok(result)
                }
            }
        }
        _ => {
            println!("here?");
            Err(ApplyError::TypeError)
        }
    }
}

////////////////////////////////////////
// Nat op application + helpers

fn nat_op(op: &NatOp, args: &[Value]) -> ApplyResult {
    match op {
        NatOp::Constant(n) => {
            if !args.is_empty() {
                return Err(ApplyError::ArityError);
            }
            Ok(vec![Value::Nat(NatExpr::Constant(*n))])
        }
        NatOp::Mul => nat_mul(args),
        NatOp::Add => nat_add(args),
    }
}

fn nat_mul(args: &[Value]) -> ApplyResult {
    // Multiply n natural numbers together
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let mut nat_exprs = Vec::new();
    for arg in args {
        match arg {
            Value::Nat(n) => nat_exprs.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    // If there's only one argument, return it directly
    if nat_exprs.len() == 1 {
        Ok(vec![Value::Nat(nat_exprs.into_iter().next().unwrap())])
    } else {
        Ok(vec![Value::Nat(NatExpr::Mul(nat_exprs))])
    }
}

fn nat_add(args: &[Value]) -> ApplyResult {
    // Multiply n natural numbers together
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let mut nat_exprs = Vec::new();
    for arg in args {
        match arg {
            Value::Nat(n) => nat_exprs.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    // If there's only one argument, return it directly
    if nat_exprs.len() == 1 {
        Ok(vec![Value::Nat(nat_exprs.into_iter().next().unwrap())])
    } else {
        Ok(vec![Value::Nat(NatExpr::Add(nat_exprs))])
    }
}

////////////////////////////////////////
// Tensor op application & helpers

fn tensor_op(op: &TensorOp, args: &[Value]) -> ApplyResult {
    match op {
        TensorOp::Stack => tensor_stack(args),
        TensorOp::Split => tensor_split(args),
        TensorOp::Reshape => tensor_reshape(args),
        TensorOp::MatMul => tensor_matmul(args),
        TensorOp::Map(_) => Ok(vec![args[0].clone()]), // TODO: need to know op type, assert all args have same type!
        //TensorOp::Copy => // TODO: need to know op type!
        TensorOp::Broadcast => tensor_broadcast(args),
        op => todo!("operation {op:?}"),
    }
}

fn tensor_broadcast(args: &[Value]) -> ApplyResult {
    match (&args[0], &args[1]) {
        (Value::Tensor(TypeExpr::NdArrayType(t)), Value::Shape(shape)) => {
            match (&t.shape, &shape) {
                (ShapeExpr::Shape(s1), ShapeExpr::Shape(s2)) => {
                    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                        dtype: t.dtype.clone(),
                        shape: ShapeExpr::Shape([s1.clone(), s2.clone()].concat()),
                    }))])
                }
                (ShapeExpr::Shape(s), ShapeExpr::Var(v)) if s.is_empty() => {
                    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                        dtype: t.dtype.clone(),
                        shape: ShapeExpr::Var(*v),
                    }))])
                }
                _ => Err(ApplyError::TypeError),
            }
        }
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_stack(_args: &[Value]) -> ApplyResult {
    // Construct a NdArrayType::Concat
    todo!()
}

fn tensor_split(_args: &[Value]) -> ApplyResult {
    // len = 1 or error
    todo!()
}

fn tensor_matmul(args: &[Value]) -> ApplyResult {
    // reshape takes 2 args: target type and input tensor
    if args.len() != 2 {
        return Err(ApplyError::ArityError);
    }

    match (&args[0], &args[1]) {
        (Value::Tensor(TypeExpr::NdArrayType(t)), Value::Tensor(TypeExpr::NdArrayType(u))) => {
            if t.dtype != u.dtype {
                return Err(ApplyError::TypeError);
            }

            let dtype = t.dtype.clone();

            if let (ShapeExpr::Shape(t), ShapeExpr::Shape(u)) = (t.shape.clone(), u.shape.clone()) {
                // Same rank
                if t.len() != u.len() {
                    return Err(ApplyError::TypeError);
                }

                // need at least 2 dims in each to matmul
                if t.len() < 2 {
                    return Err(ApplyError::TypeError);
                }

                // Check contraction dimension matches
                if t[t.len() - 1] != u[u.len() - 2] {
                    return Err(ApplyError::TypeError);
                }

                // check prefix dimensions match - e.g. for (N0, ..., Nk, A, B) and (N0..Nk, B, C), N0..Nk
                // should match.
                let prefix_len = t.len() - 2;
                if t[..prefix_len] != u[..prefix_len] {
                    return Err(ApplyError::TypeError);
                }

                // Result shape: all but last element of t, then append final element of u
                let mut shape = t[..t.len() - 1].to_vec();
                shape.push(u[u.len() - 1].clone());

                Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                    dtype,
                    shape: ShapeExpr::Shape(shape),
                }))])
            } else {
                Err(ApplyError::TypeError)
            }
        }
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_reshape(args: &[Value]) -> ApplyResult {
    // reshape takes 2 args: target type and input tensor
    if args.len() != 2 {
        return Err(ApplyError::ArityError);
    }

    match (&args[0], &args[1]) {
        (
            Value::Shape(target_shape),
            Value::Tensor(TypeExpr::NdArrayType(NdArrayType { dtype, shape })),
        ) => {
            if !shapes_isomorphic(shape, target_shape) {
                return Err(ApplyError::ShapeMismatch(
                    shape.clone(),
                    target_shape.clone(),
                ));
            }

            let target_type = NdArrayType {
                dtype: dtype.clone(),
                shape: target_shape.clone(),
            };

            Ok(vec![Value::Tensor(TypeExpr::NdArrayType(target_type))])
        }
        _ => Err(ApplyError::TypeError),
    }
}

// Return normalized shapes for s, t
// TODO: return ApplyResult
fn shapes_isomorphic(s: &ShapeExpr, t: &ShapeExpr) -> bool {
    match (s, t) {
        (ShapeExpr::Var(v), ShapeExpr::Var(u)) => v == u,
        (ShapeExpr::OfType(v), ShapeExpr::OfType(u)) => v == u,
        (ShapeExpr::Shape(s), ShapeExpr::Shape(t)) => {
            super::isomorphism::isomorphic(s.clone(), t.clone())
        }
        _ => false,
    }
}
