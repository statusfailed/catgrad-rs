use super::types::*;
use crate::category::shape::*;

// Get a value for each resulting NodeId.
pub fn s_apply(op: &Operation, args: &[Value]) -> ApplyResult {
    // Unwrap each optional value-
    match &op {
        Operation::Type(op) => type_op(op, args),
        Operation::DtypeConstant(d) => Ok(vec![Value::Dtype(DtypeExpr::Constant(d.clone()))]),
        Operation::Nat(op) => nat_op(op, args),
        Operation::Tensor(op) => tensor_op(op, args),
        Operation::Copy => Ok(args.iter().cloned().chain(args.iter().cloned()).collect()),
    }
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
        Value::Tensor(s) => Ok(vec![Value::Type(s.clone())]),
        _ => Err(ApplyError::TypeError),
    }
}

fn type_pack(args: &[Value]) -> ApplyResult {
    // type_pack should have 1 dtype + n nat args
    // Creates an NdArrayType from a dtype and individual nat dimensions
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let dtype = match &args[0] {
        Value::Dtype(d) => d.clone(),
        _ => return Err(ApplyError::TypeError),
    };

    let mut shape = Vec::new();
    for arg in &args[1..] {
        match arg {
            Value::Nat(n) => shape.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    Ok(vec![Value::Type(TypeExpr::NdArrayType(NdArrayType {
        dtype,
        shape: ShapeExpr::Shape(shape),
    }))])
}

fn type_unpack(args: &[Value]) -> ApplyResult {
    // type_unpack should have exactly 1 NdArrayType arg
    // Returns dtype + individual nat dimensions
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    match &args[0] {
        Value::Type(TypeExpr::NdArrayType(ty)) => {
            match &ty.shape {
                ShapeExpr::Var(_) => Err(ApplyError::TypeError),
                ShapeExpr::Shape(nat_exprs) => {
                    let mut result = Vec::new();

                    // First return the dtype
                    result.push(Value::Dtype(ty.dtype.clone()));

                    // Then return each dimension as individual nats
                    for dim in nat_exprs {
                        result.push(Value::Nat(dim.clone()));
                    }

                    Ok(result)
                }
            }
        }
        _ => Err(ApplyError::TypeError),
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
        op => todo!("{op:?}"),
    }
}

fn tensor_broadcast(args: &[Value]) -> ApplyResult {
    match (&args[0], &args[1]) {
        (Value::Tensor(TypeExpr::NdArrayType(t)), Value::Type(TypeExpr::NdArrayType(u))) => {
            if t.dtype != u.dtype {
                return Err(ApplyError::TypeError);
            }

            let dtype = t.dtype.clone();

            match (&t.shape, &u.shape) {
                (ShapeExpr::Shape(t), ShapeExpr::Shape(u)) => {
                    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                        dtype,
                        shape: ShapeExpr::Shape([t.clone(), u.clone()].concat()),
                    }))])
                }
                (ShapeExpr::Shape(t), ShapeExpr::Var(v)) if t.is_empty() => {
                    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                        dtype,
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

    // TODO: check output tensor is isomorphic!
    match (&args[0], &args[1]) {
        (Value::Type(target_type), Value::Tensor(_input_tensor)) => {
            // The output tensor has the target type
            Ok(vec![Value::Tensor(target_type.clone())])
        }
        _ => Err(ApplyError::TypeError),
    }
}
