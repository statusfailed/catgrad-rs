// Catgrad's shape checker is an abstract interpreter for the *shaped* dialect.
use crate::category::shape::*;
use crate::ssa::*;
use open_hypergraphs::lax::{EdgeId, NodeId};

#[derive(Debug)]
pub enum ShapeCheckError {
    /// SSA ordering was invalid: an op depended on some arguments which did not have a value at
    /// time of [`apply`]
    EvaluationOrder(EdgeId),

    /// Some nodes in the term were not evaluated during shapechecking
    Unevaluated(Vec<NodeId>),

    /// Error trying to apply an operation
    ApplyError(ApplyError, SSA<Object, Operation>, Vec<Value>),
}

pub type ShapeCheckResult = Result<Vec<Value>, ShapeCheckError>;

////////////////////////////////////////////////////////////////////////////////
// Value types

/// type-tagged values per node
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    /// An NdArrayType
    Type(TypeExpr),

    /// An expression whose value is a natural number
    Nat(NatExpr),

    /// A dtype (either a var, or constant)
    Dtype(DtypeExpr),

    /// A tensor (represented abstractly by its NdArrayType, without data)
    Tensor(TypeExpr),
}

// For now, type expressions are either completely opaque, or *concrete* lists of nat exprs.
// This means concat is partial: if any Var appears, we cannot handle it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeExpr {
    Var(usize),
    NdArrayType(NdArrayType),
}

/// TODO: keep *normalized* instead as a Vec<Nat>
/// A symbolic shape value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NdArrayType {
    pub dtype: DtypeExpr,
    pub shape: Vec<NatExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NatExpr {
    Var(usize),
    Mul(Vec<NatExpr>),
    Constant(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DtypeExpr {
    Var(usize),
    Constant(Dtype),
}

////////////////////////////////////////////////////////////////////////////////
// Shapecheck a term

/*
pub fn check(term: Term, ty: Term) -> ShapeCheckResult {
    // Set a "Var" (symbolic) value for each input node
    let source_values = ty
        .sources
        .iter()
        .enumerate()
        .map(|(i, id)| var(i, term.hypergraph.nodes[id.0].clone()))
        .collect();

    check_with(term, source_values)
}
*/

/// Assign a shape value to each node in a term (hypergraph).
pub fn check(term: Term, source_values: Vec<Value>) -> ShapeCheckResult {
    // Create evaluation state
    let n = term.hypergraph.nodes.len();
    let mut state: Vec<Option<Value>> = vec![None; n];

    // Set a "Var" (symbolic) value for each input node
    for (i, id) in term.sources.iter().enumerate() {
        state[id.0] = Some(source_values[i].clone());
    }

    // Create SSA
    let ssa = ssa(term.to_open_hypergraph());

    // Iterate through SSA
    for op in ssa {
        // read arg values from graph
        let mut args = vec![];
        for (NodeId(i), _) in &op.sources {
            if let Some(value) = state[*i].clone() {
                args.push(value)
            } else {
                let _v = state[*i].clone();
                return Err(ShapeCheckError::EvaluationOrder(op.edge_id));
            }
        }

        // Compute output values and write into the graph
        let coargs =
            apply(&op, &args).map_err(|e| ShapeCheckError::ApplyError(e, op.clone(), args))?;
        assert_eq!(coargs.len(), op.targets.len());
        for ((NodeId(i), _), value) in op.targets.iter().zip(coargs.into_iter()) {
            state[*i] = Some(value)
        }
    }

    node_values(state).map_err(ShapeCheckError::Unevaluated)
}

fn _var(id: usize, ty: Object) -> Value {
    match ty {
        Object::Nat => Value::Nat(NatExpr::Var(id)),
        Object::NdArrayType => Value::Type(TypeExpr::Var(id)),
        Object::Tensor => Value::Tensor(TypeExpr::Var(id)),
        Object::Dtype => Value::Dtype(DtypeExpr::Var(id)),
    }
}

fn node_values<T>(v: Vec<Option<T>>) -> Result<Vec<T>, Vec<NodeId>> {
    let mut values = Vec::with_capacity(v.len());
    let mut none_indices = Vec::new();

    for (i, opt) in v.into_iter().enumerate() {
        match opt {
            Some(val) => values.push(val),
            None => none_indices.push(NodeId(i)),
        }
    }

    if none_indices.is_empty() {
        Ok(values)
    } else {
        Err(none_indices)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Apply and helper functions

#[derive(Debug)]
pub enum ApplyError {
    ArityError,
    TypeError,
}
pub type ApplyResult = Result<Vec<Value>, ApplyError>;

// Get a value for each resulting NodeId.
pub fn apply(ssa: &SSA<Object, Operation>, args: &[Value]) -> ApplyResult {
    // Unwrap each optional value-
    match &ssa.op {
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
        shape,
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
            let mut result = Vec::new();

            // First return the dtype
            result.push(Value::Dtype(ty.dtype.clone()));

            // Then return each dimension as individual nats
            for dim in &ty.shape {
                result.push(Value::Nat(dim.clone()));
            }

            Ok(result)
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
        _ => todo!(),
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
