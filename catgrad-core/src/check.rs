// Catgrad's shape checker is an abstract interpreter for the *shaped* dialect.
use crate::category::shape::*;
use crate::ssa::*;
use open_hypergraphs::lax::NodeId;

pub enum ShapeCheckError {
    /// SSA ordering was invalid: an op depended on some arguments which did not have a value at
    /// time of [`apply`]
    EvaluationOrder,

    /// Some nodes in the term were not evaluated during shapechecking
    Unevaluated(Vec<NodeId>),

    ApplyError(ApplyError),
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
    Var(NodeId),
    NdArrayType(NdArrayType),
}

/// TODO: keep *normalized* instead as a Vec<Nat>
/// A symbolic shape value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NdArrayType {
    dtype: Dtype,
    shape: Vec<NatExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NatExpr {
    Var(NodeId),
    Mul(Vec<NatExpr>),
    Constant(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DtypeExpr {
    Var(NodeId),
    Constant(Dtype),
}

////////////////////////////////////////////////////////////////////////////////
// Shapecheck a term

/// Assign a shape value to each node in a term (hypergraph).
pub fn check(term: Term) -> ShapeCheckResult {
    // Create evaluation state
    let n = term.hypergraph.nodes.len();
    let mut state: Vec<Option<Value>> = vec![None; n];

    // Set a "Var" (symbolic) value for each input node
    for id in term.sources.iter() {
        state[id.0] = Some(var(*id, term.hypergraph.nodes[id.0].clone()))
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
                return Err(ShapeCheckError::EvaluationOrder);
            }
        }

        // Compute output values and write into the graph
        let coargs = apply(&op, &args).map_err(ShapeCheckError::ApplyError)?;
        for ((NodeId(i), _), value) in op.targets.iter().zip(coargs.into_iter()) {
            state[*i] = Some(value)
        }
    }

    node_values(state).map_err(ShapeCheckError::Unevaluated)
}

fn var(id: NodeId, ty: Object) -> Value {
    match ty {
        Object::Nat => Value::Nat(NatExpr::Var(id)),
        Object::NdArrayType => Value::Type(TypeExpr::Var(id)),
        Object::Tensor => Value::Tensor(TypeExpr::Var(id)),
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

pub enum ApplyError {
    ArityError,
    TypeError,
}
pub type ApplyResult = Result<Vec<Value>, ApplyError>;

// Get a value for each resulting NodeId.
pub fn apply(ssa: &SSA<Object, Operation>, args: &[Value]) -> ApplyResult {
    // Unwrap each optional value-
    match &ssa.op {
        Operation::Type(op) => shape_op(&op, args),
        Operation::Nat(op) => nat_op(&op, args),
        Operation::Tensor(op) => tensor_op(&op, args),
        Operation::Copy => Ok(args.iter().cloned().chain(args.iter().cloned()).collect()),
    }
}

////////////////////////////////////////
// TypeOp application + helpers

fn shape_op(op: &TypeOp, args: &[Value]) -> ApplyResult {
    use TypeOp::*;
    match op {
        Concat => type_concat(args),
        Split => type_split(args),
        _ => todo!(),
    }
}

fn type_concat(args: &[Value]) -> ApplyResult {
    // concat should have n Value::Type(t) args.
    // Match on all, then compute their concatenation as the final result.
    let mut types: Vec<NdArrayType> = Vec::new();
    for arg in args {
        match arg {
            Value::Type(TypeExpr::NdArrayType(ty)) => {
                types.push(ty.clone());
            }
            _ => return Err(ApplyError::TypeError),
        }
    }

    let dtype = unique(types.iter().map(|t| &t.dtype))
        .ok_or(ApplyError::TypeError)?
        .clone();
    let shape = types.into_iter().flat_map(|t| t.shape).collect();

    Ok(vec![Value::Type(TypeExpr::NdArrayType(NdArrayType {
        dtype,
        shape,
    }))])
}

fn type_split(args: &[Value]) -> ApplyResult {
    // len = 1 or error
    todo!()
}

// Get a unique item from a list, or None if no or multiple elements.
fn unique<T: PartialEq + Clone>(mut iter: impl Iterator<Item = T>) -> Option<T> {
    let first = iter.next()?;
    for item in iter {
        if item != first {
            return None;
        }
    }
    Some(first)
}

////////////////////////////////////////
// Nat op application + helpers

fn nat_op(op: &NatOp, args: &[Value]) -> ApplyResult {
    todo!()
}

fn nat_mul(args: &[Value]) -> ApplyResult {
    // Construct a NdArrayType::Concat
    todo!()
}

fn nat_lift(args: &[Value]) -> ApplyResult {
    // len = 1 or error
    todo!()
}

////////////////////////////////////////
// Tensor op application & helpers

fn tensor_op(op: &TensorOp, args: &[Value]) -> ApplyResult {
    match op {
        TensorOp::Stack => tensor_stack(args),
        TensorOp::Split => tensor_split(args),
        _ => todo!(),
    }
}

fn tensor_stack(args: &[Value]) -> ApplyResult {
    // Construct a NdArrayType::Concat
    todo!()
}

fn tensor_split(args: &[Value]) -> ApplyResult {
    // len = 1 or error
    todo!()
}
