use crate::category::bidirectional::*;
use crate::category::core::Dtype;
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
    /// TODO: remove
    Type(TypeExpr),

    Shape(ShapeExpr),

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeExpr {
    Var(usize),
    OfType(usize), // shape of a *type* variable
    Shape(Vec<NatExpr>),
}

/// TODO: keep *normalized* instead as a Vec<Nat>
/// A symbolic shape value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NdArrayType {
    pub dtype: DtypeExpr,
    pub shape: ShapeExpr,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NatExpr {
    Var(usize),
    Constant(usize),
    Mul(Vec<NatExpr>),
    Add(Vec<NatExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DtypeExpr {
    Var(usize),
    Constant(Dtype),
}

#[derive(Debug)]
pub enum ApplyError {
    ArityError,
    TypeError,
    UnknownOp(Path),
    ShapeMismatch(ShapeExpr, ShapeExpr),
}
pub type ApplyResult = Result<Vec<Value>, ApplyError>;
