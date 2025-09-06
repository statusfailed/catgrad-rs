use crate::category::lang::*;
use crate::category::{core, core::Dtype};
use crate::ssa::*;

use open_hypergraphs::lax::{EdgeId, NodeId};

use std::collections::HashMap;

#[derive(Debug)]
pub enum ShapeCheckError {
    /// SSA ordering was invalid: an op depended on some arguments which did not have a value at
    /// time of [`apply`]
    EvaluationOrder(EdgeId),

    /// Some nodes in the term were not evaluated during shapechecking
    Unevaluated(Vec<NodeId>),

    /// Error trying to apply an operation
    ApplyError(ApplyError, SSA<Object, Operation>, Vec<Value>),

    /// A cycle was detected in the input term
    CyclicHypergraph,

    /// SSA conversion error
    SSAError(SSAError),
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

// DtypeExpr::Var is allowed, but not as a top-level free variable in the program;
// it must resolve to a concrete value during shapechecking.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DtypeExpr {
    Var(usize),
    OfType(usize), // dtype of a *type* variable
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

impl From<SSAError> for ShapeCheckError {
    fn from(error: SSAError) -> Self {
        ShapeCheckError::SSAError(error)
    }
}

impl From<core::NdArrayType> for NdArrayType {
    fn from(core_type: core::NdArrayType) -> Self {
        NdArrayType {
            dtype: DtypeExpr::Constant(core_type.dtype),
            shape: ShapeExpr::Shape(
                core_type
                    .shape
                    .0
                    .into_iter()
                    .map(|dim| NatExpr::Constant(dim))
                    .collect(),
            ),
        }
    }
}

#[derive(PartialEq, Clone, Debug, Default)]
pub struct Parameters(pub HashMap<Path, Type>);

impl From<HashMap<Path, Type>> for Parameters {
    fn from(map: HashMap<Path, Type>) -> Self {
        Parameters(map)
    }
}

impl<const N: usize> From<[(Path, Type); N]> for Parameters {
    fn from(arr: [(Path, Type); N]) -> Self {
        Parameters(HashMap::from(arr))
    }
}

/*
pub fn param_declaration(
    name: &Path,
    dtype: Dtype,
    shape: Vec<usize>,
) -> (Path, (core::Operation, Type)) {
    let ty = tensor_type(dtype, shape);
    let path = path(vec!["param"]).concat(name);
    (path, core::Operation::Parameter(name.clone()), ty)
}

/// Helper to create a Value::Tensor with constant dtype and shape
fn tensor_type(dtype: Dtype, shape: Vec<usize>) -> Value {
    let shape_expr = ShapeExpr::Shape(
        shape
            .into_iter()
            .map(|dim| NatExpr::Constant(dim))
            .collect(),
    );

    let ndarray_type = NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape: shape_expr,
    };

    Value::Tensor(TypeExpr::NdArrayType(ndarray_type))
}
*/
