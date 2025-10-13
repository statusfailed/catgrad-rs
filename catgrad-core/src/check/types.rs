use crate::category::core;
use crate::category::lang::*;
use crate::ssa::*;

use open_hypergraphs::lax::{EdgeId, NodeId};

use std::collections::HashMap;

#[derive(Debug)]
pub enum ShapeCheckError {
    /// SSA ordering was invalid: an op depended on some arguments which did not have a value at
    /// time of operation application
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

pub(crate) use crate::typecheck::interpreter::Value;
pub(crate) use crate::typecheck::value_types::TypeExpr;
pub(crate) use crate::typecheck::value_types::*;

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
                    .map(NatExpr::Constant)
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

impl<'a> IntoIterator for &'a Parameters {
    type Item = &'a Path;
    type IntoIter = std::collections::hash_map::Keys<'a, Path, Type>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.keys()
    }
}

impl Parameters {
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, Path, Type> {
        self.0.keys()
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
