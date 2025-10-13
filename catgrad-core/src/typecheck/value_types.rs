use crate::abstract_interpreter::{CoreSSA, EvalResult, InterpreterError};
use crate::category::core::Dtype;

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

/// A symbolic type value
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

////////////////////////////////////////////////////////////////////////////////
// Helper impls

impl TypeExpr {
    pub(crate) fn into_ndarraytype(self, ssa: &CoreSSA) -> EvalResult<NdArrayType> {
        match self {
            Self::NdArrayType(t) => Ok(t),
            _ => Err(InterpreterError::TypeError(ssa.edge_id)),
        }
    }

    pub(crate) fn into_shapeexpr_dtype(self, ssa: &CoreSSA) -> EvalResult<(ShapeExpr, DtypeExpr)> {
        match self {
            Self::NdArrayType(NdArrayType { shape, dtype }) => Ok((shape, dtype)),
            _ => Err(InterpreterError::TypeError(ssa.edge_id)),
        }
    }
}

// TODO: utils:
// - normalize and extract usize from list of nats (see FIXME: normalize)
// - get concrete shape expr from an NdArrayType
//      - unpack to dtype and ShapeExpr::Shape values?
// - add IncompatibleShapes error?
