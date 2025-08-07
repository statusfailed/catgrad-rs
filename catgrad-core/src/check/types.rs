use crate::category::bidirectional::*;
use crate::category::core::Dtype;
use crate::ssa::*;
use open_hypergraphs::lax::{EdgeId, NodeId};
use std::fmt::{Display, Formatter, Result as FmtResult};

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
    Constant(usize),
    Mul(Vec<NatExpr>),
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
}
pub type ApplyResult = Result<Vec<Value>, ApplyError>;

////////////////////////////////////////////////////////////////////////////////
// Display instances

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Value::Type(type_expr) => write!(f, "{type_expr}"),
            Value::Nat(nat_expr) => write!(f, "{nat_expr}"),
            Value::Dtype(dtype_expr) => write!(f, "{dtype_expr}"),
            Value::Tensor(type_expr) => write!(f, "{type_expr}"),
        }
    }
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TypeExpr::Var(n) => write!(f, "v{n}"),
            TypeExpr::NdArrayType(array_type) => write!(f, "{array_type}"),
        }
    }
}

impl Display for NdArrayType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}[", self.dtype)?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

impl Display for NatExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            NatExpr::Var(n) => write!(f, "v{n}"),
            NatExpr::Constant(c) => write!(f, "{c}"),
            NatExpr::Mul(terms) => {
                if terms.is_empty() {
                    write!(f, "1")
                } else if terms.len() == 1 {
                    write!(f, "{}", terms[0])
                } else {
                    write!(f, "(")?;
                    for (i, term) in terms.iter().enumerate() {
                        if i > 0 {
                            write!(f, "*")?;
                        }
                        write!(f, "{term}")?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}

impl Display for DtypeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            DtypeExpr::Var(n) => write!(f, "v{n}"),
            DtypeExpr::Constant(dtype) => write!(f, "{dtype}"),
        }
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Dtype::F32 => write!(f, "f32"),
            Dtype::U32 => write!(f, "u32"),
        }
    }
}
