//! This is the primary interface for users to construct programs.
use open_hypergraphs::lax::*;
use std::collections::HashMap;
use std::fmt;

use crate::category::core;

pub use crate::category::core::Dtype;
pub use crate::category::shape::Object;

use super::path::*;

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    F32(f32),
    U32(u32),
    Dtype(core::Dtype),
}

// Operations are the core shape operations (and operation schemas like 'Constant') extended
// with "Definitions".
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Operations declared to exist, but without a definition (externally interpreted)
    Declaration(Path),

    /// Extended set of base operations with definitions
    Definition(Path),

    /// Literals (floats, ints, dtypes, nats, etc.) are elements (maps with unit domain)
    Literal(Literal),
}

// The actual data of a definition: a term, its type, and ...?
pub struct OperationDefinition {
    pub term: Term,
    pub source_type: Term,
    pub target_type: Term,
}

// The set of operations in the category
pub struct Environment {
    pub operations: HashMap<Path, OperationDefinition>,
}

pub type Term = OpenHypergraph<Object, Operation>;
pub type Var = open_hypergraphs::lax::var::Var<Object, Operation>;

use std::cell::RefCell;
use std::rc::Rc;
pub type Builder = Rc<RefCell<Term>>;

////////////////////////////////////////////////////////////////////////////////
// Instances

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::F32(v) => write!(f, "{v}"),
            Literal::U32(v) => write!(f, "{v}"),
            Literal::Dtype(dtype) => write!(f, "{dtype:?}"),
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Declaration(path) => write!(f, "{path}"),
            Operation::Definition(path) => write!(f, "{path}"),
            Operation::Literal(literal) => write!(f, "{literal}"),
        }
    }
}
