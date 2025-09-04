use crate::category::core;
use crate::category::lang;

use super::def::*;

use std::collections::HashMap;

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        vec![$($x),*].try_into().expect("invalid operation name")
    };
}

/// The set of operations in the category
#[derive(Debug, Clone)]
pub struct Environment {
    pub operations: HashMap<lang::Path, lang::TypedTerm>,
}

/// Declared operations that map directly to a core op
/// NOTE: this interface is likely to change in future
#[derive(Debug, Clone)]
pub struct Declarations {
    pub operations: HashMap<lang::Path, core::Operation>,
}

/// Interpretations of declared operations
pub fn core_declarations() -> Declarations {
    use crate::category::core::{NatOp, Operation, ScalarOp::*, TensorOp::*, TypeOp};
    use std::collections::HashMap;
    let operations = HashMap::from([
        (path!["cartesian", "copy"], Operation::Copy),
        // tensor ops
        (path!["tensor", "add"], Operation::Tensor(Map(Add))),
        (path!["tensor", "neg"], Operation::Tensor(Map(Neg))),
        (path!["tensor", "mul"], Operation::Tensor(Map(Mul))),
        (path!["tensor", "div"], Operation::Tensor(Map(Div))),
        (path!["tensor", "pow"], Operation::Tensor(Map(Pow))),
        (path!["tensor", "matmul"], Operation::Tensor(MatMul)),
        (path!["tensor", "reshape"], Operation::Tensor(Reshape)),
        (path!["tensor", "broadcast"], Operation::Tensor(Broadcast)),
        (path!["tensor", "shape"], Operation::Type(TypeOp::Shape)),
        // shape ops
        (path!["shape", "pack"], Operation::Type(TypeOp::Pack)),
        (path!["shape", "unpack"], Operation::Type(TypeOp::Unpack)),
        (path!["nat", "mul"], Operation::Nat(NatOp::Mul)),
    ]);
    Declarations { operations }
}

fn to_pair<const A: usize, const B: usize, T: Def<A, B>>(def: T) -> (lang::Path, lang::TypedTerm) {
    (def.path(), def.term())
}

/// Standard library of definitions
pub fn stdlib() -> Environment {
    use super::nn::*;

    // NOTE: can't just map this since each invocation of to_pair is differently typed
    let operations = HashMap::from([
        to_pair(Sigmoid),
        to_pair(Exp),
        //
    ]);

    Environment { operations }
}
