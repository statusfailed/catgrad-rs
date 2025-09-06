use crate::category::core;
use crate::category::lang;

use super::def::*;

use std::collections::HashMap;

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        vec![$($x),*].try_into().expect("invalid operation name")
    };
}

/// Declared and defined operations.
/// Currently, a declaration must map to a Core operation (subject to change!)
#[derive(Debug, Clone)]
pub struct Environment {
    pub definitions: HashMap<lang::Path, lang::TypedTerm>,
    pub declarations: HashMap<lang::Path, core::Operation>,
}


/// Interpretations of declared operations
fn core_declarations() -> HashMap<lang::Path, core::Operation> {
    use crate::category::core::{NatOp, Operation, ScalarOp::*, TensorOp::*, TypeOp};
    use std::collections::HashMap;
    HashMap::from([
        (path!["cartesian", "copy"], Operation::Copy),
        // tensor ops (which actually affect tensor data)
        (path!["tensor", "add"], Operation::Tensor(Map(Add))),
        (path!["tensor", "neg"], Operation::Tensor(Map(Neg))),
        (path!["tensor", "mul"], Operation::Tensor(Map(Mul))),
        (path!["tensor", "div"], Operation::Tensor(Map(Div))),
        (path!["tensor", "pow"], Operation::Tensor(Map(Pow))),
        (path!["tensor", "matmul"], Operation::Tensor(MatMul)),
        (path!["tensor", "reshape"], Operation::Tensor(Reshape)),
        (path!["tensor", "broadcast"], Operation::Tensor(Broadcast)),
        (path!["tensor", "cast"], Operation::Tensor(Cast)),
        // Mixed Tensor/Type ops
        (path!["tensor", "shape"], Operation::Type(TypeOp::Shape)),
        (path!["tensor", "dtype"], Operation::Type(TypeOp::Dtype)),
        // Shape ops
        (path!["shape", "pack"], Operation::Type(TypeOp::Pack)),
        (path!["shape", "unpack"], Operation::Type(TypeOp::Unpack)),
        (path!["nat", "mul"], Operation::Nat(NatOp::Mul)),
    ])
}

// helper to simplify stdlib defs list
fn to_pair<const A: usize, const B: usize, T: Def<A, B>>(def: T) -> (lang::Path, lang::TypedTerm) {
    (def.path(), def.term().unwrap())
}

/// Standard library of definitions
fn definitions() -> HashMap<lang::Path, lang::TypedTerm> {
    use super::nn::*;

    // NOTE: can't just map this since each invocation of to_pair is differently typed
    HashMap::from([
        to_pair(Sigmoid),
        to_pair(Exp),
        //
    ])
}

/// Standard library declarations and definitions
pub fn stdlib() -> Environment {
    Environment {
        declarations: core_declarations(),
        definitions: definitions(),
    }
}
