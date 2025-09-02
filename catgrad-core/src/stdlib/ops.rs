use crate::category::core;
use crate::category::lang;

use super::nn::*;

use std::collections::HashMap;

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        vec![$($x),*].try_into().expect("invalid operation name")
    };
}

// The set of operations in the category
pub struct Environment {
    pub operations: HashMap<lang::Path, lang::TypedTerm>,
}

// Interpretations of declared operations
pub fn core_declarations() -> std::collections::HashMap<lang::Path, core::Operation> {
    use crate::category::core::{NatOp, Operation, ScalarOp::*, TensorOp::*, TypeOp};
    use std::collections::HashMap;
    HashMap::from([
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
    ])
}

pub fn stdlib() -> Environment {
    let operations = HashMap::from([
        (
            path!["nn", "sigmoid"],
            lang::TypedTerm {
                term: sigmoid_term(),
                source_type: sigmoid_source(),
                target_type: sigmoid_target(),
            },
        ),
        (
            path!["nn", "exp"],
            lang::TypedTerm {
                term: exp_term(),
                source_type: exp_source(),
                target_type: exp_target(),
            },
        ),
    ]);

    Environment { operations }
}
