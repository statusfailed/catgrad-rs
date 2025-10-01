use std::collections::HashMap;

use crate::category::{core, lang};
use crate::definition::Def;
use crate::interpreter;

/// Lower a `lang::Term` to a `core::Term`.
/// If a Definition or Declaration was not a core operation, it is assumed to be a definition in
/// [`core`].
pub fn to_core(term: lang::Term) -> core::Term {
    let core_ops = core_declarations();

    term.map_edges(|e| match e {
        lang::Operation::Definition(path) => Def::Def(path),
        lang::Operation::Declaration(path) => match core_ops.get(&path) {
            Some(op) => Def::Arr(op.clone()),
            None => Def::Def(path.clone()),
        },
        lang::Operation::Literal(lit) => Def::Arr(match lit {
            lang::Literal::F32(x) => {
                core::Operation::Tensor(core::TensorOp::Constant(core::Constant::F32(x)))
            }
            lang::Literal::U32(x) => {
                core::Operation::Tensor(core::TensorOp::Constant(core::Constant::U32(x)))
            }
            lang::Literal::Nat(n) => core::Operation::Nat(core::NatOp::Constant(n as usize)),
            lang::Literal::Dtype(d) => core::Operation::DtypeConstant(d),
        }),
    })
}

/// Lower an entire stdlib::Environment to the core, discarding type maps.
pub fn env_to_core(env: crate::stdlib::Environment) -> interpreter::Environment {
    let definitions = env
        .definitions
        .into_iter()
        .map(|(k, v)| {
            let v = to_core(v.term);
            (k, v)
        })
        .collect();

    interpreter::Environment { definitions }
}

////////////////////////////////////////////////////////////////////////////////
// NOTE: below is duplicated from stdlib/ops.rs.
// These should eventually replace that duplicate code!

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        vec![$($x),*].try_into().expect("invalid operation name")
    };
}

/// Interpretations of declared operations
pub(crate) fn core_declarations() -> HashMap<lang::Path, core::Operation> {
    use crate::category::core::{NatOp, Operation, ScalarOp::*, TensorOp::*, TypeOp};
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
        (path!["tensor", "transpose"], Operation::Tensor(Transpose)),
        (path!["tensor", "broadcast"], Operation::Tensor(Broadcast)),
        (path!["tensor", "cast"], Operation::Tensor(Cast)),
        (path!["tensor", "index"], Operation::Tensor(Index)),
        (path!["tensor", "sum"], Operation::Tensor(Sum)),
        (path!["tensor", "max"], Operation::Tensor(Max)),
        (path!["tensor", "arange"], Operation::Tensor(Arange)),
        (path!["tensor", "concat"], Operation::Tensor(Concat)),
        (path!["tensor", "scalar"], Operation::Tensor(Scalar)),
        // Mixed Tensor/Type ops
        (path!["tensor", "shape"], Operation::Type(TypeOp::Shape)),
        (path!["tensor", "dtype"], Operation::Type(TypeOp::Dtype)),
        // Shape ops
        (path!["shape", "pack"], Operation::Type(TypeOp::Pack)),
        (path!["shape", "unpack"], Operation::Type(TypeOp::Unpack)),
        (path!["nat", "mul"], Operation::Nat(NatOp::Mul)),
    ])
}
