use super::types::*;
use open_hypergraphs::lax::var;

//macro_rules! op {
//[$($x:expr),* $(,)?] => {
//Operation::Declaration(vec!["op", $($x),*].try_into().expect("invalid operation name"))
//};
//}

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        vec![$($x),*].try_into().expect("invalid operation name")
    };
}

macro_rules! op {
    [$($x:expr),* $(,)?] => {
        Operation::Declaration(path![$($x),*])
    };
}

//Operation::Declaration(vec![$($x),*].try_into().expect("invalid operation name"))

////////////////////////////////////////////////////////////////////////////////
// Types

// Copy lets us use HasVar
impl var::HasVar for Operation {
    fn var() -> Self {
        op!["cartesian", "copy"]
    }
}

impl var::HasAdd<Object, Operation> for Operation {
    fn add(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        match lhs {
            Object::Nat => (lhs, op!["nat", "add"]),
            Object::Tensor => (lhs, op!["tensor", "add"]),
            _ => panic!("multiply undefined for {lhs:?}"),
        }
    }
}

impl var::HasMul<Object, Operation> for Operation {
    fn mul(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        // NOTE: this is a bit of a hack- we explicitly treat nat/tensor muls differently.
        // Would be better to have proper polymorphic ops
        match lhs {
            Object::Nat => (lhs, op!["nat", "mul"]),
            Object::Tensor => (lhs, op!["tensor", "mul"]),
            _ => panic!("multiply undefined for {lhs:?}"),
        }
    }
}

impl var::HasDiv<Object, Operation> for Operation {
    fn div(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["tensor", "div"])
    }
}

impl var::HasNeg<Object, Operation> for Operation {
    fn neg(operand_type: Object) -> (Object, Operation) {
        (operand_type, op!["tensor", "neg"])
    }
}

pub fn pow(builder: &Builder, value: Var, exponent: Var) -> Var {
    var::fn_operation(
        builder,
        &[value, exponent],
        Object::Tensor,
        op!["tensor", "pow"],
    )
}

////////////////////////////////////////////////////////////////////////////////
// Declarations

pub fn lit(builder: &Builder, lit: Literal) -> Var {
    var::fn_operation(builder, &[], Object::Tensor, Operation::Literal(lit))
}

pub fn constant_f32(builder: &Builder, v: f32) -> Var {
    lit(builder, Literal::F32(v))
}

/// Pack a fixed number of Nat values into a specific shape
pub fn pack<const N: usize>(builder: &Builder, extents: [Var; N]) -> Var {
    for x in &extents {
        assert_eq!(x.label, Object::Nat);
    }

    var::fn_operation(builder, &extents, Object::NdArrayType, op!["shape", "pack"])
}

/// Unpack a shape into a dtype and its constituent Nat dimensions
pub fn unpack<const N: usize>(builder: &Builder, x: Var) -> [Var; N] {
    assert_eq!(x.label, Object::NdArrayType);

    let ty = vec![Object::Nat; N];
    let elements = var::operation(builder, &[x], ty, op!["shape", "unpack"]);

    crate::util::iter_to_array(elements.into_iter()).expect("N elements")
}

// Tensor → NdArrayType
pub fn shape(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::NdArrayType, op!["tensor", "shape"])
}

pub fn dtype_constant(builder: &Builder, dtype: Dtype) -> Var {
    var::fn_operation(
        builder,
        &[],
        Object::Dtype,
        Operation::Literal(Literal::Dtype(dtype)),
    )
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Helpers

// x : t    s : Shape
// ------------------ broadcast
//     x : (s × t)
pub fn broadcast(builder: &Builder, x: Var, s: Var) -> Var {
    var::fn_operation(builder, &[x, s], Object::Tensor, op!["tensor", "broadcast"])
}

pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(builder, &[t, x], Object::Tensor, op!["tensor", "reshape"])
}

/// Batch matmul
pub fn matmul(builder: &Builder, f: Var, g: Var) -> Var {
    // checked during shapechecking, but errors easier to follow here.
    assert_eq!(f.label, Object::Tensor);
    assert_eq!(g.label, Object::Tensor);

    var::fn_operation(builder, &[f, g], Object::Tensor, op!["tensor", "matmul"])
}

////////////////////////////////////////////////////////////////////////////////
// S-interpretations of operations
//

// basic declarations
pub fn op_decls() -> std::collections::HashMap<super::path::Path, crate::category::core::Operation>
{
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
