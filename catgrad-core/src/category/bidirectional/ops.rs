use super::types::*;
use open_hypergraphs::lax::var;

macro_rules! op {
    [$($x:expr),* $(,)?] => {
        Operation::Declaration(vec!["op", $($x),*].try_into().expect("invalid operation name"))
    };
}

////////////////////////////////////////////////////////////////////////////////
// Types

// Copy lets us use HasVar
impl var::HasVar for Operation {
    fn var() -> Self {
        op!["copy"]
    }
}

impl var::HasAdd<Object, Operation> for Operation {
    fn add(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["add"])
    }
}

impl var::HasMul<Object, Operation> for Operation {
    fn mul(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        match lhs {
            Object::Nat => (
                lhs,
                Operation::Declaration(
                    vec!["nat", "mul"]
                        .try_into()
                        .expect("invalid operation name"),
                ),
            ),
            Object::Tensor => (lhs, op!["mul"]),
            _ => panic!("multiply undefined for {lhs:?}"),
        }
    }
}

impl var::HasDiv<Object, Operation> for Operation {
    fn div(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["div"])
    }
}

impl var::HasNeg<Object, Operation> for Operation {
    fn neg(operand_type: Object) -> (Object, Operation) {
        (operand_type, op!["neg"])
    }
}

pub fn pow(builder: &Builder, value: Var, exponent: Var) -> Var {
    var::fn_operation(builder, &[value, exponent], Object::Tensor, op!["pow"])
}

////////////////////////////////////////////////////////////////////////////////
// Declarations

pub fn constant_f32(builder: &Builder, v: f32) -> Var {
    let l = Operation::Literal(Literal::F32(v));
    var::fn_operation(builder, &[], Object::Tensor, l)
}

/// Pack a fixed number of Nat values into a specific shape
pub fn pack<const N: usize>(builder: &Builder, dtype: Var, xs: [Var; N]) -> Var {
    // should all be *shapes*.
    // TODO: if a nat, auto-lift to shape using Lift?
    assert_eq!(dtype.label, Object::Dtype);

    for x in &xs {
        assert_eq!(x.label, Object::Nat);
    }

    let args: Vec<Var> = std::iter::once(dtype).chain(xs).collect();
    var::fn_operation(builder, &args, Object::NdArrayType, op!["pack"])
}

/// Unpack a shape into a dtype and its constituent Nat dimensions
pub fn unpack<const N: usize>(builder: &Builder, x: Var) -> (Var, [Var; N]) {
    assert_eq!(x.label, Object::NdArrayType);

    let mut ty = vec![Object::Nat; N + 1];
    ty[0] = Object::Dtype;

    let elements = var::operation(builder, &[x], ty, op!["unpack"]);

    let mut iter = elements.into_iter();
    let head = iter.next().unwrap();
    let tail: [Var; N] = crate::util::iter_to_array(iter).expect("N elements");
    (head, tail)
}

// Tensor → NdArrayType
pub fn shape(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::NdArrayType, op!["shape"])
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

pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(builder, &[t, x], Object::Tensor, op!["reshape"])
}

/// Batch matmul
pub fn matmul(builder: &Builder, f: Var, g: Var) -> Var {
    // checked during shapechecking, but errors easier to follow here.
    assert_eq!(f.label, Object::Tensor);
    assert_eq!(g.label, Object::Tensor);

    var::fn_operation(builder, &[f, g], Object::Tensor, op!["matmul"])
}

////////////////////////////////////////////////////////////////////////////////
// S-interpretations of operations
//
macro_rules! path{
    [$($x:expr),* $(,)?] => {
        vec![$($x),*].try_into().expect("invalid operation name")
    };
}

// basic declarations
pub fn op_decls() -> std::collections::HashMap<super::path::Path, crate::category::shape::Operation>
{
    use crate::category::core::{ScalarOp::*, TensorOp::*};
    use crate::category::shape::{NatOp, Operation, TypeOp};
    use std::collections::HashMap;
    HashMap::from([
        // TODO: rename tensor.copy etc.
        (path!["op", "copy"], Operation::Tensor(Copy)),
        (path!["op", "add"], Operation::Tensor(Map(Add))),
        (path!["op", "neg"], Operation::Tensor(Map(Neg))),
        (path!["op", "mul"], Operation::Tensor(Map(Mul))),
        (path!["op", "div"], Operation::Tensor(Map(Div))),
        (path!["op", "pow"], Operation::Tensor(Map(Pow))),
        (path!["op", "matmul"], Operation::Tensor(MatMul)),
        (path!["op", "reshape"], Operation::Tensor(Reshape)),
        // shape ops
        (path!["nat", "mul"], Operation::Nat(NatOp::Mul)),
        (path!["op", "shape"], Operation::Type(TypeOp::Shape)),
        (path!["op", "pack"], Operation::Type(TypeOp::Pack)),
        (path!["op", "unpack"], Operation::Type(TypeOp::Unpack)),
        // todo
    ])
}
