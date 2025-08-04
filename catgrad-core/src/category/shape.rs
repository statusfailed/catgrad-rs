//! A category of shape-polymorphic tensor programs.
//! Extends [`core`] with shape operations.

use super::core;
use open_hypergraphs::lax::{OpenHypergraph, var};

pub use super::core::{Dtype, TensorOp};

/// Objects of the category.
/// Note that Nat and Rank-1 shapes are only isomorphic so we can safely index by naturals.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum Object {
    Nat, // natural numbers
    Dtype,
    NdArrayType, // tuples of natural numbers (TODO: dtype)
    Tensor,
}

/// Operations are those of core, extended with operations on shapes
#[derive(Debug, PartialEq, Clone)]
pub enum Operation {
    Type(TypeOp),
    Nat(NatOp),
    DtypeConstant(Dtype),
    Tensor(core::TensorOp),
    Copy,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum NatOp {
    Constant(usize),

    // Multiply n naturals
    Mul,
}

/// Operations involving shapes
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum TypeOp {
    /// Pack a Dtype and k Nats into a shape
    /// Pack : Dtype × Nat^k → Type
    Pack,

    /// Split a shape into dtype and nat dimensions
    /// Unpack : Type → Dtype × Nat^k
    Unpack,

    /// Get the shape of a tensor
    /// Tensor → Shape
    Shape,
}

// Copy lets us use HasVar
impl var::HasVar for Operation {
    fn var() -> Self {
        Operation::Copy
    }
}

impl var::HasMul<Object, Operation> for Operation {
    fn mul(lhs_type: Object, rhs_type: Object) -> (Object, Operation) {
        assert_eq!(lhs_type, rhs_type);
        match lhs_type {
            Object::Nat => (Object::Nat, Operation::Nat(NatOp::Mul)),
            Object::Tensor => (Object::Tensor, Operation::Nat(NatOp::Mul)),
            obj => panic!("no Mul operator for Object {obj:?}"),
        }
    }
}

pub type Var = var::Var<Object, Operation>;
pub type Term = OpenHypergraph<Object, Operation>;

use std::cell::RefCell;
use std::rc::Rc;
pub type Builder = Rc<RefCell<Term>>;

////////////////////////////////////////////////////////////////////////////////
// Type Helpers

/// Unpack a shape into a dtype and its constituent Nat dimensions
pub fn unpack<const N: usize>(builder: &Builder, x: Var) -> (Var, [Var; N]) {
    assert_eq!(x.label, Object::NdArrayType);

    let mut ty = vec![Object::Nat; N + 1];
    ty[0] = Object::Dtype;

    let elements = var::operation(builder, &[x], ty, Operation::Type(TypeOp::Unpack));

    let mut iter = elements.into_iter();
    let head = iter.next().unwrap();
    let tail: [Var; N] = iter_to_array(iter).expect("N elements");
    (head, tail)
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
    var::fn_operation(
        builder,
        &args,
        Object::NdArrayType,
        Operation::Type(TypeOp::Pack),
    )
}

fn iter_to_array<T, const N: usize>(mut iter: impl Iterator<Item = T>) -> Option<[T; N]> {
    let mut vec = Vec::with_capacity(N);
    for _ in 0..N {
        vec.push(iter.next()?);
    }
    vec.try_into().ok()
}

// Tensor → NdArrayType
pub fn shape(builder: &Builder, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x],
        Object::NdArrayType,
        Operation::Type(TypeOp::Shape),
    )
}

pub fn dtype_constant(builder: &Builder, dtype: Dtype) -> Var {
    var::fn_operation(builder, &[], Object::Dtype, Operation::DtypeConstant(dtype))
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Helpers

/// Batch matmul
pub fn matmul(builder: &Builder, f: Var, g: Var) -> Var {
    // checked during shapechecking, but errors easier to follow here.
    assert_eq!(f.label, Object::Tensor);
    assert_eq!(g.label, Object::Tensor);

    var::fn_operation(
        builder,
        &[f, g],
        Object::Tensor,
        Operation::Tensor(core::TensorOp::MatMul),
    )
}

// reshape (t : Type) (x : s) (s ≅ t) : (y : t)
pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[t, x],
        Object::Tensor,
        Operation::Tensor(core::TensorOp::Reshape),
    )
}
