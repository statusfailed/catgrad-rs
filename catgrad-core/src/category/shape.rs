//! A category of shape-polymorphic tensor programs.
//! Extends [`core`] with shape operations.

use super::core;
use open_hypergraphs::lax::var;

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

impl std::fmt::Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
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
