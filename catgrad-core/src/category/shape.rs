//! A category of shape-polymorphic tensor programs.
//! Extends [`core`] with shape operations.

use super::core;
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
    /// Pack k Nats into a shape
    /// Pack : Nat^k → Type
    Pack,

    /// Split a shape into dtype and nat dimensions
    /// Unpack : Type → Nat^k
    Unpack,

    /// Get the shape of a tensor (not its dtype!)
    /// Tensor → Shape
    Shape,
}
