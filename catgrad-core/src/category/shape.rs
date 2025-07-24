//! A category of shape-polymorphic tensor programs.
//! Extends [`core`] with shape operations.

use super::core;
use open_hypergraphs::lax::{OpenHypergraph, var};

pub use super::core::{Dtype, TensorOp};

/// Objects of the category.
/// Note that Nat and Rank-1 shapes are only isomorphic so we can safely index by naturals.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Object {
    Nat,         // natural numbers
    NdArrayType, // tuples of natural numbers (TODO: dtype)
    Tensor,
}

/// Operations are those of core, extended with operations on shapes
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Operation {
    Type(TypeOp),
    Nat(NatOp),
    Tensor(core::TensorOp),
    Copy,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NatOp {
    Constant(usize),

    // Multiply
    Mul,

    /// Lift a natural number to a rank-1 shape and dtype
    /// `Lift : Nat → Type`
    Lift(Dtype),
}

/// Operations involving shapes
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TypeOp {
    /// Concatenate n shapes
    Concat,

    /// Split a shape into a fixed number of nats
    Split,
}

// Copy lets us use HasVar
impl var::HasVar for Operation {
    fn var() -> Self {
        Operation::Copy
    }
}

pub type Var = var::Var<Object, Operation>;
pub type Term = OpenHypergraph<Object, Operation>;

use std::cell::RefCell;
use std::rc::Rc;
pub type Builder = Rc<RefCell<Term>>;

////////////////////////////////////////////////////////////////////////////////
// Helpers

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
