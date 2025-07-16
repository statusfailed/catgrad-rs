//! A category of shape-polymorphic tensor programs.
//! Extends [`core`] with shape operations.

use super::core;
use open_hypergraphs::lax::{OpenHypergraph, var};

/// Objects of the category.
/// Note that Nat and Rank-1 shapes are only isomorphic so we can safely index by naturals.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Object {
    Nat,
    Shape,
    Tensor,
}

/// Operations are those of core, extended with operations on shapes
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Operation {
    Shape(ShapeOp),
    Tensor(core::TensorOp),
    Copy,
}

impl var::HasVar for Operation {
    fn var() -> Self {
        Operation::Copy
    }
}

/// Operations involving shapes
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ShapeOp {
    /// Lift a natural number to a rank-1 shape
    /// `Lift : Nat → Shape`
    Lift,
}

pub type Var = var::Var<Object, Operation>;

pub type Term = OpenHypergraph<Object, Operation>;

use std::cell::RefCell;
use std::rc::Rc;
pub type Builder = Rc<RefCell<Term>>;

////////////////////////////////////////////////////////////////////////////////
// Helpers

pub fn mat_mul(builder: &Builder, f: Var, g: Var) -> Var {
    var::fn_operation(
        builder,
        &[f, g],
        Object::Tensor,
        Operation::Tensor(core::TensorOp::MatMul),
    )
}
