//! A category of shape-polymorphic tensor programs.
//! Extends [`core`] with shape operations.

use super::core;
use open_hypergraphs::lax::{OpenHypergraph, var};

/// Objects of the category.
/// Note that Nat and Rank-1 shapes are only isomorphic so we can safely index by naturals.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Object {
    Nat,   // natural numbers
    Shape, // tuples of natural numbers (TODO: dtype)
    Tensor,
}

/// Operations are those of core, extended with operations on shapes
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Operation {
    Shape(ShapeOp),
    Tensor(core::TensorOp),
    Copy,
}

/// Operations involving shapes
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ShapeOp {
    /// Concatenate two shapes
    Concat,

    /// Split a shape into a fixed number of nats
    Split,

    /// Lift a natural number to a rank-1 shape
    /// `Lift : Nat → Shape`
    Lift,

    // TODO: *semiring* ops on nats?
    /// Multiply two nats
    Mul,

    // TODO: docs
    // `Tensor → Type × Tensor` and its inverse
    Annotate,
    Coannotate,

    // Syntactic equality(?)
    Equal,
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

// [x, t] → [x : t]
#[allow(dead_code)]
fn annotate(builder: &Builder, x: Var, t: Var) -> Var {
    assert_eq!(x.label, Object::Tensor);
    assert_eq!(x.label, Object::Shape); // TODO: should be NdArrayType, not just shape

    var::fn_operation(
        builder,
        &[x, t],
        Object::Tensor,
        Operation::Shape(ShapeOp::Annotate),
    )
}

// (x : t) → [x, t]
#[allow(dead_code)]
fn coannotate(builder: &Builder, x: Var) -> (Var, Var) {
    assert_eq!(x.label, Object::Tensor);

    let result = var::operation(
        builder,
        &[x],
        vec![Object::Tensor, Object::Shape],
        Operation::Shape(ShapeOp::Coannotate),
    );

    assert_eq!(result.len(), 2);
    (result[0].clone(), result[1].clone())
}

#[allow(dead_code)]
fn split<const N: usize>(builder: &Builder, x: Var) -> [Var; N] {
    assert_eq!(x.label, Object::Shape);

    let result = var::operation(
        builder,
        &[x],
        vec![Object::Shape; N],
        Operation::Shape(ShapeOp::Coannotate),
    );

    result.try_into().expect("unexpected size!")
}

#[allow(dead_code)]
fn concat<const N: usize>(builder: &Builder, xs: [Var; N]) -> Var {
    // should all be *shapes*.
    // TODO: if a nat, auto-lift to shape using Lift?
    for x in &xs {
        assert_eq!(x.label, Object::Shape);
    }

    var::fn_operation(
        builder,
        &xs,
        Object::Shape,
        Operation::Shape(ShapeOp::Coannotate),
    )
}

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

// reshape (t : Shape) (x : s) (s ≅ t) : (y : t)
pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[t, x],
        Object::Tensor,
        Operation::Tensor(core::TensorOp::Reshape),
    )
}
