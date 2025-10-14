//! # Catgrad Typechecker
//!
//! Evaluates the compute graph using the symbolic types in [`value_types`] to ensure the graph is
//! well-typed.
//! The [`check`](check::check) function returns an assignment of these symbolic [`Value`]s to each node in the
//! input term.
pub(crate) mod interpreter;
pub(crate) mod isomorphism;
pub(crate) mod tensor_op;

pub mod display;

pub mod check;
pub mod parameters;
pub mod value_types;

// public interface: value types and check function
pub use crate::abstract_interpreter::{InterpreterError, Value};
pub use check::{check, check_with};
pub use parameters::Parameters;
pub use value_types::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr};
