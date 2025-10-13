//! # Abstract Interpreter
//!
//! An interpreter for [`crate::category::core::Term`] which is parametric over the underlying type
//! representation.
//! This allows implementation of both typechecking and evaluation with [`eval`](eval::eval).
pub mod types;
pub use types::*;

pub mod util;

pub mod eval;
pub use eval::{eval, eval_with};
