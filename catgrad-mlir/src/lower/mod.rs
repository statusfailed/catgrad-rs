//! # Lowering to MLIR text
//!
//! This module provides a single function: [`lang_to_mlir`], which lowers a [`catgrad::prelude::TypedTerm`] to
//! MLIR text, returning a [`Func`]
//!
//! # Details
//!
//! Catgrad types are lowered to MLIR types as follows:
//!
//! - `Nat` -> `index`
//! - `Dtype` -> `bool` (erased)
//! - `Shape` -> `Tensor` (NOTE: this needs to change; we need a real Shape runtime rep.)
//! - `Tensor` -> `tensor`
//! - `NdArrayType` -> error (no runtime rep - TODO: remove case!)
//!
//! Here, `bool` is used as a dummy ('erased') type; Dtype has no runtime representation in the
//! MLIR backend.

/// Simplified MLIR grammar for rendering
mod grammar;

// Preprocess
mod preprocess;

// Rendering of ops to MLIR fragments
mod emit_mlir;

mod ops;
mod util;

/// Forget operations which amount to identities (like identity casts)
mod functor;

/// Factorise models into parameters & compute
mod factor;
#[cfg(test)]
mod test_factor;

// TODO: replace with catgrad::definition::inline
// Inline all definitions within a top-level term
mod inline;

pub use emit_mlir::term_to_func;
pub use factor::factor;
pub use grammar::Func;
pub use preprocess::preprocess;
