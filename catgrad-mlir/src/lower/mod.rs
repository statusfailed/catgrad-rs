//! # Lowering to MLIR
//!
//! This module provides a single function: [`lang_to_mlir`], which lowers a catgrad TypedTerm to
//! MLIR text, returning a [`Func`]

/// Simplified MLIR grammar for rendering
mod grammar;

// Top-level interface to the compiler pass
mod pass;

// Rendering of ops to MLIR fragments
mod lower_term;
mod ops;
mod util;

/// Forget operations which amount to identities (like identity casts)
mod functor;

// TODO: replace with catgrad::definition::inline
// Inline all definitions within a top-level term
mod inline;

pub use grammar::Func;
pub use pass::lang_to_mlir;
