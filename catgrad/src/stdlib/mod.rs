//! Standard library of declarations and definitions for catgrad's `lang` category.

// Actual definitions and the `to_load_ops` helper.
mod stdlib_definitions;
pub use stdlib_definitions::*;

// The `Module` helper for creating language definitions
mod module;
pub use module::*;

// Neural networks
pub mod nn;
