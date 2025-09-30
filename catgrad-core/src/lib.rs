#![doc = include_str!("../../README.md")]
pub mod category;
pub mod stdlib;

// path::Path is a type of dot-separated strings used to name definitions and lang ops.
pub mod path;

// Shapechecking & Evaluation
pub mod check;
pub mod interpreter;
pub mod ssa;

// general compiler tools
pub mod definition;

// Utilities
#[cfg(feature = "svg")]
pub mod svg;
pub mod util;

// entry point
pub mod prelude;
