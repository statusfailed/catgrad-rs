#![doc = include_str!("../../README.md")]
pub mod category;
pub mod stdlib;

// Shapechecking & Evaluation
pub mod check;
pub mod interpreter;
pub mod ssa;

// Utilities
#[cfg(feature = "svg")]
pub mod svg;
pub mod util;

// entry point
pub mod prelude;
