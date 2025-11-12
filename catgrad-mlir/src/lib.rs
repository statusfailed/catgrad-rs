//! # Catgrad MLIR backend

pub mod lower;

/// Transform MLIR text into a shared object file
pub mod codegen;

/// Execution of compiled shared objects
pub mod runtime;
