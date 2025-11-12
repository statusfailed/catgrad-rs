/// Lower [`catgrad::TypedTerm`]s to MLIR text
pub mod lower;

/// Transform MLIR text into a .so
pub mod codegen;

/// Execution of compiled shared objects
pub mod runtime;
