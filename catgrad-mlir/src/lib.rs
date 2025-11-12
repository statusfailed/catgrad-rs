pub mod grammar;
pub mod lower;
pub mod pass;

// rendering Declarations
pub mod ops;
pub mod util;

pub mod functor;

// TODO: replace with catgrad::definition::inline
pub mod inline;

pub mod runtime;

/// Transform MLIR text into a .so
pub mod codegen;
