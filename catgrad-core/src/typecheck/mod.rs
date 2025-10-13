pub(crate) mod interpreter;
pub(crate) mod isomorphism;
pub(crate) mod tensor_op;

pub mod display;

// public interface: value types and check function
pub mod check;
pub mod parameters;
pub mod value_types;

pub use crate::abstract_interpreter::Value;
pub use check::{check, check_with};
pub use parameters::Parameters;
pub use value_types::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr};

pub type TypecheckError = crate::abstract_interpreter::InterpreterError;
