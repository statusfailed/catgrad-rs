pub(crate) mod interpreter;
pub(crate) mod isomorphism;
pub(crate) mod tensor_op;

// public interface: value types and check function
pub mod check;
pub mod value_types;

pub use check::check;
pub use value_types::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr};
