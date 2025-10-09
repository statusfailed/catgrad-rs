//! Abstract interpreter and types
use open_hypergraphs::lax::{EdgeId, NodeId};
use std::fmt::Debug;

use crate::category::core::{Object, Operation, TensorOp};
use crate::definition::Def;
use crate::path::Path;
use crate::ssa::{SSA, SSAError};

pub type CoreSSA = SSA<Object, Def<Path, Operation>>;

/// [`ValueTypes`] defines a set of types which the interpreter will use to represent values
/// at runtime. For example:
///
/// - In the tensor backend `Nat` is a `usize`
/// - In the typechecker, `Nat` is an *expression* over natural numbers
///
/// Each associated type must implement its corresopnding trait. So for example Nats can be added,
/// multiplied etc, while Dtypes have constants, and so on.
pub trait ValueTypes: Clone {
    type Nat: Clone + Debug;
    type Dtype: Clone + Debug;
    type Shape: Clone + Debug;
    type NdArrayType: Clone + Debug;
    type Tensor: Clone + Debug;

    // type ops
    fn pack(dims: Vec<Self::Nat>) -> Self::Shape;
    fn unpack(shape: Self::Shape) -> Option<Vec<Self::Nat>>;
    fn shape(tensor: Self::Tensor) -> Option<Self::Shape>;
    fn dtype(tensor: Self::Tensor) -> Option<Self::Dtype>;

    /// Handler for Def(path) ops.
    fn handle_definition(
        ssa: &CoreSSA,
        args: Vec<Value<Self>>,
        path: &Path,
    ) -> EvalResultValues<Self>;

    // tensor ops are very backend-specific, so we let the interpreter handle them directly
    // TODO: rename handle_tensor_op
    fn tensor_op(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value<Self>>,
        op: &TensorOp,
    ) -> EvalResultValues<Self>;
}

/// Interpreting is the process of associating a `Value<V>` with each wire of a term
#[derive(Debug, Clone)]
pub enum Value<V: ValueTypes> {
    Nat(V::Nat),
    Dtype(V::Dtype),
    Shape(V::Shape),
    Type(V::NdArrayType),
    Tensor(V::Tensor),
}

//pub type EvalResult<V> = std::result::Result<Vec<Value<V>>, InterpreterError>;
pub type EvalResult<T> = std::result::Result<T, InterpreterError>;
pub type EvalResultValues<V> = std::result::Result<Vec<Value<V>>, InterpreterError>;

pub enum InterpreterError {
    /// A node appeared as a *source* of multiple hyperedges, and so interpreting tried to read a
    /// value that had already been consumed.
    MultipleRead(NodeId),
    /// A node appeared as a *target* of multiple hyperedges, and so was written to multiple times
    /// during interpretation.
    MultipleWrite(NodeId),
    /// A term could not be mapped into SSA form
    SSAError(SSAError),
    /// Could not apply an operation because arguments were not of the correct form.
    TypeError(EdgeId),
    /// Unexpected number of arguments to an operation
    ArityError(EdgeId),
}

impl From<SSAError> for InterpreterError {
    fn from(value: SSAError) -> Self {
        InterpreterError::SSAError(value)
    }
}
