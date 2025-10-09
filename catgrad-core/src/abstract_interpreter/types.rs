//! Abstract interpreter and types
use open_hypergraphs::lax::{EdgeId, NodeId};
use std::fmt::Debug;

use crate::category::core::{Object, Operation};
use crate::definition::Def;
use crate::path::Path;
use crate::ssa::{SSA, SSAError};

pub type CoreSSA = SSA<Object, Def<Path, Operation>>;

/// [`InterpreterValue`] defines a set of types which the interpreter will use to represent values
/// at runtime. For example, the `Nat` type might be...
///
/// - In the tensor backend `Nat` is a `usize`
/// - In the typechecker, `Nat` is an *expressions* over natural numbers
///
/// Each associated type must implement its corresopnding trait. So for example Nats can be added,
/// multiplied etc, while Dtypes have constants, and so on.
pub trait InterpreterValue: Clone {
    type Nat: Nat;
    type Dtype: Dtype;
    type Shape: Shape;
    type NdArrayType: NdArrayType;
    type Tensor: Tensor;

    // type ops
    fn pack(dims: Vec<Self::Nat>) -> Self::Shape;
    fn unpack(shape: Self::Shape) -> Vec<Self::Nat>;
    fn shape(tensor: Self::Tensor) -> Self::Shape;
    fn dtype(tensor: Self::Tensor) -> Self::Dtype;

    // tensor ops
    fn matmul(f: Self::Tensor, g: Self::Tensor) -> EvalResult<Self::Tensor>;
}

// TODO!
pub trait Nat: Clone + Debug {}
pub trait Dtype: Clone + Debug {}
pub trait Shape: Clone + Debug {}
pub trait NdArrayType: Clone + Debug {}
pub trait Tensor: Clone + Debug {}

/// Interpreting is the process of associating a `Value<V>` with each wire of a term
#[derive(Debug, Clone)]
pub enum Value<V: InterpreterValue> {
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
