//! Abstract interpreter and types

use crate::ssa::SSAError;
use open_hypergraphs::lax::NodeId;

pub trait Nat {}
pub trait Dtype {}
pub trait Shape {}
pub trait NdArrayType {}
pub trait Tensor {}

pub trait InterpreterValue {
    type Nat: Nat;
    type Dtype: Dtype;
    type Shape: Shape;
    type NdArrayType: NdArrayType;
    type Tensor: Tensor;
}

/// Values used during evaluation
#[derive(Debug, Clone)]
pub enum Value<V: InterpreterValue> {
    Nat(V::Nat),
    Dtype(V::Dtype),
    Shape(V::Shape),
    Type(V::NdArrayType),
    Tensor(V::Tensor),
}

pub type EvalResult<I> = std::result::Result<Vec<Value<I>>, InterpreterError>;

pub enum InterpreterError {
    MultipleRead(NodeId),
    MultipleWrite(NodeId),
    SSAError(SSAError),
}

impl From<SSAError> for InterpreterError {
    fn from(value: SSAError) -> Self {
        InterpreterError::SSAError(value)
    }
}
