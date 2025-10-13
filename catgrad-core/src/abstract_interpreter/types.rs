//! Type-parametric value for evaluation
use open_hypergraphs::lax::{EdgeId, NodeId};
use std::fmt::Debug;

use crate::category::{
    core,
    core::{Object, Operation, TensorOp},
};
use crate::definition::Def;
use crate::path::Path;
use crate::ssa::{SSA, SSAError};

pub type CoreSSA = SSA<Object, Def<Path, Operation>>;

/// An [`Interpreter`] defines a set of types used to represent values at runtime. For example:
///
/// - In the tensor backend `Nat` is a `usize`
/// - In the typechecker, `Nat` is an *expression* over natural numbers
///
/// In addition, functions like `pack`, `unpack`, etc. allow the interpreter to parametrise
/// the behaviour of [`eval`](super::eval::eval).
pub trait Interpreter: Clone {
    type Nat: Clone + Debug + PartialEq;
    type Dtype: Clone + Debug + PartialEq;
    type Shape: Clone + Debug + PartialEq;
    type NdArrayType: Clone + Debug + PartialEq;
    type Tensor: Clone + Debug;

    // type ops
    fn pack(dims: Vec<Self::Nat>) -> Self::Shape;
    fn unpack(shape: Self::Shape) -> Option<Vec<Self::Nat>>;
    fn shape(tensor: Self::Tensor) -> Option<Self::Shape>;
    fn dtype(tensor: Self::Tensor) -> Option<Self::Dtype>;

    // Dtype ops
    fn dtype_constant(dtype: core::Dtype) -> Self::Dtype;

    // Nat ops
    fn nat_constant(nat: usize) -> Self::Nat;
    fn nat_add(a: Self::Nat, b: Self::Nat) -> Self::Nat;
    fn nat_mul(a: Self::Nat, b: Self::Nat) -> Self::Nat;

    /// TODO: handle loads as declarations - see Issue #245
    fn handle_load(&self, ssa: &CoreSSA, path: &Path) -> Option<Vec<Value<Self>>>;

    /// Handler for Def(path) ops.
    fn handle_definition(
        &self,
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

/// Tagged value types for a given [`Interpreter`] type
#[derive(Debug, Clone)]
pub enum Value<V: Interpreter> {
    Nat(V::Nat),
    Dtype(V::Dtype),
    Shape(V::Shape),
    Type(V::NdArrayType),
    Tensor(V::Tensor),
}

impl<I: Interpreter> PartialEq for Value<I>
where
    I::Tensor: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nat(l0), Self::Nat(r0)) => l0 == r0,
            (Self::Dtype(l0), Self::Dtype(r0)) => l0 == r0,
            (Self::Shape(l0), Self::Shape(r0)) => l0 == r0,
            (Self::Type(l0), Self::Type(r0)) => l0 == r0,
            (Self::Tensor(l0), Self::Tensor(r0)) => l0 == r0,
            _ => false,
        }
    }
}

pub type EvalResult<T> = std::result::Result<T, InterpreterError>;
pub type EvalResultValues<V> = std::result::Result<Vec<Value<V>>, InterpreterError>;

#[derive(Clone, Debug)]
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
    /// Interpreter failed to handle a Load operation
    Load(EdgeId, Path),
}

impl From<SSAError> for InterpreterError {
    fn from(value: SSAError) -> Self {
        InterpreterError::SSAError(value)
    }
}
