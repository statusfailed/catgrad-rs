//! Catgrad reference interpreter

use crate::category::core::{NdArrayType, Shape};
use crate::ssa::SSA;

use super::ndarray::TaggedNdArray;
use crate::category::bidirectional::*;

#[derive(PartialEq, Debug, Clone)]
pub struct ApplyError {
    pub kind: ApplyErrorKind,
    pub ssa: SSA<Object, Operation>,
    pub args: Vec<Value>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum ApplyErrorKind {
    TypeError,
    MissingOperation(Path),  // Operation declaration not found in ops
    MissingDefinition(Path), // Operation definition not found in env
}

// Actual values produced by the interpreter
#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    /// A concrete natural number
    Nat(usize),

    /// A concrete dtype
    Dtype(Dtype),

    /// A concrete shape (list of natural numbers)
    Shape(Shape),

    /// A concrete NdArrayType (dtype + shape)
    Type(NdArrayType),

    /// A tensor with actual data
    NdArray(TaggedNdArray),
}
