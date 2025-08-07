// Core: a category of tensor programs with static shapes.
use open_hypergraphs::lax::OpenHypergraph;

/// Generating objects in Core
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdArrayType {
    pub dtype: Dtype,
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    U32,
}

pub type Shape = Vec<usize>;

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    F32(f32),
    U32(i32),
}

/// Generating tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    /// Lift a scalar operation `f : m → n` to `m` input and `n` output arrays.
    /// `Map_f : S₀ ● ..m.. ● S_m → S₀ ● ..n.. ● Sn`
    Map(ScalarOp),

    /// Reduce a tensor one one dimension using binary operation which is assumed to be associative
    /// `Reduce (.., N, ..) → (.., 1, ..)`
    Reduce(ScalarOp, i8),

    /// Constant(i) : [] → [()]
    Constant(Constant),

    /// S ● ... ● S → N×S
    Stack,

    /// N×S → S ● ... ● S
    Split,

    /// Reshape one operation into another
    Reshape,

    /// Batch matrix multiplication
    /// `MatMul : (N, A, B) ● (N, B, C) → (N, A, C)`
    MatMul,

    // Array lookup indices
    // `Index: (N,) ● (M,) → (N,)`
    Index,

    // broadcast a tensor by appending a shape
    // Broadcast : (x : S) ● (T: Shape) → (y : T×S)
    Broadcast,

    // Copy a tensor
    Copy,
}

/// For now, we assume that every Dtype defines a ring & has comparisons
/// TODO: constants, comparisons
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum ScalarOp {
    Add,
    Mul,
    Div,
    Neg,
    Pow,
    Zero,
    One,
}

pub type Object = NdArrayType;
pub type Arrow = TensorOp;
pub type Term = OpenHypergraph<NdArrayType, TensorOp>;
