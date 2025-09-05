//! Core operations on shapes, natural numbers, and tensors.
//! A simple, portable IR.

////////////////////////////////////////////////////////////////////////////////
// Basic types.

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Product of extents
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    /// Compute contiguous strides for a shape
    pub fn contiguous_strides(&self) -> Vec<isize> {
        let mut strides: Vec<isize> = vec![1];
        for dim in self.0.iter().skip(1).rev() {
            strides.push(strides.last().unwrap() * (*dim as isize));
        }
        strides.reverse();
        strides
    }
}

////////////////////////////////////////////////////////////////////////////////
// objects

/// Objects of the category.
/// Note that Nat and Rank-1 shapes are only isomorphic so we can safely index by naturals.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum Object {
    Nat, // natural numbers
    Dtype,
    NdArrayType, // tuples of natural numbers (TODO: dtype)
    Shape,
    Tensor,
}

impl std::fmt::Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

////////////////////////////////////////////////////////////////////////////////
// Operations

/// Operations are those of core, extended with operations on shapes
#[derive(Debug, PartialEq, Clone)]
pub enum Operation {
    Type(TypeOp),
    Nat(NatOp),
    DtypeConstant(Dtype),
    Tensor(TensorOp),
    Copy,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum NatOp {
    Constant(usize),

    // Multiply n naturals
    Mul,

    // Add n naturals
    Add,
}

/// Operations involving shapes
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum TypeOp {
    /// Pack k Nats into a shape
    /// Pack : Nat^k → Type
    Pack,

    /// Split a shape into nat dimensions
    /// Unpack : Type → Nat^k
    Unpack,

    /// Get the shape of a tensor (not its dtype!)
    /// Tensor → Shape
    Shape,

    /// Get the dtype of a tensor (not its dtype!)
    /// Tensor → Shape
    Dtype,
}

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

    /// Cast a tensor to a dtype
    /// `Tensor × Dtype → Tensor`
    Cast,

    /// Batch matrix multiplication
    /// `MatMul : (N, A, B) ● (N, B, C) → (N, A, C)`
    MatMul,

    /// A tensor with shape () having a single value.
    Constant(Constant),

    /// Sum last dimension of a tensor
    /// `Sum : Tensor → Tensor`
    Sum,

    /// Max last dimension of a tensor
    /// `Max : Tensor → Tensor`
    Max,

    /// Argmax last dimension of a tensor
    /// `Argmax : Tensor → Tensor`
    Argmax,

    // broadcast a Tensor of shape S to one of shape (N × S) (prepends shape N to tensor)
    Broadcast,

    /// Reshape a tensor into an isomorphic shape
    Reshape,

    // TODO:
    // Parameters
    //Parameter(String)
    //Index
    //Slice
    //Concat
    //Arange
    /// S ● ... ● S → N×S
    Stack,

    /// N×S → S ● ... ● S
    Split,

    // Array lookup indices
    // `Index: (N,) ● (M,) → (N,)`
    Index,

    // Copy a tensor
    Copy,
}

/// For now, we assume that every Dtype defines a ring & has comparisons
/// TODO: constants, comparisons
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum ScalarOp {
    Add, // 2 → 1
    Mul, // 2 → 1
    Div, // 2 → 1
    Neg, // 1 → 1
    Pow, // 2 → 1
    LT,  // 2 → 1
    EQ,  // 2 → 1
}

impl ScalarOp {
    /// The *profile* of an operation is the pair of its arity and coarity.
    pub fn profile(&self) -> (usize, usize) {
        match self {
            ScalarOp::Add => (2, 1),
            ScalarOp::Mul => (2, 1),
            ScalarOp::Div => (2, 1),
            ScalarOp::Neg => (1, 1),
            ScalarOp::Pow => (2, 1),
            ScalarOp::LT => (2, 1),
            ScalarOp::EQ => (2, 1),
        }
    }
}
