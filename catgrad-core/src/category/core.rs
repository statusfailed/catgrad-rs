//! Core operations on shapes, natural numbers, and tensors.

////////////////////////////////////////////////////////////////////////////////
// Basic types.
// TODO: move these to interpreter?

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
    Add, // 2 → 1
    Mul, // 2 → 1
    Div, // 2 → 1
    Neg, // 1 → 1
    Pow, // 2 → 1
}
