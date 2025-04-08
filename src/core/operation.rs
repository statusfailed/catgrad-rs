use open_hypergraphs::prelude::*;

use super::object::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    /// Pointwise composition of N matrices `x_i : A ⇒ B` with `y_i : B ⇒ C`
    /// for `i ∈ N`.
    ///
    MatrixMultiply {
        n: Shape,
        a: Nat,
        b: Nat,
        c: Nat,
        dtype: Dtype,
    },

    /// Const value
    Const { x: NdArrayType, k: f32 },

    /// Max value across last dimension
    Max(NdArrayType),

    /// Sum value across last dimension
    Sum(NdArrayType),

    /// Broadcast a value of shape x to one of shape n+x.
    Broadcast { n: Shape, x: NdArrayType },

    /// Reshape x
    Reshape { x: NdArrayType, shape: Shape },

    /// Transpose (swap) two dimensions of a tensor
    Transpose {
        x: NdArrayType,
        dim0: usize,
        dim1: usize,
    },
    /// Create a copy
    Copy(NdArrayType),

    /// Pointwise addition of two values of similar shapes
    Add(NdArrayType),

    /// Pointwise subtraction of two values of similar shapes
    Sub(NdArrayType),

    /// Pointwise multiplication of two values of similar shapes
    Mul(NdArrayType),

    /// Pointwise division of two values of similar shapes
    Div(NdArrayType),

    /// Pointwise raising to power of two values of similar shapes
    Pow(NdArrayType),

    /// Pointwise negation of value
    Negate(NdArrayType),

    /// Inputs injected at runtime (model parameters)
    Parameter { x: NdArrayType, name: String },
}

pub type Term = OpenHypergraph<PrimitiveType, Operation>;

impl Operation {
    /// Check an operation is *valid* - e.g., for Reshape the input and output types must be
    /// isomorphic.
    pub fn validate(self) -> Option<Self> {
        use Operation::*;
        match &self {
            Reshape { x, shape } => {
                if x.size() == shape.size() {
                    Some(self)
                } else {
                    None
                }
            }
            Transpose { x, dim0, dim1 } => {
                // Validate that dimensions are within bounds
                if *dim0 < x.shape.0.len() && *dim1 < x.shape.0.len() {
                    Some(self)
                } else {
                    None
                }
            }
            _ => Some(self),
        }
    }

    pub fn interface(&self) -> Interface {
        use Operation::*;
        match self {
            Const { x, k: _ } | Parameter { x, name: _ } => {
                let target = x.clone();
                (vec![], vec![target])
            }

            Max(x) | Sum(x) => {
                let source = x.clone();
                let target = NdArrayType {
                    shape: Shape(x.shape.0[..x.shape.0.len() - 1].to_vec()),
                    dtype: x.dtype.clone(),
                };
                (vec![source], vec![target])
            }
            MatrixMultiply { n, a, b, c, dtype } => {
                let source0 = NdArrayType {
                    shape: n + a + b,
                    dtype: dtype.clone(),
                };

                let source1 = NdArrayType {
                    shape: n + b + c,
                    dtype: dtype.clone(),
                };

                let target = NdArrayType {
                    shape: n + a + c,
                    dtype: dtype.clone(),
                };
                (vec![source0, source1], vec![target])
            }

            Broadcast { n, x } => {
                let source = x.clone();
                let target = n + x;
                (vec![source], vec![target])
            }

            Reshape { x, shape } => {
                let source = x.clone();
                let target = NdArrayType {
                    shape: shape.clone(),
                    dtype: x.dtype.clone(),
                };
                (vec![source], vec![target])
            }

            Transpose { x, dim0, dim1 } => {
                let source = x.clone();

                // Create new shape with swapped dimensions
                let mut new_shape = x.shape.0.clone();
                new_shape.swap(*dim0, *dim1);

                let target = NdArrayType {
                    shape: Shape(new_shape),
                    dtype: x.dtype.clone(),
                };

                (vec![source], vec![target])
            }

            Copy(x) => (vec![x.clone()], vec![x.clone(), x.clone()]),

            Add(x) | Sub(x) | Mul(x) | Div(x) | Pow(x) => {
                (vec![x.clone(), x.clone()], vec![x.clone()])
            }

            Negate(x) => (vec![x.clone()], vec![x.clone()]),
        }
    }

    // Make an OpenHypergraph from this operation
    pub fn term(self) -> Term {
        let (s, t) = self.interface();
        OpenHypergraph::singleton(
            self,
            SemifiniteFunction::new(VecArray(s)),
            SemifiniteFunction::new(VecArray(t)),
        )
    }
}

pub fn identity(t: Type) -> Term {
    OpenHypergraph::identity(SemifiniteFunction::new(VecArray(t)))
}
