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
    Const(f32),

    /// Max value across last dimension
    Max,

    /// Sum value across last dimension
    Sum,

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
    Copy,

    /// Pointwise addition of two values of similar shapes
    Add,

    /// Pointwise subtraction of two values of similar shapes
    Sub,

    /// Pointwise multiplication of two values of similar shapes
    Mul,

    /// Pointwise division of two values of similar shapes
    Div,

    /// Pointwise raising to power of two values of similar shapes
    Pow,

    /// Pointwise negation of value
    Negate,

    /// Inputs injected at runtime (model parameters)
    Parameter(String),
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

            _ => panic!("Not implemented"),
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

    // Make an OpenHypergraph for the Copy operation
    pub fn copy(x: NdArrayType) -> Term {
        OpenHypergraph::singleton(
            Operation::Copy,
            SemifiniteFunction::new(VecArray(vec![x.clone()])),
            SemifiniteFunction::new(VecArray(vec![x.clone(), x.clone()])),
        )
    }

    // Make an OpenHypergraph for the given operation
    fn reduceop(x: NdArrayType, op: Operation) -> Term {
        let source = x.clone();
        let target = NdArrayType {
            shape: Shape(x.shape.0[..x.shape.0.len() - 1].to_vec()),
            dtype: x.dtype.clone(),
        };
        OpenHypergraph::singleton(
            op,
            SemifiniteFunction::new(VecArray(vec![source])),
            SemifiniteFunction::new(VecArray(vec![target])),
        )
    }

    // Make an OpenHypergraph for the given operation
    fn unop(x: NdArrayType, op: Operation) -> Term {
        OpenHypergraph::singleton(
            op,
            SemifiniteFunction::new(VecArray(vec![x.clone()])),
            SemifiniteFunction::new(VecArray(vec![x.clone()])),
        )
    }

    // Make an OpenHypergraph for the Parameter operation
    pub fn parameter(x: NdArrayType, name: &str) -> Term {
        OpenHypergraph::singleton(
            Operation::Parameter(name.to_string()),
            SemifiniteFunction::new(VecArray(vec![])),
            SemifiniteFunction::new(VecArray(vec![x.clone()])),
        )
    }

    // Make an OpenHypergraph for the Const operation
    pub fn constop(x: NdArrayType, k: f32) -> Term {
        OpenHypergraph::singleton(
            Operation::Const(k),
            SemifiniteFunction::new(VecArray(vec![])),
            SemifiniteFunction::new(VecArray(vec![x.clone()])),
        )
    }

    // Make an OpenHypergraph for the Negate operation
    pub fn negate(x: NdArrayType) -> Term {
        Operation::unop(x, Operation::Negate)
    }

    // Make an OpenHypergraph for the given binary operation
    fn binop(x: NdArrayType, op: Operation) -> Term {
        OpenHypergraph::singleton(
            op,
            SemifiniteFunction::new(VecArray(vec![x.clone(), x.clone()])),
            SemifiniteFunction::new(VecArray(vec![x.clone()])),
        )
    }

    // Make an OpenHypergraph for the Add operation
    pub fn add(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Add)
    }

    // Make an OpenHypergraph for the Sub operation
    pub fn sub(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Sub)
    }

    // Make an OpenHypergraph for the Mul operation
    pub fn mul(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Mul)
    }

    // Make an OpenHypergraph for the Div operation
    pub fn div(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Div)
    }

    // Make an OpenHypergraph for the Pow operation
    pub fn pow(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Pow)
    }

    // Make an OpenHypergraph for a Sum operation
    pub fn sum(x: NdArrayType) -> Term {
        Operation::reduceop(x, Operation::Sum)
    }

    // Make an OpenHypergraph for a Max operation
    pub fn max(x: NdArrayType) -> Term {
        Operation::reduceop(x, Operation::Max)
    }
}

pub fn identity(t: Type) -> Term {
    OpenHypergraph::identity(SemifiniteFunction::new(VecArray(t)))
}
