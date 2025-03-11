use super::object::*;

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

    /// Broadcast a value of shape x to one of shape n+x.
    Broadcast {
        n: Shape,
        x: NdArrayType,
    },

    /// Reshape a
    Reshape {
        x: NdArrayType,
        y: NdArrayType,
    },

    Copy(NdArrayType),
}

impl Operation {
    /// Check an operation is *valid* - e.g., for Reshape the input and output types must be
    /// isomorphic.
    pub fn validate(self) -> Option<Self> {
        use Operation::*;
        match &self {
            Reshape { x, y } => {
                if x.size() == y.size() {
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
                let source = NdArrayType {
                    shape: n + a + b,
                    dtype: dtype.clone(),
                };
                let target = NdArrayType {
                    shape: n + b + c,
                    dtype: dtype.clone(),
                };
                (vec![source], vec![target])
            }

            Broadcast { n, x } => {
                let source = x.clone();
                let target = n + x;
                (vec![source], vec![target])
            }

            Reshape { x, y } => (vec![x.clone()], vec![y.clone()]),

            Copy(x) => (vec![x.clone()], vec![x.clone(), x.clone()]),
        }
    }
}
