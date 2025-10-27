//! Types representing a subset of MLIR grammar.
//!
//! Sufficient to represent simple functions in SSA form, e.g.:
//!
//! ```mlir
//!  func.func @matmul_chain(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<4x16xf32> {
//!    %0 = tensor.empty() : tensor<4x16xf32>
//!    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%0 : tensor<4x16xf32>) -> tensor<4x16xf32>
//!    %2 = tensor.empty() : tensor<4x16xf32>
//!    %3 = linalg.matmul ins(%1, %arg2 : tensor<4x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<4x16xf32>) -> tensor<4x16xf32>
//!    return %3 : tensor<4x16xf32>
//!  }
//! ```
use std::fmt;

/// A top-level MLIR function, for example:
///
/// ```mlir
/// func.func <name>(<parameters>) -> <return_type> {
///     <body>
///     <return_stmt>
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Func {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Vec<Type>,
    pub body: Vec<Assignment>,
    pub return_stmt: Return,
}

/// A single parameter to a function
///
/// ```mlir
/// <name>: <param_type>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
}

/// MLIR type annotation. Either `index` or a `TensorType`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Index,
    U32,
    F32,
    TensorType(TensorType),
    Tuple(Vec<Type>),
}

/// A type like `tensor<4x8xf32>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIdentifier {
    pub id: Identifier,
    pub ty: Type,
}

/// An assignment of expression to variable name, e.g.
///
/// ```mlir
/// %0 = tensor.empty() : tensor<4x16xf32>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub result: Vec<Identifier>,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    // function call
    Call(Call),
    // operation
    Operation(Operation),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub id: Identifier,
    pub args: Vec<TypedIdentifier>,
    pub return_type: Vec<Type>, // always a tuple?
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    pub name: String,
    pub ins: Vec<TypedIdentifier>,
    pub outs: Vec<TypedIdentifier>,
    pub return_types: Vec<Type>,
    pub attrs: Option<String>,
    pub inner_block: Option<String>,
}

/// A return statement, e.g.
///
/// ```mlir
/// return %3 : tensor<4x16xf32>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Return(pub Vec<TypedIdentifier>);

////////////////////////////////////////////////////////////////////////////////
// Display helpers

// Helper to create comma-separated lists from items with Display/ToString
fn comma_separated<T: ToString>(items: &[T]) -> String {
    items
        .iter()
        .map(|item| item.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// Render a list of AnnotatedIdentifier as `{id_0}, {id_1}, ... : {ty_0}, {ty_1}, ...`
fn render_annotated_identifiers(ids: &Vec<TypedIdentifier>) -> String {
    let id_names = comma_separated(&ids.iter().map(|v| &v.id).collect::<Vec<_>>());
    let id_types = comma_separated(&ids.iter().map(|v| &v.ty).collect::<Vec<_>>());
    format!("{} : {}", id_names, id_types)
}

////////////////////////////////////////////////////////////////////////////////
// Display instances to render fragments as text

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Index => write!(f, "index"),
            Type::TensorType(tensor_type) => tensor_type.fmt(f),
            Type::U32 => write!(f, "u32"),
            Type::F32 => write!(f, "f32"),
            Type::Tuple(types) => {
                write!(f, "(")?;
                write!(f, "{}", comma_separated(types))?;
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tensor<{}x{}>",
            self.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.dtype
        )
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

impl fmt::Display for TypedIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.id, self.ty)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Call(call) => call.fmt(f),
            Expr::Operation(operation) => operation.fmt(f),
        }
    }
}

impl fmt::Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.id, render_annotated_identifiers(&self.args))?;
        if !self.return_type.is_empty() {
            write!(f, " -> {}", comma_separated(&self.return_type))?;
        }
        Ok(())
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Special case for tensor.empty() - it uses : instead of ->
        if self.name == "tensor.empty()" && !self.return_types.is_empty() {
            write!(f, "{} : {}", self.name, self.return_types[0])?;
            return Ok(());
        }

        write!(f, "{}", self.name)?;

        // For linalg.generic, attributes go in curly braces right after the operation name
        if self.name == "linalg.generic" {
            if let Some(attrs) = &self.attrs {
                write!(f, " {{{}}}", attrs)?;
            }
        }

        // Format as: ins(%v0, %v1 : type1, type2)
        if !self.ins.is_empty() {
            write!(f, " ins({})", render_annotated_identifiers(&self.ins))?;
        }

        // Format as: outs(%v0, %v1 : type1, type2)
        if !self.outs.is_empty() {
            write!(f, " outs({})", render_annotated_identifiers(&self.outs))?;
        }

        // TODO: hack alert! we check the name of the Operation here.
        // For non-linalg.generic operations, attributes go after ins/outs
        if self.name != "linalg.generic" {
            if let Some(attrs) = &self.attrs {
                write!(f, " {}", attrs)?;
            }
        }

        // Inner block comes before return type
        if let Some(inner_block) = &self.inner_block {
            write!(f, " {{{}}}", inner_block)?;
        }

        if !self.return_types.is_empty() {
            write!(f, " -> {}", comma_separated(&self.return_types))?;
        }

        Ok(())
    }
}

impl fmt::Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lhs = comma_separated(&self.result);
        write!(f, "{} = {}", lhs, self.expr)
    }
}

impl fmt::Display for Return {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, "return")
        } else {
            write!(f, "return {}", render_annotated_identifiers(&self.0))
        }
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{} : {}", self.name, self.param_type)
    }
}

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params_str = comma_separated(&self.parameters);
        let return_types_str = comma_separated(&self.return_type);

        writeln!(
            f,
            "func.func @{}({}) -> {} {{",
            self.name, params_str, return_types_str
        )?;

        for assignment in &self.body {
            writeln!(f, "  {}", assignment)?;
        }

        writeln!(f, "  {}", self.return_stmt)?;
        write!(f, "}}")
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // Check we can reconstruct the example in the docstring
    #[test]
    fn test_matmul_chain_example() {
        let func = Func {
            name: "matmul_chain".to_string(),
            parameters: vec![
                Parameter {
                    name: "arg0".to_string(),
                    param_type: Type::TensorType(TensorType {
                        shape: vec![4, 8],
                        dtype: "f32".to_string(),
                    }),
                },
                Parameter {
                    name: "arg1".to_string(),
                    param_type: Type::TensorType(TensorType {
                        shape: vec![8, 16],
                        dtype: "f32".to_string(),
                    }),
                },
                Parameter {
                    name: "arg2".to_string(),
                    param_type: Type::TensorType(TensorType {
                        shape: vec![16, 16],
                        dtype: "f32".to_string(),
                    }),
                },
            ],
            return_type: vec![Type::TensorType(TensorType {
                shape: vec![4, 16],
                dtype: "f32".to_string(),
            })],
            body: vec![
                Assignment {
                    result: vec![Identifier(0)],
                    expr: Expr::Operation(Operation {
                        name: "tensor.empty()".to_string(),
                        ins: vec![],
                        outs: vec![],
                        return_types: vec![Type::TensorType(TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        })],
                        attrs: None,
                        inner_block: None,
                    }),
                },
                Assignment {
                    result: vec![Identifier(1)],
                    expr: Expr::Operation(Operation {
                        name: "linalg.matmul".to_string(),
                        ins: vec![
                            TypedIdentifier {
                                id: Identifier(0),
                                ty: Type::TensorType(TensorType {
                                    shape: vec![4, 8],
                                    dtype: "f32".to_string(),
                                }),
                            },
                            TypedIdentifier {
                                id: Identifier(1),
                                ty: Type::TensorType(TensorType {
                                    shape: vec![8, 16],
                                    dtype: "f32".to_string(),
                                }),
                            },
                        ],
                        outs: vec![TypedIdentifier {
                            id: Identifier(0),
                            ty: Type::TensorType(TensorType {
                                shape: vec![4, 16],
                                dtype: "f32".to_string(),
                            }),
                        }],
                        return_types: vec![Type::TensorType(TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        })],
                        attrs: None,
                        inner_block: None,
                    }),
                },
                Assignment {
                    result: vec![Identifier(2)],
                    expr: Expr::Operation(Operation {
                        name: "tensor.empty()".to_string(),
                        ins: vec![],
                        outs: vec![],
                        return_types: vec![Type::TensorType(TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        })],
                        attrs: None,
                        inner_block: None,
                    }),
                },
                Assignment {
                    result: vec![Identifier(3)],
                    expr: Expr::Operation(Operation {
                        name: "linalg.matmul".to_string(),
                        ins: vec![
                            TypedIdentifier {
                                id: Identifier(1),
                                ty: Type::TensorType(TensorType {
                                    shape: vec![4, 16],
                                    dtype: "f32".to_string(),
                                }),
                            },
                            TypedIdentifier {
                                id: Identifier(2),
                                ty: Type::TensorType(TensorType {
                                    shape: vec![16, 16],
                                    dtype: "f32".to_string(),
                                }),
                            },
                        ],
                        outs: vec![TypedIdentifier {
                            id: Identifier(2),
                            ty: Type::TensorType(TensorType {
                                shape: vec![4, 16],
                                dtype: "f32".to_string(),
                            }),
                        }],
                        return_types: vec![Type::TensorType(TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        })],
                        attrs: None,
                        inner_block: None,
                    }),
                },
            ],
            return_stmt: Return(vec![TypedIdentifier {
                id: Identifier(3),
                ty: Type::TensorType(TensorType {
                    shape: vec![4, 16],
                    dtype: "f32".to_string(),
                }),
            }]),
        };

        println!("{}", func);

        let output = func.to_string();
        assert!(output.contains("func.func @matmul_chain"));
        assert!(output.contains("tensor.empty() : tensor<4x16xf32>"));
        assert!(output.contains("linalg.matmul"));
        assert!(output.contains("return %v3 : tensor<4x16xf32>"));
    }
}
