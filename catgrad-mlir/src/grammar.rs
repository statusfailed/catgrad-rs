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
    pub return_type: TensorType,
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
    pub param_type: TensorType,
}

/// MLIR type annotation. Either `index` or a `TensorType`.
pub enum Type {
    Index,
    TensorType(TensorType),
}

/// A type like `tensor<4x8xf32>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueWithType {
    pub value: Value,
    pub value_type: Option<TensorType>,
}

/// An assignment of expression to variable name, e.g.
///
/// ```mlir
/// %0 = tensor.empty() : tensor<4x16xf32>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub result: Value,
    pub operation: Operation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    pub name: String,
    pub ins: Vec<ValueWithType>,
    pub outs: Vec<ValueWithType>,
    pub return_types: Vec<TensorType>,
    pub attrs: Option<String>,
    pub inner_block: Option<String>,
}

/// A return statement, e.g.
///
/// ```mlir
/// return %3 : tensor<4x16xf32>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Return {
    pub value: Value,
    pub value_type: TensorType,
}

////////////////////////////////////////////////////////////////////////////////
// Display instances to render fragments as text

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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.name)
    }
}

impl fmt::Display for ValueWithType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value_type {
            Some(t) => write!(f, "{} : {}", self.value, t),
            None => write!(f, "{}", self.value),
        }
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

        if !self.ins.is_empty() {
            // Format as: ins(%v0, %v1 : type1, type2)
            let var_names = self
                .ins
                .iter()
                .map(|v| v.value.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let types = self
                .ins
                .iter()
                .filter_map(|v| v.value_type.as_ref())
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, " ins({} : {})", var_names, types)?;
        }

        if !self.outs.is_empty() {
            // Format as: outs(%v0 : type0)
            let var_names = self
                .outs
                .iter()
                .map(|v| v.value.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let types = self
                .outs
                .iter()
                .filter_map(|v| v.value_type.as_ref())
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, " outs({} : {})", var_names, types)?;
        }

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
            let ret_str = self
                .return_types
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, " -> {}", ret_str)?;
        }

        Ok(())
    }
}

impl fmt::Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.result, self.operation)
    }
}

impl fmt::Display for Return {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "return {} : {}", self.value, self.value_type)
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{} : {}", self.name, self.param_type)
    }
}

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params_str = self
            .parameters
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        writeln!(
            f,
            "func.func @{}({}) -> {} {{",
            self.name, params_str, self.return_type
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
                    param_type: TensorType {
                        shape: vec![4, 8],
                        dtype: "f32".to_string(),
                    },
                },
                Parameter {
                    name: "arg1".to_string(),
                    param_type: TensorType {
                        shape: vec![8, 16],
                        dtype: "f32".to_string(),
                    },
                },
                Parameter {
                    name: "arg2".to_string(),
                    param_type: TensorType {
                        shape: vec![16, 16],
                        dtype: "f32".to_string(),
                    },
                },
            ],
            return_type: TensorType {
                shape: vec![4, 16],
                dtype: "f32".to_string(),
            },
            body: vec![
                Assignment {
                    result: Value {
                        name: "0".to_string(),
                    },
                    operation: Operation {
                        name: "tensor.empty()".to_string(),
                        ins: vec![],
                        outs: vec![],
                        return_types: vec![TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        }],
                        attrs: None,
                        inner_block: None,
                    },
                },
                Assignment {
                    result: Value {
                        name: "1".to_string(),
                    },
                    operation: Operation {
                        name: "linalg.matmul".to_string(),
                        ins: vec![
                            ValueWithType {
                                value: Value {
                                    name: "arg0".to_string(),
                                },
                                value_type: Some(TensorType {
                                    shape: vec![4, 8],
                                    dtype: "f32".to_string(),
                                }),
                            },
                            ValueWithType {
                                value: Value {
                                    name: "arg1".to_string(),
                                },
                                value_type: Some(TensorType {
                                    shape: vec![8, 16],
                                    dtype: "f32".to_string(),
                                }),
                            },
                        ],
                        outs: vec![ValueWithType {
                            value: Value {
                                name: "0".to_string(),
                            },
                            value_type: Some(TensorType {
                                shape: vec![4, 16],
                                dtype: "f32".to_string(),
                            }),
                        }],
                        return_types: vec![TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        }],
                        attrs: None,
                        inner_block: None,
                    },
                },
                Assignment {
                    result: Value {
                        name: "2".to_string(),
                    },
                    operation: Operation {
                        name: "tensor.empty()".to_string(),
                        ins: vec![],
                        outs: vec![],
                        return_types: vec![TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        }],
                        attrs: None,
                        inner_block: None,
                    },
                },
                Assignment {
                    result: Value {
                        name: "3".to_string(),
                    },
                    operation: Operation {
                        name: "linalg.matmul".to_string(),
                        ins: vec![
                            ValueWithType {
                                value: Value {
                                    name: "1".to_string(),
                                },
                                value_type: Some(TensorType {
                                    shape: vec![4, 16],
                                    dtype: "f32".to_string(),
                                }),
                            },
                            ValueWithType {
                                value: Value {
                                    name: "arg2".to_string(),
                                },
                                value_type: Some(TensorType {
                                    shape: vec![16, 16],
                                    dtype: "f32".to_string(),
                                }),
                            },
                        ],
                        outs: vec![ValueWithType {
                            value: Value {
                                name: "2".to_string(),
                            },
                            value_type: Some(TensorType {
                                shape: vec![4, 16],
                                dtype: "f32".to_string(),
                            }),
                        }],
                        return_types: vec![TensorType {
                            shape: vec![4, 16],
                            dtype: "f32".to_string(),
                        }],
                        attrs: None,
                        inner_block: None,
                    },
                },
            ],
            return_stmt: Return {
                value: Value {
                    name: "3".to_string(),
                },
                value_type: TensorType {
                    shape: vec![4, 16],
                    dtype: "f32".to_string(),
                },
            },
        };

        println!("{}", func);

        let output = func.to_string();
        assert!(output.contains("func.func @matmul_chain"));
        assert!(output.contains("tensor.empty() : tensor<4x16xf32>"));
        assert!(output.contains("linalg.matmul"));
        assert!(output.contains("return %3 : tensor<4x16xf32>"));
    }
}
