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
use derive_more::From;
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
    pub body: Vec<Statement>,
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
    Bool,
    U32,
    F32,
    TensorType(TensorType),
    Tuple(Vec<Type>),
}

/// A type like `tensor<4x8xf32>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub shape: Shape,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    Unknown,
    Shape(Vec<Option<usize>>),
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::Shape(dims.into_iter().map(Some).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIdentifier {
    pub id: Identifier,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, From)]
pub enum Statement {
    Assignment(Assignment),
    Custom(String),
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

#[derive(Debug, Clone, PartialEq, Eq, From)]
pub enum Expr {
    // function call
    Call(Call),
    // operation
    Operation(Operation),
    // constant value
    Constant(Constant),
    // elementwise operation
    Elementwise(Elementwise),
    // custom MLIR expression
    Custom(String),
    // bare identifiers
    Identifier(Identifier),
}

impl Expr {
    pub fn into_assignment<T, O>(self, ssa: &catgrad::ssa::SSA<T, O>) -> Assignment {
        let result = ssa
            .targets
            .iter()
            .map(|(target_node, _)| Identifier(target_node.0))
            .collect();

        Assignment { result, expr: self }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    pub args: Vec<TypedIdentifier>,
    pub return_type: Vec<Type>, // always a tuple?
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constant {
    pub name: String,
    pub value: Option<String>,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elementwise {
    pub name: String,
    pub operands: Vec<Identifier>,
    pub ty: Type,
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
fn render_annotated_identifiers(ids: &[TypedIdentifier]) -> String {
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
            Type::Bool => write!(f, "bool"),
            Type::TensorType(tensor_type) => tensor_type.fmt(f),
            Type::U32 => write!(f, "i32"), // NOTE: mlir dtype is `i32` not `u32`
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
        match &self.shape {
            Shape::Shape(dims) if dims.is_empty() => {
                // Scalar tensor: tensor<f32>
                write!(f, "tensor<{}>", self.dtype)
            }
            _ => {
                // Non-scalar tensor: tensor<4x8xf32>
                write!(f, "tensor<{}x{}>", self.shape, self.dtype)
            }
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shape::Unknown => write!(f, "*"),
            Shape::Shape(dims) => {
                let dims_str = dims
                    .iter()
                    .map(|d| match d {
                        None => "?".to_string(),
                        Some(d) => d.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("x");
                write!(f, "{}", dims_str)
            }
        }
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

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Assignment(assignment) => assignment.fmt(f),
            Statement::Custom(content) => write!(f, "{}", content),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Call(call) => call.fmt(f),
            Expr::Operation(operation) => operation.fmt(f),
            Expr::Constant(constant) => constant.fmt(f),
            Expr::Elementwise(elementwise) => elementwise.fmt(f),
            Expr::Custom(content) => write!(f, "{}", content),
            Expr::Identifier(identifier) => identifier.fmt(f),
        }
    }
}

// E.g. `arith.negf %v0 : tensor<3x1x4xf32>` or `arith.addf %v0, %v1 : tensor<3x1x4xf32>`
impl fmt::Display for Elementwise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        for (i, operand) in self.operands.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, " {}", operand)?;
        }
        write!(f, " : {}", self.ty)
    }
}

// E.g. `arith.constant 5.0 : f32`, `tensor.empty() : tensor<4x16xf32>`, or `arith.constant false`
impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(value) = &self.value {
            write!(f, " {}", value)?;
        }
        if let Some(ty) = &self.ty {
            write!(f, " : {}", ty)?;
        }
        Ok(())
    }
}

// E.g. `func.call @id(%v3) : (index) -> (index)`
impl fmt::Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arg_names = comma_separated(&self.args.iter().map(|v| &v.id).collect::<Vec<_>>());
        let arg_types = comma_separated(&self.args.iter().map(|v| &v.ty).collect::<Vec<_>>());

        write!(f, "func.call @{}({})", self.name, arg_names)?;
        if !self.args.is_empty() {
            write!(f, " : ({})", arg_types)?;
        }
        if !self.return_type.is_empty() {
            write!(f, " -> ({})", comma_separated(&self.return_type))?;
        }
        Ok(())
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        let return_types_str = if self.return_type.len() > 1 {
            format!("({})", comma_separated(&self.return_type))
        } else {
            comma_separated(&self.return_type)
        };

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
