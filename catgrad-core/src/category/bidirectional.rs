//! This is the primary interface for users to construct programs.
use open_hypergraphs::lax::*;
use std::collections::HashMap;
use std::fmt;

use crate::category::{core, shape};

////////////////////////////////////////////////////////////////////////////////
// Macros

macro_rules! op {
    [$($x:expr),* $(,)?] => {
        Operation::Declaration(vec!["op", $($x),*].try_into().expect("invalid operation name"))
    };
}

////////////////////////////////////////////////////////////////////////////////
// Operation names

// Names of definitions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PathComponent(String); // only [a-zA-Z_]

impl fmt::Display for PathComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Path(Vec<PathComponent>);

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let components: Vec<String> = self.0.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", components.join("."))
    }
}

impl TryFrom<String> for PathComponent {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.chars().all(|c| c.is_alphanumeric() || c == '_') {
            Ok(PathComponent(value))
        } else {
            Err(format!(
                "PathComponent must only contain alphanumeric characters and underscores, got: {value}"
            ))
        }
    }
}

impl TryFrom<Vec<&str>> for Path {
    type Error = String;

    fn try_from(value: Vec<&str>) -> Result<Self, Self::Error> {
        let components: Result<Vec<PathComponent>, String> = value
            .into_iter()
            .map(|s| s.to_string().try_into())
            .collect();
        components.map(Path)
    }
}

pub fn path(components: Vec<&str>) -> Path {
    components.try_into().expect("invalid path")
}

////////////////////////////////////////////////////////////////////////////////
// Operations

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    F32(f32),
    I32(i32),
    Dtype(core::Dtype),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::F32(v) => write!(f, "{v}"),
            Literal::I32(v) => write!(f, "{v}"),
            Literal::Dtype(dtype) => write!(f, "{dtype:?}"),
        }
    }
}

// Operations are the core shape operations (and operation schemas like 'Constant') extended
// with "Definitions".
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Operations declared to exist, but without a definition (externally interpreted)
    Declaration(Path),

    /// Extended set of base operations with definitions
    Definition(Path),

    /// Literals (floats, ints, dtypes, nats, etc.) are elements (maps with unit domain)
    Literal(Literal),
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Declaration(path) => write!(f, "{path}"),
            Operation::Definition(path) => write!(f, "{path}"),
            Operation::Literal(literal) => write!(f, "{literal}"),
        }
    }
}

// The actual data of a definition: a term, its type, and ...?
pub struct OperationDefinition {
    pub term: Term,
    pub source_type: Term,
    pub target_type: Term,
}

// The set of operations in the category
pub struct Environment {
    pub operations: HashMap<Path, OperationDefinition>,
}

// TODO: use Object from Shape cat
pub type Object = super::shape::Object;

pub type Term = OpenHypergraph<Object, Operation>;

pub type Var = open_hypergraphs::lax::var::Var<Object, Operation>;

use std::cell::RefCell;
use std::rc::Rc;
pub type Builder = Rc<RefCell<Term>>;

////////////////////////////////////////////////////////////////////////////////
// Rust helpers for declarations

// Copy lets us use HasVar
impl var::HasVar for Operation {
    fn var() -> Self {
        op!["copy"]
    }
}

impl var::HasAdd<Object, Operation> for Operation {
    fn add(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["add"])
    }
}

impl var::HasMul<Object, Operation> for Operation {
    fn mul(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["mul"])
    }
}

impl var::HasDiv<Object, Operation> for Operation {
    fn div(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["div"])
    }
}

impl var::HasNeg<Object, Operation> for Operation {
    fn neg(operand_type: Object) -> (Object, Operation) {
        (operand_type, op!["neg"])
    }
}

pub fn pow(builder: &Builder, value: Var, exponent: Var) -> Var {
    var::fn_operation(builder, &[value, exponent], Object::Tensor, op!["pow"])
}

////////////////////////////////////////////////////////////////////////////////
// Declarations

// TODO: definition for exp
pub fn exp(builder: &Builder, x: Var) -> Var {
    let e = constant_f32(builder, std::f32::consts::E);
    pow(builder, e, x)
}

pub fn constant_f32(builder: &Builder, v: f32) -> Var {
    let l = Operation::Literal(Literal::F32(v));
    var::fn_operation(builder, &[], Object::Tensor, l)
}

////////////////////////////////////////////////////////////////////////////////
// Operation helpers

/// Pack a fixed number of Nat values into a specific shape
pub fn pack<const N: usize>(builder: &Builder, dtype: Var, xs: [Var; N]) -> Var {
    // should all be *shapes*.
    // TODO: if a nat, auto-lift to shape using Lift?
    assert_eq!(dtype.label, Object::Dtype);

    for x in &xs {
        assert_eq!(x.label, Object::Nat);
    }

    let args: Vec<Var> = std::iter::once(dtype).chain(xs).collect();
    var::fn_operation(builder, &args, Object::NdArrayType, op!["pack"])
}

/// Unpack a shape into a dtype and its constituent Nat dimensions
pub fn unpack<const N: usize>(builder: &Builder, x: Var) -> (Var, [Var; N]) {
    assert_eq!(x.label, Object::NdArrayType);

    let mut ty = vec![Object::Nat; N + 1];
    ty[0] = Object::Dtype;

    let elements = var::operation(builder, &[x], ty, op!["unpack"]);

    let mut iter = elements.into_iter();
    let head = iter.next().unwrap();
    let tail: [Var; N] = crate::util::iter_to_array(iter).expect("N elements");
    (head, tail)
}

// Tensor → NdArrayType
pub fn shape(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::NdArrayType, op!["shape"])
}

pub fn dtype_constant(builder: &Builder, dtype: shape::Dtype) -> Var {
    var::fn_operation(
        builder,
        &[],
        Object::Dtype,
        Operation::Literal(Literal::Dtype(dtype)),
    )
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Helpers

pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(builder, &[t, x], Object::Tensor, op!["reshape"])
}

/// Batch matmul
pub fn matmul(builder: &Builder, f: Var, g: Var) -> Var {
    // checked during shapechecking, but errors easier to follow here.
    assert_eq!(f.label, Object::Tensor);
    assert_eq!(g.label, Object::Tensor);

    var::fn_operation(builder, &[f, g], Object::Tensor, op!["matmul"])
}
