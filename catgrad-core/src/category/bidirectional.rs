//! This is the primary interface for users to construct programs.
use open_hypergraphs::lax::*;
use std::collections::HashMap;

use crate::category::{core, shape};

////////////////////////////////////////////////////////////////////////////////
// Macros

macro_rules! path {
    [$($x:expr),* $(,)?] => {
        Operation::Definition(vec![$($x),*].try_into().expect("invalid operation name"))
    };
}

////////////////////////////////////////////////////////////////////////////////
// Operation names

// Names of definitions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PathComponent(String); // only [a-zA-Z_]

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Path(Vec<PathComponent>);

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

// Operations are the core shape operations (and operation schemas like 'Constant') extended
// with "Definitions".
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Definition(Path),
    Operation(super::shape::Operation),
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
        path!["std", "copy"]
    }
}

impl var::HasAdd<Object, Operation> for Operation {
    fn add(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, path!["std", "add"])
    }
}

impl var::HasMul<Object, Operation> for Operation {
    fn mul(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, path!["std", "add"])
    }
}

impl var::HasDiv<Object, Operation> for Operation {
    fn div(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, path!["std", "div"])
    }
}

impl var::HasNeg<Object, Operation> for Operation {
    fn neg(operand_type: Object) -> (Object, Operation) {
        (operand_type, path!["std", "negate"])
    }
}

////////////////////////////////////////////////////////////////////////////////
// Declarations

pub fn copy_term(_env: &Environment) -> Term {
    todo!();
}

pub fn exp_term(_env: &Environment) -> Term {
    todo!();
}

pub fn exp(_x: Var) -> Var {
    todo!()
}

pub fn constant_f32(builder: &Builder, v: f32) -> Var {
    var::fn_operation(
        builder,
        &[],
        Object::Tensor,
        Operation::Operation(shape::Operation::Tensor(core::TensorOp::ConstantF32(v))),
    )
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
    var::fn_operation(
        builder,
        &args,
        Object::NdArrayType,
        Operation::Operation(shape::Operation::Type(shape::TypeOp::Pack)),
    )
}

/// Unpack a shape into a dtype and its constituent Nat dimensions
pub fn unpack<const N: usize>(builder: &Builder, x: Var) -> (Var, [Var; N]) {
    assert_eq!(x.label, Object::NdArrayType);

    let mut ty = vec![Object::Nat; N + 1];
    ty[0] = Object::Dtype;

    let elements = var::operation(
        builder,
        &[x],
        ty,
        Operation::Operation(shape::Operation::Type(shape::TypeOp::Unpack)),
    );

    let mut iter = elements.into_iter();
    let head = iter.next().unwrap();
    let tail: [Var; N] = crate::util::iter_to_array(iter).expect("N elements");
    (head, tail)
}

// Tensor → NdArrayType
pub fn shape(builder: &Builder, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x],
        Object::NdArrayType,
        Operation::Operation(shape::Operation::Type(shape::TypeOp::Shape)),
    )
}

pub fn dtype_constant(builder: &Builder, dtype: shape::Dtype) -> Var {
    var::fn_operation(
        builder,
        &[],
        Object::Dtype,
        Operation::Operation(shape::Operation::DtypeConstant(dtype)),
    )
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Helpers

pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[t, x],
        Object::Tensor,
        Operation::Operation(shape::Operation::Tensor(core::TensorOp::Reshape)),
    )
}

/// Batch matmul
pub fn matmul(builder: &Builder, f: Var, g: Var) -> Var {
    // checked during shapechecking, but errors easier to follow here.
    assert_eq!(f.label, Object::Tensor);
    assert_eq!(g.label, Object::Tensor);

    var::fn_operation(
        builder,
        &[f, g],
        Object::Tensor,
        Operation::Operation(shape::Operation::Tensor(core::TensorOp::MatMul)),
    )
}
