use super::Parameters;
use super::value_types::*;

use crate::category::{core, core::TensorOp, lang};
use crate::{
    abstract_interpreter,
    abstract_interpreter::{CoreSSA, eval},
};

use crate::pass::to_core::Environment;

use super::tensor_op::tensor_op;

pub type Value = abstract_interpreter::Value<Interpreter>;
pub type ResultValues = abstract_interpreter::EvalResultValues<Interpreter>;

#[derive(Clone, std::fmt::Debug)]
pub struct Interpreter {
    pub(crate) environment: Environment,
    pub(crate) parameters: Parameters,
}

impl Interpreter {
    pub(crate) fn new(environment: Environment, parameters: Parameters) -> Self {
        Interpreter {
            environment,
            parameters,
        }
    }

    pub fn check_with(&self, term: core::Term, source_values: Vec<Value>) -> ResultValues {
        eval(self, term, source_values)
    }
}

impl abstract_interpreter::Interpreter for Interpreter {
    type Nat = NatExpr;
    type Dtype = DtypeExpr;
    type Shape = ShapeExpr;
    type NdArrayType = TypeExpr;
    type Tensor = TypeExpr;

    fn pack(dims: Vec<Self::Nat>) -> Self::Shape {
        ShapeExpr::Shape(dims)
    }

    fn unpack(shape: Self::Shape) -> Option<Vec<Self::Nat>> {
        match shape {
            ShapeExpr::Var(_) => None,
            ShapeExpr::OfType(_) => None,
            ShapeExpr::Shape(nat_exprs) => Some(nat_exprs),
        }
    }

    fn shape(tensor: Self::Tensor) -> Option<Self::Shape> {
        match tensor {
            TypeExpr::Var(_) => None,
            TypeExpr::NdArrayType(nd_array_type) => Some(nd_array_type.shape),
        }
    }

    fn dtype(tensor: Self::Tensor) -> Option<Self::Dtype> {
        match tensor {
            TypeExpr::Var(_) => None,
            TypeExpr::NdArrayType(nd_array_type) => Some(nd_array_type.dtype),
        }
    }

    fn dtype_constant(d: core::Dtype) -> Self::Dtype {
        DtypeExpr::Constant(d)
    }

    fn nat_constant(nat: usize) -> Self::Nat {
        NatExpr::Constant(nat)
    }

    fn nat_add(a: Self::Nat, b: Self::Nat) -> Self::Nat {
        NatExpr::Add(vec![a, b])
    }

    fn nat_mul(a: Self::Nat, b: Self::Nat) -> Self::Nat {
        NatExpr::Mul(vec![a, b])
    }

    fn handle_load(&self, _ssa: &CoreSSA, path: &crate::prelude::Path) -> Option<Vec<Value>> {
        self.parameters.0.get(path).map(|t| vec![t.clone()])
    }

    fn handle_definition(
        &self,
        _ssa: &CoreSSA,
        args: Vec<abstract_interpreter::Value<Self>>,
        path: &crate::prelude::Path,
    ) -> abstract_interpreter::EvalResultValues<Self> {
        let source_values = args.to_vec();
        let lang::TypedTerm { term, .. } = self
            .environment
            .definitions
            .get(path)
            .unwrap_or_else(|| panic!("definition {path} not found"));
        // TODO: can we remove this clone?
        let term = self.environment.to_core(term.clone());
        self.check_with(term, source_values)
    }

    fn tensor_op(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value>,
        op: &TensorOp,
    ) -> abstract_interpreter::EvalResultValues<Self> {
        tensor_op(ssa, args, op)
    }
}
