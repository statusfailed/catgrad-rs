//! Catgrad reference interpreter

use crate::category::{core, lang::*};
use crate::ssa::{SSA, parallel_ssa};
use crate::stdlib::{Declarations, Environment};

use super::backend::*;
use super::types::*;

use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

pub struct Interpreter<B: Backend> {
    pub backend: B,
    pub ops: Declarations,
    pub env: Environment,
}

impl<B: Backend> Interpreter<B> {
    // specific to this interpreter (probably?)
    pub fn new(backend: B, ops: Declarations, env: Environment) -> Self {
        Self { backend, ops, env }
    }

    /// Run the interpreter with specified input values
    pub fn run(
        &self,
        term: Term,
        values: Vec<Value<B>>,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        assert_eq!(values.len(), term.sources.len());

        // create initial state by moving argument values into state
        let mut state = HashMap::<NodeId, Value<B>>::new();
        for (node_id, value) in term.sources.iter().zip(values) {
            state.insert(*node_id, value);
        }

        // Save target nodes before moving term
        let target_nodes = term.targets.clone();

        // Iterate through partially-ordered SSA ops
        for par in parallel_ssa(term.to_strict()) {
            // PERFORMANCE: we can do these ops in parallel. Does it get speedups?
            for op in par {
                // get args: Vec<Value> by popping each id in op.sources from state - take
                // ownership.
                let mut args = Vec::new();
                for (node_id, _) in &op.sources {
                    match state.remove(node_id) {
                        Some(value) => args.push(value),
                        None => return Err(InterpreterError::NonMonogamousRead(*node_id)),
                    }
                }

                let results = self.apply(&op, args)?;

                // write each result into state at op.targets ids
                for ((node_id, _), result) in op.targets.iter().zip(results) {
                    if state.insert(*node_id, result).is_some() {
                        return Err(InterpreterError::NonMonogamousWrite(*node_id));
                    }
                }
            }
        }

        // Extract target values and return them
        let mut target_values = Vec::new();
        for target_node in &target_nodes {
            match state.remove(target_node) {
                Some(value) => target_values.push(value),
                None => return Err(InterpreterError::NonMonogamousRead(*target_node)),
            }
        }

        Ok(target_values)
    }

    fn get_op(
        &self,
        path: &Path,
        ssa: &SSA<Object, Operation>,
    ) -> Result<&core::Operation, InterpreterError> {
        Ok(self.ops.operations.get(path).ok_or(ApplyError {
            kind: ApplyErrorKind::MissingOperation(path.clone()),
            ssa: ssa.clone(),
        })?)
    }

    pub fn apply(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value<B>>,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        match &ssa.op {
            Operation::Literal(lit) => {
                let v = lit_to_value(&self.backend, lit);
                Ok(vec![v])
            }
            Operation::Declaration(path) => self.apply_declaration(ssa, args, path),
            Operation::Definition(path) => self.apply_definition(ssa, args, path),
        }
    }

    fn apply_declaration(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value<B>>,
        path: &Path,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        let op = self.get_op(path, ssa)?;
        use super::shape_op::{apply_dtype_constant, apply_nat_op, apply_type_op};
        use super::tensor_op::apply_tensor_op;
        Ok(match op {
            core::Operation::Type(type_op) => apply_type_op(type_op, args, ssa)?,
            core::Operation::Nat(nat_op) => apply_nat_op(nat_op, args, ssa)?,
            core::Operation::DtypeConstant(dtype) => apply_dtype_constant(dtype, args, ssa)?,
            core::Operation::Tensor(tensor_op) => {
                apply_tensor_op(&self.backend, tensor_op, args, ssa)?
            }
            core::Operation::Copy => apply_copy(args, ssa)?,
        })
    }

    fn apply_definition(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value<B>>,
        def: &Path,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        // PERFORMANCE: does explicit recursion cost us much here?
        let definition = self.env.operations.get(def).ok_or(ApplyError {
            kind: ApplyErrorKind::MissingDefinition(def.clone()),
            ssa: ssa.clone(),
        })?;

        self.run(definition.term.clone(), args)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum InterpreterError {
    /// A value (identified by a node id) was written to multiple times
    NonMonogamousWrite(NodeId),

    /// a value either didn't exist or was already used
    NonMonogamousRead(NodeId),

    /// An error during application of an operation
    ApplyError(Box<ApplyError>),
}

impl From<ApplyError> for InterpreterError {
    fn from(err: ApplyError) -> Self {
        InterpreterError::ApplyError(Box::new(err))
    }
}

impl From<Box<ApplyError>> for InterpreterError {
    fn from(err: Box<ApplyError>) -> Self {
        InterpreterError::ApplyError(err)
    }
}

pub(crate) fn lit_to_value<B: Backend>(backend: &B, lit: &Literal) -> Value<B> {
    match lit {
        Literal::F32(x) => Value::NdArray(TaggedNdArray::F32([B::scalar(backend, *x)])),
        Literal::U32(x) => Value::NdArray(TaggedNdArray::U32([B::scalar(backend, *x)])),
        Literal::Dtype(d) => Value::Dtype(d.clone()),
    }
}

fn apply_copy<B: Backend>(
    mut args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    use super::shape_op::expect_arity;
    expect_arity(&args, 1, ssa)?;
    let n = ssa.targets.len();
    for _ in 1..n {
        args.push(args[0].clone());
    }
    Ok(args)
}
