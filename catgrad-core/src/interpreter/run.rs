//! Catgrad reference interpreter

use super::ndarray::NdArray;
use crate::ssa::{SSA, parallel_ssa};

use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

use crate::category::{bidirectional::*, shape};

#[derive(PartialEq, Debug, Clone)]
pub enum InterpreterError {
    NonMonogamousWrite(NodeId), // A value (identified by a node id) was written to multiple times
    NonMonogamousRead(NodeId),  // a value either didn't exist or was already used
    ApplyError(Box<ApplyError>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct ApplyError {
    pub kind: ApplyErrorKind,
    pub ssa: SSA<Object, Operation>,
    pub args: Vec<Value>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum ApplyErrorKind {
    TypeError,
    MissingOperation(Path),  // Operation declaration not found in ops
    MissingDefinition(Path), // Operation definition not found in env
}

impl From<ApplyError> for InterpreterError {
    fn from(err: ApplyError) -> Self {
        InterpreterError::ApplyError(Box::new(err))
    }
}

// Actual values produced by the interpreter
#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Dtype(Dtype),
    NdArray(NdArray),
}

pub struct Interpreter {
    ops: HashMap<Path, shape::Operation>,
    env: Environment,
}

impl Interpreter {
    // specific to this interpreter (probably?)
    pub fn new(ops: HashMap<Path, shape::Operation>, env: Environment) -> Self {
        Self { ops, env }
    }

    pub fn run(&self, term: Term, values: Vec<Value>) -> Result<Vec<Value>, InterpreterError> {
        assert_eq!(values.len(), term.sources.len());

        let mut state = HashMap::<NodeId, Value>::new();

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
        args: &[Value],
    ) -> Result<&shape::Operation, InterpreterError> {
        Ok(self.ops.get(path).ok_or(ApplyError {
            kind: ApplyErrorKind::MissingOperation(path.clone()),
            ssa: ssa.clone(),
            args: args.to_vec(),
        })?)
    }

    pub fn apply(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value>,
    ) -> Result<Vec<Value>, InterpreterError> {
        match &ssa.op {
            Operation::Literal(lit) => {
                let v = lit_to_value(lit);
                Ok(vec![v])
            }
            Operation::Declaration(path) => self.apply_declaration(ssa, args, path),
            Operation::Definition(path) => self.apply_definition(ssa, args, path),
        }
    }

    fn apply_declaration(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value>,
        path: &Path,
    ) -> Result<Vec<Value>, InterpreterError> {
        let op = self.get_op(path, ssa, &args)?;
        match op {
            shape::Operation::Type(_type_op) => todo!(),
            shape::Operation::Nat(_nat_op) => todo!(),
            shape::Operation::DtypeConstant(_dtype) => todo!(),
            shape::Operation::Tensor(_tensor_op) => todo!(),
            shape::Operation::Copy => todo!(),
        }
    }

    fn apply_definition(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value>,
        def: &Path,
    ) -> Result<Vec<Value>, InterpreterError> {
        // PERFORMANCE: does explicit recursion cost us much here?
        let definition = self.env.operations.get(def).ok_or(ApplyError {
            kind: ApplyErrorKind::MissingDefinition(def.clone()),
            ssa: ssa.clone(),
            args: args.clone(),
        })?;

        self.run(definition.term.clone(), args)
    }
}

pub(crate) fn lit_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::U32(x) => {
            let buf = x.to_ne_bytes().to_vec();
            Value::NdArray(NdArray {
                buf,
                shape: vec![],
                strides: vec![],
                offset: 0,
            })
        }
        Literal::F32(x) => {
            let buf = x.to_ne_bytes().to_vec();
            Value::NdArray(NdArray {
                buf,
                shape: vec![],
                strides: vec![],
                offset: 0,
            })
        }
        Literal::Dtype(d) => Value::Dtype(d.clone()),
    }
}
