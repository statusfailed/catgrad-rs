//! Catgrad reference interpreter

use crate::ssa::{SSA, parallel_ssa};
use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

use crate::category::{bidirectional::*, shape};

// Tensors
#[derive(PartialEq, Debug, Clone)]
pub struct Tensor {
    // raw buffer of bytes (we'll cast this to do operations)
    pub buf: Vec<u8>,
    pub shape: Vec<usize>, // todo: use lib?
    pub strides: Vec<isize>,
    pub offset: usize,
}

#[derive(PartialEq, Debug, Clone)]
pub enum InterpreterError {
    NonMonogamousWrite(NodeId), // A value (identified by a node id) was written to multiple times
    NonMonogamousRead(NodeId),  // a value either didn't exist or was already used
    ApplyError(ApplyError, SSA<Object, Operation>, Vec<Value>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum ApplyError {
    TypeError,
}

// Actual values produced by the interpreter
#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Dtype(Dtype),
    Tensor(Tensor),
}

#[allow(dead_code)]
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

        // Create SSA
        let ssa = parallel_ssa(term.to_strict());

        // Iterate through SSA
        for par in ssa {
            // PERFORMANCE: we can do these ops in parallel. Does it get speedups?
            for op in par {
                // get args: Vec<Value> by popping each id in op.sources from state - take
                // ownership.
                let mut args = Vec::new();
                for (node_id, _) in &op.sources {
                    match state.remove(node_id) {
                        Some(value) => args.push(value),
                        None => return Err(InterpreterError::NonMonogamousWrite(*node_id)),
                    }
                }

                let results = self.apply(&op, args)?;

                // write each result into state at op.targets ids
                for ((node_id, _), result) in op.targets.iter().zip(results) {
                    if state.insert(*node_id, result).is_some() {
                        return Err(InterpreterError::NonMonogamousRead(*node_id));
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

    pub fn apply(
        &self,
        ssa: &SSA<Object, Operation>,
        args: Vec<Value>,
    ) -> Result<Vec<Value>, InterpreterError> {
        match &ssa.op {
            Operation::Literal(lit) => {
                let v = Ok(lit_to_value(lit))
                    .map_err(|e| InterpreterError::ApplyError(e, ssa.clone(), args.to_vec()))?;
                Ok(vec![v])
            }
            Operation::Declaration(_path) => todo!(),
            Operation::Definition(_path) => todo!(),
        }
    }
}

pub(crate) fn lit_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::U32(x) => {
            let buf = x.to_ne_bytes().to_vec();
            Value::Tensor(Tensor {
                buf,
                shape: vec![],
                strides: vec![],
                offset: 0,
            })
        }
        Literal::F32(x) => {
            let buf = x.to_ne_bytes().to_vec();
            Value::Tensor(Tensor {
                buf,
                shape: vec![],
                strides: vec![],
                offset: 0,
            })
        }
        Literal::Dtype(d) => Value::Dtype(d.clone()),
    }
}
