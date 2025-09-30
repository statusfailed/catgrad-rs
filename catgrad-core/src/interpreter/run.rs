//! Catgrad reference interpreter

use super::backend::*;
use super::types::*;
use crate::category::{core, core::*, lang};
use crate::definition::Def;
use crate::pass::to_core::{env_to_core, to_core};
use crate::path::Path;
use crate::ssa::{SSAError, parallel_ssa};

use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

/// Parameter values dict
#[derive(Clone, Debug)]
pub struct Parameters<B: Backend>(HashMap<Path, TaggedNdArray<B>>);

#[derive(Clone, Debug)]
pub struct Environment {
    pub definitions: HashMap<Path, core::Term>,
}

// Needed so Backend doesn't have to implement Default
impl<B: Backend> Default for Parameters<B> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<B: Backend> From<HashMap<Path, TaggedNdArray<B>>> for Parameters<B> {
    fn from(map: HashMap<Path, TaggedNdArray<B>>) -> Self {
        Parameters(map)
    }
}

impl<B: Backend, const N: usize> From<[(Path, TaggedNdArray<B>); N]> for Parameters<B> {
    fn from(arr: [(Path, TaggedNdArray<B>); N]) -> Self {
        Parameters(HashMap::from(arr))
    }
}

pub struct Interpreter<B: Backend> {
    /// Array kernel backend implementation
    pub backend: B,
    /// Environment (definitions & declarations)
    pub env: Environment,
    /// Parameter tensors
    pub params: Parameters<B>,
}

impl<B: Backend> Interpreter<B> {
    // specific to this interpreter (probably?)
    pub fn from_core(backend: B, env: Environment, params: Parameters<B>) -> Self {
        Self {
            backend,
            env,
            params,
        }
    }

    pub fn new(backend: B, env: crate::stdlib::Environment, params: Parameters<B>) -> Self {
        let env = env_to_core(env);
        Self::from_core(backend, env, params)
    }

    pub fn run(
        &self,
        term: lang::Term,
        values: Vec<Value<B>>,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        self.run_core(to_core(term), values)
    }

    /// Run the interpreter with specified input values
    pub fn run_core(
        &self,
        term: core::Term,
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
        for par in parallel_ssa(term.to_strict())? {
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

    pub fn apply(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value<B>>,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        match &ssa.op {
            Def::Def(path) => self.apply_definition(ssa, args, path),
            Def::Arr(op) => self.apply_op(ssa, args, op),
        }
    }

    // Dispatch ops
    fn apply_op(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value<B>>,
        op: &Operation,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        use super::shape_op::{apply_dtype_constant, apply_nat_op, apply_type_op};
        use super::tensor_op::apply_tensor_op;
        Ok(match op {
            core::Operation::Load(path) => self.load(path)?,
            core::Operation::Type(type_op) => apply_type_op(type_op, args, ssa)?,
            core::Operation::Nat(nat_op) => apply_nat_op(nat_op, args, ssa)?,
            core::Operation::DtypeConstant(dtype) => apply_dtype_constant(dtype, args, ssa)?,
            core::Operation::Tensor(tensor_op) => {
                apply_tensor_op(&self.backend, tensor_op, args, ssa)?
            }
            core::Operation::Copy => apply_copy(args, ssa)?,
        })
    }

    // Dispatch definitions by recursing
    fn apply_definition(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value<B>>,
        def: &Path,
    ) -> Result<Vec<Value<B>>, InterpreterError> {
        // PERFORMANCE: does explicit recursion cost us much here?
        let definition = self.env.definitions.get(def).ok_or(ApplyError {
            kind: ApplyErrorKind::MissingDefinition(def.clone()),
            ssa: ssa.clone(),
        })?;

        // PERFORMANCE: we shouldn't really need to clone terms here;
        // Interpreter only takes ownership because it has to call to_strict() to do the SSA
        // decomposition. But this is not strictly necessary.
        self.run_core(definition.clone(), args)
    }

    fn load(&self, path: &Path) -> Result<Vec<Value<B>>, InterpreterError> {
        let value = self.params.0.get(path);
        let value = value.ok_or(InterpreterError::LoadError(path.clone()))?;
        // TODO: fix unnecessary clone (or ensure backend deals with this!)
        Ok(vec![Value::NdArray(value.clone())])
    }
}

#[derive(Debug, Clone)]
pub enum InterpreterError {
    /// A value (identified by a node id) was written to multiple times
    NonMonogamousWrite(NodeId),

    /// a value either didn't exist or was already used
    NonMonogamousRead(NodeId),

    /// An error during application of an operation
    ApplyError(Box<ApplyError>),

    /// SSA Conversion error
    SSAError(SSAError),

    /// No such parameter
    LoadError(Path),
}

impl From<SSAError> for InterpreterError {
    fn from(err: SSAError) -> Self {
        InterpreterError::SSAError(err)
    }
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

fn apply_copy<B: Backend>(
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    use super::shape_op::expect_arity;
    expect_arity(&args, 1, ssa)?;
    let n = ssa.targets.len();
    for _ in 1..n {
        args.push(args[0].clone());
    }
    Ok(args)
}
