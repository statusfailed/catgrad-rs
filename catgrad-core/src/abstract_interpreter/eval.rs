//! Abstract interpreter and types

use super::types::*;

use crate::category::lang::Term;
use crate::prelude::{Object, Operation};
use crate::ssa::{SSA, parallel_ssa};

use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

/// Run the interpreter with specified input values
/// TODO: abstract backend/state ?
pub fn eval<V: InterpreterValue>(term: Term, values: Vec<Value<V>>) -> EvalResult<V> {
    assert_eq!(values.len(), term.sources.len());

    // create initial state by moving argument values into state
    let mut state = HashMap::<NodeId, Value<V>>::new();
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
                    None => return Err(InterpreterError::MultipleRead(*node_id)),
                }
            }

            let results = apply(&op, args)?;

            // write each result into state at op.targets ids
            for ((node_id, _), result) in op.targets.iter().zip(results) {
                if state.insert(*node_id, result).is_some() {
                    return Err(InterpreterError::MultipleWrite(*node_id));
                }
            }
        }
    }

    // Extract target values and return them
    let mut target_values = Vec::new();
    for target_node in &target_nodes {
        match state.remove(target_node) {
            Some(value) => target_values.push(value),
            None => return Err(InterpreterError::MultipleRead(*target_node)),
        }
    }

    Ok(target_values)
}

fn apply<V: InterpreterValue>(_op: &SSA<Object, Operation>, _args: Vec<Value<V>>) -> EvalResult<V> {
    todo!();
}
