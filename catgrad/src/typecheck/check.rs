//! Abstract interpretation of [`crate::category::core`]
use super::interpreter::{Interpreter, ResultValues, Value};
use super::parameters::Parameters;
use open_hypergraphs::lax::NodeId;

use crate::abstract_interpreter::{InterpreterError, eval_with};
use crate::category::lang;
use crate::pass::to_core::Environment;

/// Typecheck a [`lang::TypedTerm`] and return a symbolic value for each of its nodes.
pub fn check(env: &Environment, params: &Parameters, term: lang::TypedTerm) -> ResultValues {
    let lang::TypedTerm {
        term, source_type, ..
    } = term;

    check_with(env, params, term, source_type)
}

/// Typecheck a [`lang::TypedTerm`] with a specified input type
pub fn check_with(
    env: &Environment,
    params: &Parameters,
    term: lang::Term,
    source_values: Vec<Value>,
) -> ResultValues {
    // TODO: can we do without this cloning params?
    let interpreter = Interpreter::new(env.clone(), params.clone());
    let term = interpreter.environment.to_core(term);

    // The `eval` function just returns output values, but we need all intermediate nodes in the
    // graph, so we use eval_with and the `on_write` function which lets us collect all the writes
    // to all nodes in the graph.
    let mut results = vec![None; term.hypergraph.nodes.len()];
    let _ = eval_with(&interpreter, term, source_values, |node_id, value| {
        results[node_id.0] = Some(value.clone())
    });

    // if we didn't get an output value, give a MultipleRead error (meaning 0 reads where 1 was
    // expected)
    results
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.ok_or(InterpreterError::MultipleRead(NodeId(i))))
        .collect()
}
