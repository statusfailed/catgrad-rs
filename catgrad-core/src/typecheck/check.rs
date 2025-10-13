//! Abstract interpretation of [`crate::category::core`]
use super::interpreter::{Interpreter, ResultValues, Value};
use super::parameters::Parameters;
use open_hypergraphs::lax::NodeId;

use crate::abstract_interpreter::{InterpreterError, eval_with};
use crate::category::lang;
use crate::pass::to_core::Environment;

// TODO: remove?
pub type ShapeCheckResult = ResultValues;

/// Typecheck a [`lang::TypedTerm`] and return a value for each of its nodes.
pub fn check(env: &Environment, params: &Parameters, term: lang::TypedTerm) -> ShapeCheckResult {
    let lang::TypedTerm {
        term, source_type, ..
    } = term;

    check_with(env, params, term, source_type)
}

/// Typecheck a [`lang::TypedTerm`] by abstract interpretation over specified symbolic input
/// values.
pub fn check_with(
    env: &Environment,
    params: &Parameters,
    term: lang::Term,
    source_values: Vec<Value>,
) -> ShapeCheckResult {
    // TODO: can we do without this cloning params?
    let interpreter = Interpreter::new(env.clone(), params.clone());
    let term = interpreter.environment.to_core(term);

    let mut results = vec![None; term.hypergraph.nodes.len()];
    let _ = eval_with(&interpreter, term, source_values, |node_id, value| {
        results[node_id.0] = Some(value.clone())
    });
    results
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.ok_or(InterpreterError::MultipleRead(NodeId(i))))
        .collect()
}
