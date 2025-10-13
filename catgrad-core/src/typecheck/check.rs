// Catgrad's shape checker is an abstract interpreter for the *core* dialect.
// It uses the core declarations in [`crate::stdlib`].

use super::interpreter::{Interpreter, ResultValues};
use crate::category::lang;
use crate::pass::to_core::Environment;

use crate::abstract_interpreter::eval;

// TODO: remove?
pub type ShapeCheckResult = ResultValues;

// TODO: return node values for *every node in the diagram!*
// TODO: add Parameters
#[allow(clippy::result_large_err)]
pub fn check(env: &Environment, term: lang::TypedTerm) -> ShapeCheckResult {
    let lang::TypedTerm {
        term, source_type, ..
    } = term;

    let interpreter = Interpreter::new();
    eval(interpreter, env.to_core(term), source_type)
}
