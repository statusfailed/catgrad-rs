use catgrad_core::category::lang::*;
use catgrad_core::stdlib::*;
use catgrad_core::typecheck::*;

pub mod test_utils;
use test_utils::{get_forget_core_declarations, save_diagram_if_enabled};
pub mod test_models;
use test_models::LinearSigmoid;

#[test]
fn test_construct_linear_sigmoid() {
    let sigmoid = nn::Sigmoid.term().unwrap();
    println!("{sigmoid:?}");

    let term = LinearSigmoid.term().unwrap();
    println!("{term:?}");
}

#[test]
fn test_graph_sigmoid() {
    let term = nn::Sigmoid.term().unwrap().term;
    let term = open_hypergraphs::lax::var::forget::forget_monogamous(&term);
    save_diagram_if_enabled("test_graph_sigmoid.svg", &term);
}

#[test]
fn test_graph_linear_sigmoid() {
    let term = LinearSigmoid.term().unwrap().term;
    let term = open_hypergraphs::lax::var::forget::forget_monogamous(&term);
    save_diagram_if_enabled("test_graph_linear_sigmoid.svg", &term);
}

// Shapecheck the linear-sigmoid term.
// This should allow us to generate a diagram similar to the one in test_graph_linear_sigmoid(),
// but where objects are "symbolic shapes".
#[test]
fn test_check_linear_sigmoid() {
    run_check_test(LinearSigmoid.term(), "test_check_linear_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_sigmoid() {
    run_check_test(nn::Sigmoid.term(), "test_check_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_exp() {
    run_check_test(nn::Exp.term(), "test_check_exp.svg").expect("valid");
}

#[allow(clippy::result_large_err)]
pub fn run_check_test(
    term: Option<catgrad_core::category::lang::TypedTerm>,
    svg_filename: &str,
) -> Result<(), InterpreterError> {
    let TypedTerm {
        term, source_type, ..
    } = term.unwrap();

    let term = open_hypergraphs::lax::var::forget::forget_monogamous(&term);
    let env = get_forget_core_declarations();

    let result = check_with(&env, &Parameters::default(), term.clone(), source_type)?;
    println!("result: {result:?}");

    let typed_term = term
        .with_nodes(|_| result.into_iter().map(|e| format!("{e:?}")).collect())
        .unwrap();
    save_diagram_if_enabled(svg_filename, &typed_term);

    Ok(())
}

/*
#[test]
fn test_cyclic_definition_fails() {
    todo!()
}
*/
