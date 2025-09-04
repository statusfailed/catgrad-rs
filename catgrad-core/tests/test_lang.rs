use catgrad_core::category::lang::*;
use catgrad_core::check::*;
use catgrad_core::stdlib::{nn::*, *};
use catgrad_core::svg::to_svg;

pub mod test_utils;
use test_utils::{
    get_forget_core_declarations, replace_nodes_in_hypergraph, save_diagram_if_enabled,
};
pub mod test_models;
use test_models::LinearSigmoid;

#[test]
fn test_construct_linear_sigmoid() {
    let sigmoid = Sigmoid.term();
    println!("{sigmoid:?}");

    let term = LinearSigmoid.term();
    println!("{term:?}");
}

#[test]
fn test_graph_sigmoid() {
    let term = Sigmoid.term().term;
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_graph_sigmoid.svg", svg_bytes);
}

#[test]
fn test_graph_linear_sigmoid() {
    let term = LinearSigmoid.term().term;

    use open_hypergraphs::lax::functor::*;
    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);

    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_graph_linear_sigmoid.svg", svg_bytes);
}

// Shapecheck the linear-sigmoid term.
// This should allow us to generate a diagram similar to the one in test_graph_linear_sigmoid(),
// but where objects are "symbolic shapes".
#[test]
fn test_check_linear_sigmoid() {
    let TypedTerm {
        term, source_type, ..
    } = LinearSigmoid.term();

    run_check_test(term, source_type, "test_check_linear_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_sigmoid() {
    let TypedTerm {
        term, source_type, ..
    } = Sigmoid.term();

    run_check_test(term, source_type, "test_check_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_exp() {
    let TypedTerm {
        term, source_type, ..
    } = Exp.term();
    run_check_test(term, source_type, "test_check_exp.svg").expect("valid");
}

#[allow(clippy::result_large_err)]
pub fn run_check_test(
    term: catgrad_core::category::lang::Term,
    input_types: Vec<Value>,
    svg_filename: &str,
) -> Result<(), ShapeCheckError> {
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let (ops, env) = get_forget_core_declarations();

    let result = check_with(&ops, &env, term.clone(), input_types)?;
    println!("result: {result:?}");

    let typed_term = replace_nodes_in_hypergraph(term, result);
    let svg_bytes = to_svg(&typed_term).expect("create svg");
    save_diagram_if_enabled(svg_filename, svg_bytes);

    Ok(())
}

/*
#[test]
fn test_cyclic_definition_fails() {
    todo!()
}
*/
