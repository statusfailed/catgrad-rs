use catgrad::prelude::*;

use super::grammar;
use super::lower;

// Render a TypedTerm and its associated environment of definitions to MLIR
pub fn lang_to_mlir(
    env: &Environment,
    params: &typecheck::Parameters,
    mut typed_term: TypedTerm,
) -> Vec<grammar::Func> {
    // Forget extraneous Copy operations
    typed_term.term = open_hypergraphs::lax::var::forget::forget_monogamous(&typed_term.term);

    // Typecheck `term` and get an open hypergraph annotated with *normalized* types
    let node_annotations = typecheck::check(env, params, typed_term.clone())
        .unwrap()
        .into_iter()
        .map(typecheck::normalize)
        .collect();

    // Create a term with normalized types for node labels
    let checked_term = typed_term
        .term
        .clone()
        .with_nodes(|_| node_annotations)
        .unwrap();

    // Convert term to MLIR
    let mlir = lower::term_to_func("term", checked_term);

    // TODO: Produce MLIR for each used dependency: a list of MLIR fragments
    // let _definitions = todo!(); // list of MLIR strings / structs

    // TODO: Return fully-rendered MLIR, including:
    //  - top level term (entrypoint?)
    //  - each used dependency
    //let result = todo!("concat _entry_fragment and _definitions");

    vec![mlir]
}
