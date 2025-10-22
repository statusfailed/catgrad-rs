use catgrad::category::lang;
use catgrad::prelude::*;
use open_hypergraphs::lax::OpenHypergraph;

// TODO
pub struct MLIR;

// Render a TypedTerm and its associated environment of definitions to MLIR
pub fn lang_to_mlir(env: Environment, params: typecheck::Parameters, term: TypedTerm) -> MLIR {
    // Typecheck `term` and get an open hypergraph annotated with types
    let node_annotations = typecheck::check(&env, &params, term.clone()).unwrap();
    let checked_term = term.term.clone().with_nodes(|_| node_annotations).unwrap();

    // Produce MLIR from SSA, ignore dependencies
    let entry_fragment = checked_term_to_mlir(checked_term);

    // TODO: Produce MLIR for each used dependency: a list of MLIR fragments
    // let _definitions = todo!(); // list of MLIR strings / structs

    // TODO: Return fully-rendered MLIR, including:
    //  - top level term (entrypoint?)
    //  - each used dependency
    //let result = todo!("concat _entry_fragment and _definitions");

    entry_fragment
}

fn checked_term_to_mlir(_term: OpenHypergraph<typecheck::Type, lang::Operation>) -> MLIR {
    todo!()
}
