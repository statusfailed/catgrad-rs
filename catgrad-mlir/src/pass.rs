use catgrad::category::lang;
use catgrad::prelude::*;
use open_hypergraphs::lax::*;

use super::grammar;
use super::lower;

/// Ensure the boundaries of the entry point term do not have free Dtype vars
fn is_dtype_monomorphic(ty: &Type) -> bool {
    match &ty {
        Type::Dtype(d) => match d {
            typecheck::DtypeExpr::Constant(_) => true,
            // Non-constant dtypes not permitted
            _ => false,
        },
        _ => true,
    }
}

// Render a TypedTerm and its associated environment of definitions to MLIR
pub fn lang_to_mlir(
    env: &Environment,
    params: &typecheck::Parameters,
    mut typed_term: TypedTerm,
) -> Vec<grammar::Func> {
    // Verify no free Dtype variables (which cannot be compiled in MLIR)

    // Forget extraneous Copy operations
    typed_term.term = open_hypergraphs::lax::var::forget::forget_monogamous(&typed_term.term);

    // Typecheck `term` and get an open hypergraph annotated with *normalized* types
    let node_annotations: Vec<_> = typecheck::check(env, params, typed_term.clone())
        .unwrap()
        .into_iter()
        .map(typecheck::normalize)
        .collect();

    // Verify there were no non-constant Dtypes in the entrypoint term's node annotations
    // TODO: replace assertion with a Result
    assert!(node_annotations.iter().all(is_dtype_monomorphic));

    // Create a term with normalized types for node labels
    let checked_term = typed_term
        .term
        .clone()
        .with_nodes(|_| node_annotations)
        .unwrap();

    // Forget all copy ops for MLIR (it's harder to deal with)
    let checked_term = open_hypergraphs::lax::var::forget::forget(&checked_term);

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
