use catgrad::prelude::*;

use super::grammar;
use super::lower_term;

// TODO: PERFORMANCE: this is inefficient when called multiple times: inlining should be done "bottom-up" in
// topological dependency order so we don't repeat work.
// FIX: pass a *vec* of names, then inline the whole environment bottom-up, and emit MLIR for each
// of the passed names.
/// Render a TypedTerm and its associated environment of definitions to MLIR
pub fn lang_to_mlir(
    env: &Environment,
    params: &typecheck::Parameters,
    path: Path,
) -> grammar::Func {
    let mut typed_term = env.definitions.get(&path).unwrap().clone();

    // Inline all operations
    // TODO: use catgrad's inline; refactor lang to use Def?
    let defs = env
        .definitions
        .iter()
        .map(|(path, typed_term)| (path.clone(), typed_term.term.clone()))
        .collect();
    let term = super::inline::inline(defs, typed_term.term);

    // Forget extraneous Copy operations
    typed_term.term = open_hypergraphs::lax::var::forget::forget_monogamous(&term);

    // TODO: fix use of unwrap()
    // Typecheck `term` and get an open hypergraph annotated with *normalized* types
    let node_annotations: Vec<_> = typecheck::check(env, params, typed_term.clone())
        .unwrap()
        .into_iter()
        .map(typecheck::normalize)
        .collect();

    // Verify no free Dtype variables (which cannot be compiled in MLIR) by checking there were no
    // non-constant Dtypes in the entrypoint term's node annotations
    // TODO: replace assertion with a Result
    assert!(node_annotations.iter().all(is_dtype_monomorphic));

    // Create a term with normalized types for node labels
    let checked_term = typed_term.term.with_nodes(|_| node_annotations).unwrap();

    // Map type-level ops to identities at runtime
    let checked_term = super::functor::forget_identity_casts(&checked_term);

    // Forget all copy ops for MLIR (it's harder to deal with)
    let checked_term = open_hypergraphs::lax::var::forget::forget(&checked_term);

    // Convert term to MLIR
    let mlir = lower_term::term_to_func(&path.to_string(), checked_term);

    // TODO: Produce MLIR for each used dependency: a list of MLIR fragments
    // let _definitions = todo!(); // list of MLIR strings / structs

    // TODO: Return fully-rendered MLIR, including:
    //  - top level term (entrypoint?)
    //  - each used dependency
    //let result = todo!("concat _entry_fragment and _definitions");

    mlir
}

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
