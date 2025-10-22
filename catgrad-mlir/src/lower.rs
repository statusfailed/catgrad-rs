//! Lower a catgrad::category::lang::Term to MLIR
use catgrad::category::lang;
use catgrad::prelude::*;
use open_hypergraphs::lax::OpenHypergraph;

use crate::grammar;

pub type Term = OpenHypergraph<typecheck::Type, lang::Operation>;

/// Lower a [`Term`] to MLIR text, giving it the specified name.
pub fn term_to_func(name: &str, term: &Term) -> grammar::Func {
    let parameters = get_mlir_parameters(term);
    let (return_stmt, return_type) = get_mlir_returns(term);
    let body = get_mlir_body(term);

    grammar::Func {
        name: name.to_string(),
        parameters,
        return_type,
        body,
        return_stmt,
    }
}

/// Convert source nodes of a `Term` to MLIR function parameters
fn get_mlir_parameters(term: &Term) -> Vec<grammar::Parameter> {
    term.sources
        .iter()
        .map(|&source_id| {
            let source_type = &term.hypergraph.nodes[source_id.0];
            grammar::Parameter {
                name: format!("v{}", source_id.0),
                param_type: core_type_to_mlir(source_type),
            }
        })
        .collect()
}

/// Convert a term's target nodes and types into the *return type* and *return statement* for a
/// MLIR function.
pub fn get_mlir_returns(term: &Term) -> (grammar::Return, grammar::Type) {
    // TODO: FIXME: support multiple returns!
    assert_eq!(
        term.targets.len(),
        1,
        "term_to_mlir currently supports single target only"
    );
    let target_id = term.targets[0];
    let target_type = &term.hypergraph.nodes[target_id.0];
    let return_type = core_type_to_mlir(target_type);

    let return_stmt = grammar::Return {
        value: grammar::Identifier(target_id.0),
        value_type: grammar::Type::Index, // FIXME!
    };

    (return_stmt, return_type)
}

// TODO: FIXME!
pub fn get_mlir_body(_term: &Term) -> Vec<grammar::Assignment> {
    vec![]
}

////////////////////////////////////////////////////////////////////////////////
// Helpers

/// Convert a [`typechecker::Type`] into an MLIR representation.
/// This maps everything except Nat to `Tensor`,
///
fn core_type_to_mlir(_core_type: &Type) -> grammar::Type {
    // TODO: FIXME: return real grammar::Type
    return grammar::Type::Index;
}
