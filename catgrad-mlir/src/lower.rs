//! Lower a catgrad::category::lang::Term to MLIR
use catgrad::category::lang;
use catgrad::prelude::*;
use catgrad::ssa::{SSA, ssa};

use open_hypergraphs::lax::OpenHypergraph;

use crate::grammar;

pub type Term = OpenHypergraph<typecheck::Type, lang::Operation>;

/// Lower a [`Term`] to MLIR text, giving it the specified name.
pub fn term_to_func(name: &str, term: Term) -> grammar::Func {
    let parameters = get_mlir_parameters(&term);
    let (return_stmt, return_type) = get_mlir_returns(&term);
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
pub fn get_mlir_returns(term: &Term) -> (grammar::Return, Vec<grammar::Type>) {
    let typed_ids = term
        .targets
        .iter()
        .map(|t| {
            let id = t.0;
            let catgrad_ty = &term.hypergraph.nodes[id];
            let id = grammar::Identifier(id);
            let ty = core_type_to_mlir(catgrad_ty);
            grammar::TypedIdentifier { id, ty }
        })
        .collect::<Vec<_>>();

    let types = typed_ids.iter().map(|tid| tid.ty.clone()).collect();
    (grammar::Return(typed_ids), types)
}

pub fn get_mlir_body(term: Term) -> Vec<grammar::Assignment> {
    let ops = ssa(term.to_strict()).expect("FIXME: unable to decompose input term");
    ops.iter().map(to_assignments).flatten().collect()
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

fn to_typed_identifier((n, t): &(open_hypergraphs::lax::NodeId, Type)) -> grammar::TypedIdentifier {
    grammar::TypedIdentifier {
        id: grammar::Identifier(n.0),
        ty: core_type_to_mlir(t),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Map individual ops
// TODO: own module?
// TODO: generate a unique kernel for each <name, source, target> type?

/// Make a list of assignment statements from a single op
fn to_assignments(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // TODO: prelude - generate any "outs"?

    // Get LHS of assignment and types
    let (result, return_types): (Vec<grammar::Identifier>, Vec<grammar::Type>) = ssa
        .targets
        .iter()
        .map(|(i, t)| (grammar::Identifier(i.0), core_type_to_mlir(t)))
        .unzip();

    let ins = ssa.sources.iter().map(to_typed_identifier).collect();
    let outs = ssa.targets.iter().map(to_typed_identifier).collect();

    match &ssa.op {
        // Declarations lower to explicit snippets
        lang::Operation::Declaration(path) => {
            let expr = grammar::Expr::Operation(grammar::Operation {
                name: format!("\"{}\"", path.to_string()),
                ins,
                outs,
                return_types,
                attrs: None,
                inner_block: Some("TODO".to_string()),
            });

            vec![grammar::Assignment { result, expr }]
        }

        // Definitions always lower to *kernel* calls.
        lang::Operation::Definition(path) => {
            let ins = ssa.sources.iter().map(to_typed_identifier).collect();
            let outs = ssa.targets.iter().map(to_typed_identifier).collect();

            let expr = grammar::Expr::Operation(grammar::Operation {
                name: format!("\"{}\"", path.to_string()),
                ins,
                outs,
                return_types,
                attrs: None,
                inner_block: Some("TODO".to_string()),
            });

            vec![grammar::Assignment { result, expr }]
        }
        lang::Operation::Literal(lit) => vec![grammar::Assignment {
            result,
            expr: literal_to_operation(&lit),
        }],
    }
}

fn literal_to_operation(lit: &lang::Literal) -> grammar::Expr {
    let (attr, ty) = match lit {
        lang::Literal::F32(x) => (x.to_string(), grammar::Type::F32),
        lang::Literal::U32(x) => (x.to_string(), grammar::Type::U32),
        lang::Literal::Nat(x) => (x.to_string(), grammar::Type::Index),
        lang::Literal::Dtype(_) => todo!(), // error
    };

    grammar::Expr::Operation(grammar::Operation {
        name: "arith.constant".to_string(),
        ins: vec![],
        outs: vec![],
        return_types: vec![ty],
        attrs: Some(attr),
        inner_block: None,
    })
}
