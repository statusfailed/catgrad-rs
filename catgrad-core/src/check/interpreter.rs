// Catgrad's shape checker is an abstract interpreter for the *shaped* dialect.
use crate::category::{bidirectional::*, shape};
use crate::ssa::*;
use open_hypergraphs::lax::NodeId;

use std::collections::HashMap;

use super::types::*;

/*
pub fn check(term: Term, ty: Term) -> ShapeCheckResult {
    // Set a "Var" (symbolic) value for each input node
    let source_values = ty
        .sources
        .iter()
        .enumerate()
        .map(|(i, id)| var(i, term.hypergraph.nodes[id.0].clone()))
        .collect();

    check_with(term, source_values)
}
*/

#[allow(clippy::result_large_err)]
pub fn check(term: Term, source_values: Vec<Value>) -> ShapeCheckResult {
    let ops = op_decls();
    let env = crate::nn::stdlib();

    check_with(&ops, &env, term, source_values)
}

/// Assign a shape value to each node in a term (hypergraph).
#[allow(clippy::result_large_err)]
pub fn check_with(
    ops: &HashMap<Path, shape::Operation>,
    env: &Environment,
    term: Term,
    source_values: Vec<Value>,
) -> ShapeCheckResult {
    assert_eq!(source_values.len(), term.sources.len());

    // Create evaluation state
    let n = term.hypergraph.nodes.len();
    let mut state: Vec<Option<Value>> = vec![None; n];

    // Set a "Var" (symbolic) value for each input node
    for (i, id) in term.sources.iter().enumerate() {
        state[id.0] = Some(source_values[i].clone());
    }

    // Create SSA
    let ssa = ssa(term.to_strict());

    // Iterate through SSA
    for op in ssa {
        // read arg values from graph
        let mut args = vec![];
        for (NodeId(i), _) in &op.sources {
            if let Some(value) = state[*i].clone() {
                args.push(value)
            } else {
                let _v = state[*i].clone();
                return Err(ShapeCheckError::EvaluationOrder(op.edge_id));
            }
        }

        // Compute output values and write into the graph
        let coargs = apply(ops, env, &op, &args)?;
        assert_eq!(coargs.len(), op.targets.len(), "{op:?}");

        for ((NodeId(i), _), value) in op.targets.iter().zip(coargs.into_iter()) {
            state[*i] = Some(value)
        }
    }

    node_values(state).map_err(ShapeCheckError::Unevaluated)
}

// Get values of each node, or return a list of node IDs which were unevaluated
fn node_values<T>(v: Vec<Option<T>>) -> Result<Vec<T>, Vec<NodeId>> {
    let mut values = Vec::with_capacity(v.len());
    let mut none_indices = Vec::new();

    for (i, opt) in v.into_iter().enumerate() {
        match opt {
            Some(val) => values.push(val),
            None => none_indices.push(NodeId(i)),
        }
    }

    if none_indices.is_empty() {
        Ok(values)
    } else {
        Err(none_indices)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Apply and helper functions

use super::apply::*;

// Get a value for each resulting NodeId.
#[allow(clippy::result_large_err)]
pub fn apply(
    ops: &HashMap<Path, shape::Operation>,
    env: &Environment,
    ssa: &SSA<Object, Operation>,
    args: &[Value],
) -> ShapeCheckResult {
    match &ssa.op {
        Operation::Definition(op) => {
            // look up term
            let OperationDefinition { term, .. } =
                env.operations.get(op).ok_or(ShapeCheckError::ApplyError(
                    ApplyError::UnknownOp(op.clone()),
                    ssa.clone(),
                    args.to_vec(),
                ))?;
            apply_definition(ops, env, term, args)
        }
        Operation::Declaration(op) => apply_declaration(ops, op, args, ssa)
            .map_err(|e| ShapeCheckError::ApplyError(e, ssa.clone(), args.to_vec())),
        Operation::Literal(lit) => apply_literal(lit)
            .map_err(|e| ShapeCheckError::ApplyError(e, ssa.clone(), args.to_vec())),
    }
}

fn apply_declaration(
    ops: &HashMap<Path, shape::Operation>,
    op: &Path,
    args: &[Value],
    ssa: &SSA<Object, Operation>,
) -> ApplyResult {
    let shape_op = ops.get(op).ok_or(ApplyError::UnknownOp(op.clone()))?;
    s_apply(shape_op, args, ssa)
}

// TODO: manage recursion explicitly with a stack
#[allow(clippy::result_large_err)]
fn apply_definition(
    ops: &HashMap<Path, shape::Operation>,
    env: &Environment,
    term: &Term,
    args: &[Value],
) -> Result<Vec<Value>, ShapeCheckError> {
    let source_values = args.to_vec();
    let nodes = check_with(ops, env, term.clone(), source_values)?;
    Ok(term
        .targets
        .iter()
        .map(|node_id| nodes[node_id.0].clone())
        .collect())
}

// TODO: tidy this mess up
fn apply_literal(lit: &Literal) -> ApplyResult {
    // F32/U32 literals are scalars;
    // Dtype literals are Dtypes.
    Ok(vec![match lit {
        Literal::Dtype(dtype) => Value::Dtype(DtypeExpr::Constant(dtype.clone())),
        Literal::F32(_) => {
            let dtype = Dtype::F32;
            let ty = NdArrayType {
                dtype: DtypeExpr::Constant(dtype),
                shape: ShapeExpr::Shape(vec![]),
            };
            Value::Tensor(TypeExpr::NdArrayType(ty))
        }
        Literal::U32(_) => {
            let dtype = Dtype::U32;
            let ty = NdArrayType {
                dtype: DtypeExpr::Constant(dtype),
                shape: ShapeExpr::Shape(vec![]),
            };
            Value::Tensor(TypeExpr::NdArrayType(ty))
        }
    }])
}
