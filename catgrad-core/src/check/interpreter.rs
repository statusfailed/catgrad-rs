// Catgrad's shape checker is an abstract interpreter for the *core* dialect.
// It uses the core declarations in [`crate::stdlib`].
use crate::category::lang::*;
use crate::ssa::*;
use crate::stdlib::Environment;

use open_hypergraphs::lax::NodeId;

use super::types::*;

#[allow(clippy::result_large_err)]
pub fn check(env: &Environment, params: &Parameters, term: TypedTerm) -> ShapeCheckResult {
    let TypedTerm {
        term, source_type, ..
    } = term;
    check_with(env, params, term, source_type)
}

/// Assign a shape value to each node in a term (hypergraph).
#[allow(clippy::result_large_err)]
pub fn check_with(
    env: &Environment,
    params: &Parameters,
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
    let ssa = ssa(term.to_strict())?;

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
        let coargs = apply(env, params, &op, &args)?;
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
fn apply(
    env: &Environment,
    params: &Parameters,
    ssa: &SSA<Object, Operation>,
    args: &[Value],
) -> ShapeCheckResult {
    match &ssa.op {
        Operation::Definition(op) => {
            // look up term
            let TypedTerm { term, .. } =
                env.definitions.get(op).ok_or(ShapeCheckError::ApplyError(
                    ApplyError::UnknownOp(op.clone()),
                    ssa.clone(),
                    args.to_vec(),
                ))?;
            apply_definition(env, params, term, args)
        }
        Operation::Declaration(op) => apply_declaration(env, params, op, args, ssa)
            .map_err(|e| ShapeCheckError::ApplyError(e, ssa.clone(), args.to_vec())),
        Operation::Literal(lit) => apply_literal(lit)
            .map_err(|e| ShapeCheckError::ApplyError(e, ssa.clone(), args.to_vec())),
    }
}

fn apply_declaration(
    env: &Environment,
    params: &Parameters,
    op: &Path,
    args: &[Value],
    ssa: &SSA<Object, Operation>,
) -> ApplyResult {
    let shape_op = env
        .declarations
        .get(op)
        .ok_or(ApplyError::UnknownOp(op.clone()))?;
    s_apply(env, params, shape_op, args, ssa)
}

// TODO: manage recursion explicitly with a stack
#[allow(clippy::result_large_err)]
fn apply_definition(
    env: &Environment,
    params: &Parameters,
    term: &Term,
    args: &[Value],
) -> Result<Vec<Value>, ShapeCheckError> {
    let source_values = args.to_vec();
    let nodes = check_with(env, params, term.clone(), source_values)?;
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
