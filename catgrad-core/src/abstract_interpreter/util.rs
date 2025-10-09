use super::types::*;

/// Make sure an op has exact arity m, consistent with arguments
pub(crate) fn get_exact_arity<const N: usize, T>(
    ssa: &CoreSSA,
    args: Vec<T>,
) -> EvalResult<[T; N]> {
    if ssa.sources.len() != N {
        return Err(InterpreterError::ArityError(ssa.edge_id));
    }

    if args.len() != N {
        // TODO: return a better error here
        return Err(InterpreterError::ArityError(ssa.edge_id));
    }

    args.try_into()
        .map_err(|_e| InterpreterError::ArityError(ssa.edge_id))
}

////////////////////////////////////////////////////////////////////////////////
// Match cases of Value or yield an EvalResult.
// NOTE: we don't use TryInto here because we need the ssa value to build an error.

// unwrap a Value to a nat
pub(crate) fn to_nat<V: InterpreterValue>(ssa: &CoreSSA, v: Value<V>) -> EvalResult<V::Nat> {
    match v {
        Value::Nat(v) => Ok(v),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

pub(crate) fn to_shape<V: InterpreterValue>(ssa: &CoreSSA, v: Value<V>) -> EvalResult<V::Shape> {
    match v {
        Value::Shape(s) => Ok(s),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

pub(crate) fn to_tensor<V: InterpreterValue>(ssa: &CoreSSA, v: Value<V>) -> EvalResult<V::Tensor> {
    match v {
        Value::Tensor(t) => Ok(t),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}
