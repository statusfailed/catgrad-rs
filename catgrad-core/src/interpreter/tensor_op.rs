//! Tensor operation implementations for the interpreter

use super::backend::*;
use super::{ResultValues, TaggedNdArray, TaggedNdArrayTuple, Value};
use crate::abstract_interpreter::util::{get_exact_arity, to_tensor};
use crate::abstract_interpreter::{CoreSSA, EvalResult, InterpreterError};
use crate::category::core::{Constant, Dtype, ScalarOp, TensorOp};

/// Apply a Tensor operation
pub(crate) fn tensor_op<B: Backend>(
    backend: &B,
    ssa: &CoreSSA,
    args: Vec<Value<B>>,
    tensor_op: &TensorOp,
) -> ResultValues<B> {
    match tensor_op {
        TensorOp::MatMul => binop(backend, args, ssa, B::matmul),
        /*
                TensorOp::Constant(c) => tensor_constant(backend, args, ssa, c),
                TensorOp::Sum => tensor_sum(backend, args, ssa),
                TensorOp::Max => tensor_max(backend, args, ssa),
                TensorOp::Arange => tensor_arange(backend, args, ssa),
                TensorOp::Argmax => tensor_argmax(backend, args, ssa),
                TensorOp::Broadcast => tensor_broadcast(backend, args, ssa),
                TensorOp::Reshape => tensor_reshape(backend, args, ssa),
                TensorOp::Transpose => tensor_transpose(backend, args, ssa),
        */
        TensorOp::Map(ScalarOp::Add) => binop(backend, args, ssa, B::add),
        TensorOp::Map(ScalarOp::Sub) => binop(backend, args, ssa, B::sub),
        TensorOp::Map(ScalarOp::Pow) => binop(backend, args, ssa, B::pow),
        TensorOp::Map(ScalarOp::Sin) => unary_op(backend, args, ssa, B::sin),
        TensorOp::Map(ScalarOp::Cos) => unary_op(backend, args, ssa, B::cos),
        TensorOp::Map(ScalarOp::Neg) => unary_op(backend, args, ssa, B::neg),
        TensorOp::Map(ScalarOp::Mul) => binop(backend, args, ssa, B::mul),
        TensorOp::Map(ScalarOp::Div) => binop(backend, args, ssa, B::div),
        TensorOp::Map(ScalarOp::LT) => binop(backend, args, ssa, B::lt),
        TensorOp::Map(ScalarOp::EQ) => binop(backend, args, ssa, B::eq),
        /*
                TensorOp::Cast => tensor_cast(backend, args, ssa),
                TensorOp::Scalar => tensor_scalar(backend, args, ssa),
                TensorOp::Concat => tensor_concat(backend, args, ssa),
                TensorOp::Index => tensor_index(backend, args, ssa),
                TensorOp::Slice => tensor_slice(backend, args, ssa),
                TensorOp::Copy => todo!("copy"),
        */
        op => todo!("interpreter todo op {op:?}"),
    }
}

/*
fn err<K: Into<ApplyErrorKind>>(kind: K, ssa: &CoreSSA) -> Box<ApplyError> {
    Box::new(ApplyError {
        kind: kind.into(),
        ssa: ssa.clone(),
    })
}

pub(crate) fn tensor_constant<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>, // must be empty
    ssa: &CoreSSA,
    c: &Constant,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if !args.is_empty() {
        return Err(err(ApplyErrorKind::ArityError, ssa));
    }

    let tagged = match c {
        Constant::F32(x) => TaggedNdArray::from_slice(backend, &[*x], super::Shape(vec![])),
        Constant::U32(x) => TaggedNdArray::from_slice(backend, &[*x], super::Shape(vec![])),
    }
    .map_err(|e| err(e, ssa))?;

    Ok(vec![Value::NdArray(tagged)])
}

pub(crate) fn tensor_scalar<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(err(ApplyErrorKind::ArityError, ssa));
    }

    let value: u32 = match args[0] {
        Value::Nat(n) => n
            .try_into()
            .map_err(|_| err(ApplyErrorKind::NatOverflow, ssa))?,
        _ => return Err(err(ApplyErrorKind::TypeError, ssa)),
    };

    let tensor = backend
        .ndarray_from_slice(&[value], super::Shape(vec![]))
        .map_err(|e| err(e, ssa))?;

    let result = TaggedNdArrayTuple::U32([tensor]);
    Ok(vec![Value::NdArray(result)])
}

fn tensor_cast<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];
    let target_dtype = &args[1];

    let Value::Dtype(target_dtype) = target_dtype else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.cast(x.clone(), target_dtype.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_sum<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.sum(x.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_max<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.max(x.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_argmax<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.argmax(x.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_reshape<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [new_shape, tensor] - reshape(builder, new_shape, tensor)
    if let (Value::Shape(new_shape), Value::NdArray(x)) = (args.remove(0), args.remove(0)) {
        let result = backend.reshape(x, new_shape);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_transpose<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 3 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [new_shape, tensor] - reshape(builder, new_shape, tensor)
    if let (Value::NdArray(x), Value::Nat(dim0), Value::Nat(dim1)) =
        (args.remove(0), args.remove(0), args.remove(0))
    {
        let result = backend.transpose(x, dim0, dim1);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}
fn tensor_broadcast<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    if let (Value::Shape(shape_prefix), Value::NdArray(x)) = (args.remove(1), args.remove(0)) {
        let result = backend.broadcast(x, shape_prefix);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_arange<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let Value::Nat(end) = args[0] else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.arange(end);
    Ok(vec![Value::NdArray(result)])
}

fn tensor_index<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 3 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [input, indices]
    if let (Value::NdArray(input), Value::Nat(dim), Value::NdArray(indices)) =
        (args.remove(0), args.remove(0), args.remove(0))
    {
        let result = backend.index(input, dim, indices);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_concat<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 3 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [tensor, tensor, dim]
    if let (Value::NdArray(a), Value::NdArray(b), Value::Nat(dim)) =
        (args.remove(0), args.remove(0), args.remove(0))
    {
        let result = backend.concat(a, b, dim);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_slice<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 4 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [input, dim, start, end]
    if let (Value::NdArray(input), Value::Nat(dim), Value::Nat(start), Value::Nat(len)) = (
        args.remove(0),
        args.remove(0),
        args.remove(0),
        args.remove(0),
    ) {
        let result = backend.slice(input, dim, start, len);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}
*/

#[allow(type_alias_bounds)]
type Binop<B: Backend> = fn(&B, TaggedNdArrayTuple<B, 2>) -> TaggedNdArrayTuple<B, 1>;

#[allow(type_alias_bounds)]
type Unaryop<B: Backend> = fn(&B, TaggedNdArray<B>) -> TaggedNdArray<B>;

fn binop<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
    callback: Binop<B>,
) -> ResultValues<B> {
    let args = try_into_tagged_ndarrays::<B, 2>(args, ssa)?;
    let result = callback(backend, args);
    Ok(vec![Value::Tensor(result)])
}

fn unary_op<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
    callback: Unaryop<B>,
) -> ResultValues<B> {
    let [x] = get_exact_arity(ssa, args)?;
    let result = callback(backend, to_tensor(ssa, x)?);
    Ok(vec![Value::Tensor(result)])
}

/*
// TODO: binop has some boilerplate- have to unpack/repack a lot. can we fix that using run_op &
// into() on taggedndarraytuple?
#[allow(dead_code)]
/// Run an M → 1 op taking M NdArray values of the same dtype, producing an NdArray.
fn run_op<B: Backend, F, const M: usize>(
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
    f: F,
) -> Result<Vec<Value<B>>, Box<ApplyError>>
where
    F: Fn(TaggedNdArrayTuple<B, M>) -> TaggedNdArrayTuple<B, 1>,
{
    Ok(vec![Value::NdArray(f(try_into_tagged_ndarrays::<B, M>(
        args, ssa,
    )?))])
}
*/

/// Convert a Vec<Value<B>> into TaggedNdArrays<B, N> with compile-time length checking
pub(crate) fn try_into_tagged_ndarrays<B: Backend, const N: usize>(
    values: Vec<Value<B>>, // TODO: rename args
    ssa: &CoreSSA,
) -> EvalResult<TaggedNdArrayTuple<B, N>> {
    // If no args, type is ambiguous, but this is a programmer error.
    if N == 0 {
        panic!("try_into_tagged_ndarrays is undefined for N <= 0");
    }

    // Get exactly N tensors
    let tensors: Vec<TaggedNdArray<B>> = get_exact_arity::<N, _>(ssa, values)?
        .into_iter()
        .map(|x| to_tensor(ssa, x))
        .collect::<Result<_, _>>()?;
    let dtype = tensors[0].dtype();

    // Collect each tag into its own typed array
    let mut f32_arrays = Vec::new();
    let mut u32_arrays = Vec::new();
    for x in tensors {
        match x {
            TaggedNdArrayTuple::F32([x]) => f32_arrays.push(x),
            TaggedNdArrayTuple::U32([x]) => u32_arrays.push(x),
        }

        // early exit: if one dtype didn't match, we bail. (only happens when product is nonzeroD)
        if f32_arrays.len().min(u32_arrays.len()) > 0 {
            return Err(InterpreterError::TypeError(ssa.edge_id));
        }
    }

    Ok(match dtype {
        Dtype::F32 => f32_arrays.try_into().ok().map(TaggedNdArrayTuple::F32),
        Dtype::U32 => u32_arrays.try_into().ok().map(TaggedNdArrayTuple::U32),
    }
    .unwrap()) // unwrap OK: we already checked arity!
}
