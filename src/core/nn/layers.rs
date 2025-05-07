use crate::core::{NdArrayType, Operation, PrimitiveType, Shape, Term, Var};
use open_hypergraphs::lax::var::operation;
use std::cell::RefCell;
use std::f32::consts::{E, PI};
use std::rc::Rc;

pub type Builder = Rc<RefCell<Term>>;

fn mat_mul_output_type(f: &PrimitiveType, g: &PrimitiveType) -> PrimitiveType {
    assert_eq!(f.dtype, g.dtype);
    let n = f.shape.0.len();
    let m = g.shape.0.len();
    assert_eq!(f.shape.0[n - 1], g.shape.0[m - 2]);

    let mut shape = f.shape.0[..n - 1].to_vec();
    shape.push(g.shape.0[m - 1]);
    NdArrayType {
        shape: Shape(shape),
        dtype: f.dtype,
    }
}

/// Batch matrix multiply two batches of matrices
///
/// - `f : N × A × B`
/// - `g : N × B × C`
/// - `mat_mul(builder, f, g) : N × A × C`
pub fn mat_mul(builder: &Builder, f: Var, g: Var) -> Var {
    let output_type: PrimitiveType = mat_mul_output_type(&f.label, &g.label);

    let n = f.label.shape.0.len();
    let m = g.label.shape.0.len();
    assert_eq!(n, m);

    // let batch = f.label.shape.0[..n - 2].to_vec();
    // let a = f.label.shape.0[n - 2];
    let b = f.label.shape.0[n - 1];
    let b_prime = g.label.shape.0[m - 2];
    // let c = g.label.shape.0[m - 1];

    assert_eq!(b, b_prime);

    let op = Operation::MatrixMultiply;
    operation(builder, &[f, g], output_type, op)
}

pub fn parameter(builder: &Builder, param_type: NdArrayType, name: String) -> Var {
    let op = Operation::Parameter(name);
    operation(builder, &[], param_type, op)
}

pub fn embedding(builder: &Builder, indices: Var, weights: Var) -> Var {
    let mut shape = indices.label.shape.0.clone();
    shape.push(weights.label.shape.0[1]);
    let out_type = NdArrayType {
        shape: Shape(shape),
        dtype: weights.label.dtype,
    };
    let op = Operation::Embedding;
    operation(builder, &[indices, weights], out_type, op)
}

pub fn constant(builder: &Builder, param_type: NdArrayType, k: f32) -> Var {
    let op = Operation::Const(k);
    operation(builder, &[], param_type, op)
}

pub fn lt(builder: &Builder, a: Var, b: Var) -> Var {
    let op = Operation::LT;
    operation(builder, &[a.clone(), b.clone()], a.label, op)
}

pub fn arange(builder: &Builder, param_type: NdArrayType) -> Var {
    let op = Operation::Arange;
    operation(builder, &[], param_type, op)
}

pub fn expand(builder: &Builder, shape: Shape, x: Var) -> Var {
    let out_t = NdArrayType {
        shape: shape.clone(),
        dtype: x.label.dtype,
    };
    let op = Operation::Broadcast(shape);
    operation(builder, &[x.clone()], out_t, op)
}

pub fn reshape(builder: &Builder, shape: Shape, x: Var) -> Var {
    let out_t = NdArrayType {
        shape,
        dtype: x.label.dtype,
    };
    let op = Operation::Reshape;
    operation(builder, &[x.clone()], out_t, op)
}

pub fn power(builder: &Builder, base: Var, power: Var) -> Var {
    let op = Operation::Pow;
    operation(builder, &[base.clone(), power.clone()], base.label, op)
}

pub fn sqrt(builder: &Builder, x: Var) -> Var {
    let mh = constant(builder, x.label.clone(), 0.5);
    power(builder, x, mh)
}

pub fn exp(builder: &Builder, x: Var) -> Var {
    let e = constant(builder, x.label.clone(), E);
    power(builder, e, x)
}

pub fn reduceop(builder: &Builder, op: Operation, x: Var) -> Var {
    let source = x.label.clone();

    // keep the last dimension, set it to 1
    let mut target_shape = source.shape.0.clone();
    target_shape[source.shape.0.len() - 1] = 1;
    let target = NdArrayType {
        shape: Shape(target_shape),
        dtype: source.dtype,
    };
    operation(builder, &[x.clone()], target, op)
}

pub fn sum(builder: &Builder, x: Var) -> Var {
    reduceop(builder, Operation::Sum, x)
}

pub fn max(builder: &Builder, x: Var) -> Var {
    reduceop(builder, Operation::Max, x)
}

pub fn transpose(builder: &Builder, dim0: usize, dim1: usize, x: Var) -> Var {
    let in_t = x.label.clone();

    // Create new shape with swapped dimensions
    let mut new_shape = in_t.shape.0.clone();
    new_shape.swap(dim0, dim1);

    let out_t = NdArrayType {
        shape: Shape(new_shape),
        dtype: in_t.dtype,
    };
    let op = Operation::Transpose { dim0, dim1 };
    operation(builder, &[x.clone()], out_t, op)
}

pub fn linear_b(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    name: &str,
    x: Var,
) -> Var {
    let w_type = NdArrayType {
        shape: Shape(vec![out_dim, in_dim]),
        dtype: x.label.dtype,
    };
    let w = parameter(builder, w_type.clone(), format!("{name}.weight"));

    let mut w_t = transpose(builder, 0, 1, w);

    if x.label.shape.0.len() == 3 {
        let batch_size = x.label.shape.0[0];
        w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
    }

    let m = mat_mul(builder, x.clone(), w_t);
    if bias {
        let b_type = NdArrayType {
            shape: Shape(vec![out_dim]),
            dtype: x.label.dtype,
        };
        let b = parameter(builder, b_type.clone(), format!("{name}.bias"));
        let bb = expand(builder, m.label.shape.clone(), b);
        return m + bb;
    }
    m
}

pub fn linear(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    linear_b(builder, in_dim, out_dim, true, name, x)
}

pub fn linear_no_bias(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    linear_b(builder, in_dim, out_dim, false, name, x)
}

pub fn sigmoid(builder: &Builder, x: Var) -> Var {
    let one = constant(builder, x.label.clone(), 1.0);

    one.clone() / (one + exp(builder, -x))
}

pub fn tanh(builder: &Builder, x: Var) -> Var {
    let one = constant(builder, x.label.clone(), 1.0);
    let two = constant(builder, x.label.clone(), 2.0);

    two.clone() * sigmoid(builder, two * x) - one
}

pub fn silu(builder: &Builder, x: Var) -> Var {
    x.clone() * sigmoid(builder, x)
}

// approx GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(builder: &Builder, x: Var) -> Var {
    let c = constant(builder, x.label.clone(), f32::sqrt(2. / PI));
    let one = constant(builder, x.label.clone(), 1.0);
    let three = constant(builder, x.label.clone(), 3.0);
    let half = constant(builder, x.label.clone(), 0.5);
    let k = constant(builder, x.label.clone(), 0.044715);

    half * x.clone() * (one + tanh(builder, c * (x.clone() + k * (power(builder, x, three)))))
}

fn layernorm_raw(builder: &Builder, x: Var) -> Var {
    let n = x.label.shape.0[x.label.shape.0.len() - 1];

    let s = sum(builder, x.clone());
    let constn = constant(builder, s.label.clone(), n as f32);
    let mean = sum(builder, x.clone()) / constn.clone();
    let nom = x.clone() - expand(builder, x.label.shape.clone(), mean.clone());

    let var = sum(builder, nom.clone() * nom.clone()) / constn;
    let epsilon = constant(builder, var.label.clone(), 1e-5);
    let stddev = sqrt(builder, var + epsilon);
    let denom = expand(builder, x.label.shape, stddev);

    nom / denom
}

pub fn layernorm(builder: &Builder, name: &str, x: Var) -> Var {
    let shape = vec![x.label.shape.0[x.label.shape.0.len() - 1]];
    let t = NdArrayType {
        shape: Shape(shape),
        dtype: x.label.dtype,
    };
    let gamma = parameter(builder, t.clone(), format!("{name}.weight"));
    let beta = parameter(builder, t, format!("{name}.bias"));
    let lr = layernorm_raw(builder, x);
    let gamma = expand(builder, lr.label.shape.clone(), gamma);
    let beta = expand(builder, lr.label.shape.clone(), beta);
    lr * gamma + beta
}

fn rmsnorm_raw(builder: &Builder, x: Var) -> Var {
    let n = x.label.shape.0[x.label.shape.0.len() - 1];
    let s = sum(builder, x.clone() * x.clone());
    let constn = constant(builder, s.label.clone(), n as f32);
    let ms = sum(builder, x.clone() * x.clone()) / constn;
    let epsilon = constant(builder, ms.label.clone(), 1e-5);
    let rms = sqrt(builder, ms + epsilon);
    let b = expand(builder, x.label.shape.clone(), rms);

    x / b
}

// rmsnorm(x) = x / √(E[x²] + ε) × γ
pub fn rmsnorm(builder: &Builder, name: &str, x: Var) -> Var {
    let shape = vec![x.label.shape.0[x.label.shape.0.len() - 1]];
    let t = NdArrayType {
        shape: Shape(shape),
        dtype: x.label.dtype,
    };
    let gamma = parameter(builder, t.clone(), format!("{name}.weight"));
    let lr = rmsnorm_raw(builder, x);
    let gamma = expand(builder, lr.label.shape.clone(), gamma);
    lr * gamma
}

pub fn softmax(builder: &Builder, x: Var) -> Var {
    let m = max(builder, x.clone());
    let bmax = expand(builder, x.label.shape.clone(), m);
    let x = x - bmax;
    let ex = exp(builder, x.clone());
    let s = sum(builder, ex.clone());
    let bsum = expand(builder, x.label.shape.clone(), s);
    ex / bsum
}

#[cfg(test)]
mod test {
    use super::{
        arange, expand, gelu, layernorm_raw, linear, linear_no_bias, mat_mul, reshape, rmsnorm_raw,
        sigmoid, silu, softmax, tanh, Builder,
    };
    use crate::backend::cpu::eval::EvalState;
    use crate::backend::cpu::ndarray::{NdArray, TaggedNdArray};
    use crate::core::{Dtype, NdArrayType, Shape, Term, Var};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use test_log::test;

    fn test_activation<F>(x: &[f32], exp: &[f32], act: F)
    where
        F: Fn(&Builder, Var) -> Var,
    {
        let shape = Shape(vec![1, x.len()]);
        let in_type = NdArrayType {
            shape: shape.clone(),
            dtype: Dtype::F32,
        };

        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            let result = act(&builder, x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let x = NdArray::new(x.to_vec(), shape);

        let f = Rc::try_unwrap(builder).unwrap().into_inner();
        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual.approx(6), exp);
    }

    #[test]
    fn test_tanh() {
        test_activation(&[1.0, 2.0, 3.0], &[0.761594, 0.964028, 0.995055], tanh);
    }

    #[test]
    fn test_gelu() {
        test_activation(&[1.0, 2.0, 3.0], &[0.841192, 1.954598, 2.996363], gelu);
    }

    #[test]
    fn test_sigmoid() {
        test_activation(&[1.0, 2.0, 3.0], &[0.731059, 0.880797, 0.952574], sigmoid);
    }

    #[test]
    fn test_silu() {
        test_activation(&[1.0, 2.0, 3.0], &[0.731059, 1.761594, 2.857722], silu);
    }

    #[test]
    fn test_softmax() {
        test_activation(&[1.0, 2.0, 3.0], &[0.090031, 0.244728, 0.665241], softmax);
        test_activation(
            &[100.1, 100.2, 100.3],
            &[0.300609, 0.332224, 0.367167],
            softmax,
        );
    }

    #[test]
    fn test_rmsnorm() {
        test_activation(
            &[0., 1., 2., 3., 4.],
            &[0.0, 0.408248, 0.816496, 1.224744, 1.632992],
            rmsnorm_raw,
        )
    }

    #[test]
    fn test_layernorm() {
        test_activation(
            &[0., 1., 2., 3., 4.],
            &[-1.414210, -0.707105, 0.000000, 0.707105, 1.414210],
            layernorm_raw,
        )
    }

    #[test]
    fn test_linear() {
        let in_type = NdArrayType {
            shape: Shape(vec![2, 3]),
            dtype: Dtype::F32,
        };
        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            // Run linear layer (x * w^T + b)
            let result = linear(&builder, 3, 2, "l", x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        let mut state = EvalState::from_lax(f);

        // Create test input data
        let x = NdArray::new(vec![1.0, 2.0, 3.0, 2.0, 3.0, 1.0], Shape(vec![2, 3]));

        // Parameter values
        let w = NdArray::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Shape(vec![2, 3]));
        let b = NdArray::new(vec![0.1, 0.1], Shape(vec![2]));

        let mut parameters = HashMap::new();
        parameters.insert("l.weight".to_string(), w.into());
        parameters.insert("l.bias".to_string(), b.into());

        state.set_parameters(parameters);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual.approx(6), &[1.5, 3.3, 1.2, 3.0]);
    }

    #[test]
    fn test_linear_no_bias() {
        let in_type = NdArrayType {
            shape: Shape(vec![2, 3]),
            dtype: Dtype::F32,
        };
        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            // x * w^T
            let result = linear_no_bias(&builder, 3, 2, "l", x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        let mut state = EvalState::from_lax(f);

        // Create test input data
        let x = NdArray::new(vec![1.0, 2.0, 3.0, 2.0, 3.0, 1.0], Shape(vec![2, 3]));

        // Parameter values
        let w = NdArray::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Shape(vec![2, 3]));

        let mut parameters = HashMap::new();
        parameters.insert("l.weight".to_string(), w.into());

        state.set_parameters(parameters);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual.approx(6), &[1.4, 3.2, 1.1, 2.9]);
    }

    #[test]
    fn test_linear_batch() {
        let in_type = NdArrayType {
            shape: Shape(vec![2, 2, 3]),
            dtype: Dtype::F32,
        };
        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            // Run linear layer (x * w^T + b)
            let result = linear(&builder, 3, 2, "l", x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }
        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        let mut state = EvalState::from_lax(f);

        // Create test input data
        let mut x = NdArray::new(vec![1.0, 2.0, 3.0, 2.0, 3.0, 1.0], Shape(vec![2, 3]));
        x.shape = Shape(vec![2, 2, 3]);
        x.strides = vec![0, 3, 1];

        // Parameter values
        let w = NdArray::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Shape(vec![2, 3]));
        let b = NdArray::new(vec![0.1, 0.1], Shape(vec![2]));

        let mut parameters = HashMap::new();
        parameters.insert("l.weight".to_string(), w.into());
        parameters.insert("l.bias".to_string(), b.into());

        state.set_parameters(parameters);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual.approx(6), &[1.5, 3.3, 1.2, 3.0, 1.5, 3.3, 1.2, 3.0]);
    }

    #[test]
    fn test_matmul() {
        let type_a = NdArrayType {
            shape: Shape(vec![1, 2]),
            dtype: Dtype::F32,
        };

        let type_b = NdArrayType {
            shape: Shape(vec![2, 3]),
            dtype: Dtype::F32,
        };

        let state = Rc::new(RefCell::new(Term::empty()));

        {
            let a = Var::new(state.clone(), type_a.clone());
            let b = Var::new(state.clone(), type_b.clone());

            let c = mat_mul(&state, a.clone(), b.clone());

            state.borrow_mut().sources = vec![a.new_source(), b.new_source()];
            state.borrow_mut().targets = vec![c.new_target()];
        }
        let f = Rc::try_unwrap(state).unwrap().into_inner();

        // a (1×2) matrix
        let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));
        // a (2×3) matrix
        let y = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));
        // result should be a 1×3 result
        let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));
        crate::backend::cpu::kernel::batch_matmul::<f32>(&x, &y, &mut expected);

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into(), y.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_arange() {
        let t = NdArrayType {
            shape: Shape(vec![1, 6]),
            dtype: Dtype::F32,
        };
        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let i = arange(&builder, t.clone());
            let e = expand(&builder, Shape(vec![2, 6]), i.clone());
            let r = reshape(&builder, Shape(vec![6, 2]), e.clone());
            builder.borrow_mut().sources = vec![];
            builder.borrow_mut().targets = vec![e.new_target(), r.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        let mut state = EvalState::from_lax(f);

        let [e, r] = state.eval_with(vec![])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(e.shape(), Shape(vec![2, 6]));
        assert_eq!(
            e.approx(1),
            &[0., 1., 2., 3., 4., 5., 0., 1., 2., 3., 4., 5.]
        );
        assert_eq!(r.shape(), Shape(vec![6, 2]));
    }
}
