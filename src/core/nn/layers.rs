use crate::backend::cpu::eval::Builder;
use crate::backend::cpu::ndarray::NdArray;
use crate::core::{Dtype, NdArrayType, Operation, PrimitiveType, Shape, Var};
use open_hypergraphs::lax::var::operation;
use std::f32;
use std::f32::consts::{E, PI};

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

pub fn print(builder: &Builder, name: &str, verbose: bool, x: &Var) {
    let op = Operation::Print(name.to_string(), verbose);
    let out_type = NdArrayType {
        shape: Shape(vec![]),
        dtype: Dtype::F32,
    };
    operation(builder, &[x.clone()], out_type, op);
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

pub fn eq(builder: &Builder, a: Var, b: Var) -> Var {
    let op = Operation::EQ;
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

pub fn repeat_kv(builder: &Builder, rep: usize, x: Var) -> Var {
    let shape = x.label.shape.0.clone();
    let b = shape[0];
    let num_kv_heads = shape[1];
    let s = shape[2];
    let head_dim = shape[3];

    let x = reshape(builder, Shape(vec![b, 1, num_kv_heads, s, head_dim]), x);
    let x = expand(builder, Shape(vec![b, rep, num_kv_heads, s, head_dim]), x);
    reshape(builder, Shape(vec![b, rep * num_kv_heads, s, head_dim]), x)
}

pub fn causal_mask(builder: &Builder, size: usize) -> Var {
    let t = NdArrayType {
        shape: Shape(vec![1, size]),
        dtype: Dtype::F32,
    };

    let i = arange(builder, t.clone());
    let i = expand(builder, Shape(vec![size, size]), i);

    let j = arange(builder, t.clone());
    let j = reshape(builder, Shape(vec![size, 1]), j);
    let j = expand(builder, Shape(vec![size, size]), j);

    let mask = lt(builder, j.clone(), i.clone());

    let ninf = constant(builder, mask.label.clone(), f32::MIN);

    mask * ninf
}

// Make a 2D mask with a single row set to 1 the rest to 0
// to be used to pad 1D vectors into 2D tensors
pub fn pad_mask(builder: &Builder, rows: usize, cols: usize) -> Var {
    let t = NdArrayType {
        shape: Shape(vec![1, rows]),
        dtype: Dtype::F32,
    };

    let a = arange(builder, t.clone());
    let a = reshape(builder, Shape(vec![rows, 1]), a);
    let a = expand(builder, Shape(vec![rows, cols]), a);

    let m = constant(builder, a.label.clone(), (rows - 1) as f32);
    eq(builder, a.clone(), m.clone())
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

fn layernorm_raw(builder: &Builder, eps: f32, x: Var) -> Var {
    let n = x.label.shape.0[x.label.shape.0.len() - 1];

    let s = sum(builder, x.clone());
    let constn = constant(builder, s.label.clone(), n as f32);
    let mean = sum(builder, x.clone()) / constn.clone();
    let nom = x.clone() - expand(builder, x.label.shape.clone(), mean.clone());

    let var = sum(builder, nom.clone() * nom.clone()) / constn;
    let epsilon = constant(builder, var.label.clone(), eps);
    let stddev = sqrt(builder, var + epsilon);
    let denom = expand(builder, x.label.shape, stddev);

    nom / denom
}

pub fn layernorm(builder: &Builder, eps: f32, name: &str, x: Var) -> Var {
    let shape = vec![x.label.shape.0[x.label.shape.0.len() - 1]];
    let t = NdArrayType {
        shape: Shape(shape),
        dtype: x.label.dtype,
    };
    let gamma = parameter(builder, t.clone(), format!("{name}.weight"));
    let beta = parameter(builder, t, format!("{name}.bias"));
    let lr = layernorm_raw(builder, eps, x);
    let gamma = expand(builder, lr.label.shape.clone(), gamma);
    let beta = expand(builder, lr.label.shape.clone(), beta);
    lr * gamma + beta
}

fn rmsnorm_raw(builder: &Builder, eps: f32, x: Var) -> Var {
    let n = x.label.shape.0[x.label.shape.0.len() - 1];
    let s = sum(builder, x.clone() * x.clone());
    let constn = constant(builder, s.label.clone(), n as f32);
    let ms = sum(builder, x.clone() * x.clone()) / constn;
    let epsilon = constant(builder, ms.label.clone(), eps);
    let rms = sqrt(builder, ms + epsilon);
    let b = expand(builder, x.label.shape.clone(), rms);

    x / b
}

// rmsnorm(x) = x / √(E[x²] + ε) × γ
pub fn rmsnorm(builder: &Builder, eps: f32, name: &str, x: Var) -> Var {
    let shape = vec![x.label.shape.0[x.label.shape.0.len() - 1]];
    let t = NdArrayType {
        shape: Shape(shape),
        dtype: x.label.dtype,
    };
    let gamma = parameter(builder, t.clone(), format!("{name}.weight"));
    let lr = rmsnorm_raw(builder, eps, x);
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

pub fn generate_rope_tables(
    max_position_embeddings: usize,
    head_dim: usize,
    rope_theta: f32,
) -> (Vec<f32>, NdArray<f32>, NdArray<f32>) {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        inv_freq.push(1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32));
    }

    let mut cos_table = Vec::with_capacity(max_position_embeddings * head_dim);
    let mut sin_table = Vec::with_capacity(max_position_embeddings * head_dim);

    for pos in 0..max_position_embeddings {
        for freq in inv_freq.iter().take(half_dim) {
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            // Store cos(angle) and sin(angle) for dim 2*i and 2*i+1
            cos_table.push(cos_val);
            cos_table.push(cos_val); // Repeat cos value for the pair
            sin_table.push(sin_val);
            sin_table.push(sin_val); // Repeat sin value for the pair
        }
    }

    let shape = Shape(vec![max_position_embeddings, head_dim]);
    let cos_arr = NdArray::new(cos_table, shape.clone());
    let sin_arr = NdArray::new(sin_table, shape);

    (inv_freq, cos_arr, sin_arr)
}

#[cfg(test)]
mod test {
    use super::{
        arange, causal_mask, constant, expand, gelu, generate_rope_tables, layernorm_raw, linear,
        linear_no_bias, lt, mat_mul, pad_mask, reshape, rmsnorm_raw, sigmoid, silu, softmax, tanh,
        Builder,
    };
    use crate::backend::cpu::eval::EvalState;
    use crate::backend::cpu::ndarray::{NdArray, TaggedNdArray};
    use crate::core::{Dtype, NdArrayType, Shape, Var};
    use std::collections::HashMap;
    use test_log::test;

    #[test]
    fn test_rope_tables() {
        let (inv_freq, freqs_cos, freqs_sin) = generate_rope_tables(5, 16, 100.0);
        assert_eq!(inv_freq.len(), 8);
        assert_eq!(
            inv_freq,
            vec![
                1.00000000,
                0.56234133,
                0.31622776,
                0.17782794,
                0.10000000,
                0.05623413,
                0.03162278,
                0.017782794
            ]
        );
        assert_eq!(freqs_cos.shape, Shape(vec![5, 16]));
        assert_eq!(freqs_sin.shape, Shape(vec![5, 16]));
    }

    fn test_activation<F>(x: &[f32], exp: &[f32], act: F)
    where
        F: Fn(&Builder, Var) -> Var,
    {
        let shape = Shape(vec![1, x.len()]);
        let in_type = NdArrayType {
            shape: shape.clone(),
            dtype: Dtype::F32,
        };

        let mut state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let result = act(&builder, x.clone());
            (vec![x], vec![result])
        });

        let x = NdArray::new(x.to_vec(), shape);
        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual.approx(6), exp);
    }

    fn test_norm<F>(x: &[f32], epsilon: f32, exp: &[f32], norm: F)
    where
        F: Fn(&Builder, f32, Var) -> Var,
    {
        let shape = Shape(vec![1, x.len()]);
        let in_type = NdArrayType {
            shape: shape.clone(),
            dtype: Dtype::F32,
        };

        let mut state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let result = norm(&builder, epsilon, x.clone());
            (vec![x], vec![result])
        });

        let x = NdArray::new(x.to_vec(), shape);
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
        test_norm(
            &[0., 1., 2., 3., 4.],
            1e-05,
            &[0.0, 0.408248, 0.816496, 1.224744, 1.632992],
            rmsnorm_raw,
        )
    }

    #[test]
    fn test_layernorm() {
        test_norm(
            &[0., 1., 2., 3., 4.],
            1e-05,
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

        let mut state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            // x * w^T + b
            let result = linear(&builder, 3, 2, "l", x.clone());
            (vec![x], vec![result])
        });

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

        let mut state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            // x * w^T
            let result = linear_no_bias(&builder, 3, 2, "l", x.clone());
            (vec![x], vec![result])
        });

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

        let mut state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            // x * w^T + b
            let result = linear(&builder, 3, 2, "l", x.clone());
            (vec![x], vec![result])
        });

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

        let mut state = EvalState::build(|builder| {
            let a = Var::new(builder.clone(), type_a.clone());
            let b = Var::new(builder.clone(), type_b.clone());

            let c = mat_mul(&builder, a.clone(), b.clone());

            (vec![a, b], vec![c])
        });

        // a (1×2) matrix
        let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));
        // a (2×3) matrix
        let y = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));
        // result should be a 1×3 result
        let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));
        crate::backend::cpu::kernel::batch_matmul::<f32>(&x, &y, &mut expected);

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

        let mut state = EvalState::build(|builder| {
            let i = arange(&builder, t.clone());
            let e = expand(&builder, Shape(vec![2, 6]), i.clone());
            let r = reshape(&builder, Shape(vec![6, 2]), e.clone());
            (vec![], vec![e, r])
        });

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

    #[test]
    // Make a lower triangular matrix
    fn test_tril() {
        let t = NdArrayType {
            shape: Shape(vec![1, 3]),
            dtype: Dtype::F32,
        };

        let mut state = EvalState::build(|builder| {
            // Create [[0, 1, 2],
            //         [0, 1, 2],
            //         [0, 1, 2]]
            let i = arange(&builder, t.clone());
            let i = expand(&builder, Shape(vec![3, 3]), i);

            // Create [[0, 0, 0],
            //         [1, 1, 1],
            //         [2, 2, 2]]
            let j = arange(&builder, t.clone());
            let j = reshape(&builder, Shape(vec![3, 1]), j);
            let j = expand(&builder, Shape(vec![3, 3]), j);

            let tri = !lt(&builder, j.clone(), i.clone());

            // Result [[1, 0, 0],
            //         [1, 1, 0],
            //         [1, 1, 1]]

            (vec![], vec![i, j, tri])
        });

        let [i, j, tri] = state.eval_with(vec![])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(i.shape(), Shape(vec![3, 3]));
        // assert_eq!(i.strides(), vec![0, 1]);
        assert_eq!(i.data(), &[0., 1., 2., 0., 1., 2., 0., 1., 2.]);

        assert_eq!(j.shape(), Shape(vec![3, 3]));
        // assert_eq!(j.strides(), vec![1, 0]);
        assert_eq!(j.data(), &[0., 0., 0., 1., 1., 1., 2., 2., 2.]);

        assert_eq!(tri.shape(), Shape(vec![3, 3]));
        assert_eq!(tri.strides(), vec![3, 1]);
        assert_eq!(tri.data(), &[1., 0., 0., 1., 1., 0., 1., 1., 1.]);
    }

    #[test]
    fn test_causal_mask() {
        let mut state = EvalState::build(|builder| {
            let mask = causal_mask(builder, 3);

            (vec![], vec![mask])
        });

        let [mask] = state.eval_with(vec![])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(mask.shape(), Shape(vec![3, 3]));
        assert_eq!(
            mask.data(),
            &[0., f32::MIN, f32::MIN, 0., 0., f32::MIN, 0., 0., 0.]
        );
    }

    #[test]
    fn test_pad_mask() {
        let mut state = EvalState::build(|builder| {
            let mask = pad_mask(builder, 4, 3);
            let t = NdArrayType {
                shape: Shape(vec![1, 3]),
                dtype: Dtype::F32,
            };

            let x = constant(builder, t, 5.0);
            let x = expand(builder, Shape(vec![4, 3]), x);
            let x = x * mask.clone();
            (vec![], vec![mask, x])
        });

        let [mask, x] = state.eval_with(vec![])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(mask.shape(), Shape(vec![4, 3]));
        assert_eq!(
            mask.data(),
            &[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]
        );
        assert_eq!(x.shape(), Shape(vec![4, 3]));
        assert_eq!(x.data(), &[0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 5., 5.]);
    }

    #[test]
    fn test_expand() {
        let t = NdArrayType {
            shape: Shape(vec![1, 3]),
            dtype: Dtype::F32,
        };

        let mut state = EvalState::build(|builder| {
            let i = arange(&builder, t.clone());
            let i = expand(&builder, Shape(vec![3, 3]), i);
            (vec![], vec![i])
        });

        let [i] = state.eval_with(vec![])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(i.shape(), Shape(vec![3, 3]));
        // assert_eq!(i.strides(), vec![0, 1]);
        assert_eq!(i.approx(1), &[0., 1., 2., 0., 1., 2., 0., 1., 2.]);
    }
}
