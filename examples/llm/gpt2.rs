// GPT-2 model description

use super::{Config, ModelBuilder};
use catgrad::backend::cpu::eval::{Builder, EvalState};
use catgrad::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Shape, Var};
use std::collections::HashMap;

pub struct Model;

impl Model {
    // The original GPT2 checkpoints use a Conv1D layer instead of linear,
    // equivalent to a linear layer with weights in transposed order
    fn gpt_linear(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
        let w_type = NdArrayType::new(Shape(vec![in_dim, out_dim]), x.label.dtype);
        let b_type = NdArrayType::new(Shape(vec![out_dim]), x.label.dtype);

        let w = parameter(builder, w_type, format!("{name}.weight"));
        let b = parameter(builder, b_type, format!("{name}.bias"));

        // w is already transposed in GPT-2 checkpoints
        let mut w_t = w;

        if x.label.shape.0.len() == 3 {
            let batch_size = x.label.shape.0[0];
            w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
        }

        let m = mat_mul(builder, x, w_t);
        let bb = expand(builder, m.label.shape.clone(), b);
        m + bb
    }

    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType::new(Shape(vec![config.vocab_size, config.n_embd]), Dtype::F32);
        let weights = parameter(builder, t, "wte.weight".to_string());
        let we = embedding(builder, x.clone(), weights);

        let t = NdArrayType::new(Shape(vec![config.n_positions, config.n_embd]), Dtype::F32);
        let pos = arange(builder, x.label);
        let weights = parameter(builder, t, "wpe.weight".to_string());
        let pe = embedding(builder, pos, weights);

        we + pe
    }

    pub fn attention(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let dim = config.n_embd;
        let num_heads = config.n_head;
        let head_dim = dim / num_heads;

        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        // let c_attn = gpt_linear(builder, dim, 3 * dim, &format!("{name}.c_attn"), x.clone());
        let k = Model::gpt_linear(builder, dim, dim, &format!("{name}.key"), x.clone());
        let q = Model::gpt_linear(builder, dim, dim, &format!("{name}.query"), x.clone());
        let v = Model::gpt_linear(builder, dim, dim, &format!("{name}.value"), x);

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let attn = attn / denom;

        let mask = causal_mask(builder, s);
        let mask = expand(builder, Shape(vec![b, num_heads, s, s]), mask);
        let attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let attn = reshape(builder, Shape(vec![b, s, dim]), attn);

        let c_proj = Model::gpt_linear(builder, dim, dim, &format!("{name}.c_proj"), attn);
        c_proj
    }

    pub fn mlp(builder: &Builder, dim: usize, name: &str, x: Var) -> Var {
        let x = Model::gpt_linear(builder, dim, dim * 4, &format!("{name}.c_fc"), x);
        let x = gelu(builder, x);
        let x = Model::gpt_linear(builder, dim * 4, dim, &format!("{name}.c_proj"), x);
        x
    }

    pub fn layer(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_epsilon,
            &format!("{name}.ln_1"),
            x,
        );
        let x = Model::attention(builder, config, &format!("{name}.attn"), x);
        let x = res + x;
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_epsilon,
            &format!("{name}.ln_2"),
            x,
        );
        let x = Model::mlp(builder, config.n_embd, &format!("{name}.mlp"), x);
        x + res
    }
}

impl ModelBuilder for Model {
    fn build(&mut self, batches: usize, tokens: usize, config: &Config) -> EvalState {
        let in_type = NdArrayType::new(Shape(vec![batches, tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let emb = Model::embeddings(builder, config, x.clone());

            let mut result = emb;

            for i in 0..config.n_layer {
                result = Model::layer(builder, config, &format!("h.{i}"), result);
            }

            result = layernorm(builder, config.layer_norm_epsilon, "ln_f", result);

            // GPT-2 uses weight tying so lm_head is the same as wte
            let lm_head = linear_no_bias(builder, config.n_embd, config.vocab_size, "wte", result);
            (vec![x], vec![lm_head])
        });

        state
    }

    fn post_load(&mut self, tensors: &mut HashMap<String, TaggedNdArray>) {
        let attn_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.contains("c_attn"))
            .cloned()
            .collect();

        // Split these keys into component Q, K, and V
        for attn_key in attn_keys {
            if let Some(TaggedNdArray::F32(array)) = tensors.get(&attn_key) {
                let mut shape = array.shape.clone();
                let l = shape.0.len() - 1;
                let dim = shape.0[l] / 3;

                // Create the three tensors
                let q_name = attn_key.replace("c_attn", "query");
                let k_name = attn_key.replace("c_attn", "key");
                let v_name = attn_key.replace("c_attn", "value");

                let m = shape.size() / shape.0[l];

                // Split the tensor data
                let mut q_data: Vec<f32> = Vec::with_capacity(m * dim);
                let mut k_data: Vec<f32> = Vec::with_capacity(m * dim);
                let mut v_data: Vec<f32> = Vec::with_capacity(m * dim);
                for (i, c) in array.data.chunks_exact(dim).enumerate() {
                    if i % 3 == 0 {
                        q_data.extend_from_slice(c);
                    } else if i % 3 == 1 {
                        k_data.extend_from_slice(c);
                    } else {
                        v_data.extend_from_slice(c);
                    }
                }

                shape.0[l] = dim;

                // Create new arrays with proper shapes
                let q_shape = shape.clone();
                let k_shape = shape.clone();
                let v_shape = shape;

                // Insert the new tensors
                tensors.insert(q_name, TaggedNdArray::F32(NdArray::new(q_data, q_shape)));
                tensors.insert(k_name, TaggedNdArray::F32(NdArray::new(k_data, k_shape)));
                tensors.insert(v_name, TaggedNdArray::F32(NdArray::new(v_data, v_shape)));

                tensors.remove(&attn_key);
            }
        }
    }
}
