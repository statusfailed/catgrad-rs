// Granite 3.x model description

use super::utils::{Cache, Config, ModelBuilder};
use crate::nn::layers::*;
use crate::nn::rope::apply_rope_embedding;
use catgrad::backend::cpu::eval::Builder;
use catgrad::core::{NdArrayType, Shape, Var};

pub struct Model;

impl ModelBuilder for Model {
    fn build(
        &self,
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        x: Var,
    ) -> Var {
        let tokens = x.label.shape.0[1];

        let emb = Model::embeddings(builder, config, x);

        let mut result = emb;

        for i in 0..config.num_hidden_layers {
            result = Model::layer(
                builder,
                i,
                config,
                cache,
                pos,
                &format!("model.layers.{i}"),
                result,
            );
        }

        result = rmsnorm(builder, config.rms_norm_eps, "model.norm", result);

        // Get the logits for the last token only
        if tokens > 1 {
            result = narrow(builder, 1, tokens - 1, 1, result);
        }

        let logits_scaling = constant(builder, result.label.clone(), config.logits_scaling);

        result = result / logits_scaling;
        // Use weight tying
        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            "model.embed_tokens",
            result,
        )
    }
}

// Build linear layer from weight params
fn linear_w(builder: &Builder, in_dim: usize, out_dim: usize, weight: Var, x: Var) -> Var {
    let mut w_t = transpose(builder, 1, 2, weight);
    if x.label.shape.0.len() == 3 {
        let batch_size = x.label.shape.0[0];
        w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
    }

    mat_mul(builder, x, w_t)
}

impl Model {
    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType::new(
            Shape(vec![config.vocab_size, config.hidden_size]),
            config.dtype,
        );
        let weights = parameter(builder, t, "model.embed_tokens.weight".to_string());
        let emb = embedding(builder, x, weights);
        let mul = constant(builder, emb.label.clone(), config.embedding_multiplier);
        emb * mul
    }

    pub fn attention(
        builder: &Builder,
        layer_id: usize,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = config.hidden_size / num_heads;
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear_no_bias(builder, dim, dim, &format!("{name}.q_proj"), x.clone());
        let k = linear_no_bias(
            builder,
            dim,
            dim / rep,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_no_bias(builder, dim, dim / rep, &format!("{name}.v_proj"), x);

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let k = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let mul = constant(builder, attn.label.clone(), config.attention_multiplier);
        let attn = attn * mul;

        let mask = causal_mask(builder, s, attn.label.dtype);
        let mask = expand(builder, attn.label.shape.clone(), mask);
        let attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape(builder, Shape(vec![b, s, dim]), x);
        linear_no_bias(builder, dim, dim, &format!("{name}.o_proj"), x)
    }

    pub fn mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let gated = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size,
            &format!("{name}.gate_proj"),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size,
            &format!("{name}.up_proj"),
            x,
        );
        let x = silu(builder, gated) * up; // SwiGLU

        linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.down_proj"),
            x,
        )
    }

    // MLP layer for Granite 4 models.
    pub fn shared_mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let gate_up = linear_no_bias(
            builder,
            config.hidden_size,
            2 * config.intermediate_size,
            &format!("{name}.input_linear"),
            x,
        );

        let gate_up = chunk(builder, 2, 2, gate_up);
        let gate = gate_up[0].clone();
        let up = gate_up[1].clone();
        let x = silu(builder, gate) * up; // SwiGLU

        linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.output_linear"),
            x,
        )
    }
    pub fn moe(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let moe_input_type = NdArrayType::new(
            Shape(vec![
                config.num_local_experts,
                2 * config.intermediate_size,
                config.hidden_size,
            ]),
            x.label.dtype,
        );
        let moe_input = parameter(
            builder,
            moe_input_type,
            format!("{name}.input_linear.weight"),
        );

        let moe_output_type = NdArrayType::new(
            Shape(vec![
                config.num_local_experts,
                config.hidden_size,
                config.intermediate_size,
            ]),
            x.label.dtype,
        );
        let moe_output = parameter(
            builder,
            moe_output_type,
            format!("{name}.output_linear.weight"),
        );

        let seq_len = x.label.shape.0[1];

        let routed = linear_no_bias(
            builder,
            config.hidden_size,
            config.num_local_experts,
            &format!("{name}.router.layer"),
            x.clone(),
        );

        let vi = topk(builder, config.num_experts_per_tok, routed);
        let values = vi[0].clone();
        let indices = vi[1].clone();

        let indices = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            indices,
        );

        let values = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            values,
        );

        let values = softmax(builder, values);
        let mut xs = x.label.shape.0.clone();
        xs[1] = 0;

        let sumk_type = NdArrayType::new(Shape(xs), x.label.dtype);
        let mut sumk_all = Var::new(builder.clone(), sumk_type);
        let fullx = x;
        for s in 0..seq_len {
            let x = get(builder, 1, s, fullx.clone());
            let mut sumk = constant(builder, x.label.clone(), 0.0);

            let idx = get(builder, 0, s, indices.clone());
            let idx = squeeze(builder, 0, idx);
            let val = get(builder, 0, s, values.clone());

            for i in 0..config.num_experts_per_tok {
                let n = get(builder, 0, i, idx.clone());

                let gate_up = index(builder, 0, moe_input.clone(), n.clone());
                let out = index(builder, 0, moe_output.clone(), n.clone());

                let gate_up = chunk(builder, 1, 2, gate_up);
                let gate = gate_up[0].clone();
                let up = gate_up[1].clone();

                let gate = linear_w(
                    builder,
                    config.hidden_size,
                    config.intermediate_size,
                    gate,
                    x.clone(),
                );

                let up = linear_w(
                    builder,
                    config.hidden_size,
                    config.intermediate_size,
                    up,
                    x.clone(),
                );
                let x = silu(builder, gate) * up; // SwiGLU

                let x = linear_w(
                    builder,
                    config.intermediate_size,
                    config.hidden_size,
                    out,
                    x.clone(),
                );

                let v = select(builder, 1, i, val.clone());
                let v = expand(builder, x.label.shape.clone(), v);
                sumk = sumk + x * v;
            }
            sumk_all = concat(builder, 1, sumk_all, sumk);
        }
        sumk_all
    }

    pub fn layer(
        builder: &Builder,
        layer_id: usize,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.input_layernorm"),
            x,
        );
        let x = Model::attention(
            builder,
            layer_id,
            config,
            cache,
            pos,
            &format!("{name}.self_attn"),
            x,
        );

        let mul = constant(builder, x.label.clone(), config.residual_multiplier);

        let x = res + x * mul.clone();
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.post_attention_layernorm"),
            x,
        );
        let x = if config.num_experts_per_tok == 0 {
            if config.model_type == "granitemoehybrid" {
                Model::shared_mlp(builder, config, &format!("{name}.shared_mlp"), x)
            } else {
                Model::mlp(builder, config, &format!("{name}.mlp"), x)
            }
        } else {
            Model::moe(builder, config, &format!("{name}.block_sparse_moe"), x)
        };
        res + x * mul
    }
}
