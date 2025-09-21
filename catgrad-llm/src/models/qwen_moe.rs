// Qwen3 MoE model description

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

        let lm_head_weights = if config.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        };

        // Use weight tying
        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            lm_head_weights,
            result,
        )
    }
}

impl Model {
    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType::new(
            Shape(vec![config.vocab_size, config.hidden_size]),
            config.dtype,
        );
        let weights = parameter(builder, t, "model.embed_tokens.weight".to_string());
        embedding(builder, x, weights)
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
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = config.get_head_dim();
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear_no_bias(
            builder,
            config.hidden_size,
            num_heads * head_dim,
            &format!("{name}.q_proj"),
            x.clone(),
        );
        let k = linear_no_bias(
            builder,
            config.hidden_size,
            num_kv_heads * head_dim,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_no_bias(
            builder,
            config.hidden_size,
            num_kv_heads * head_dim,
            &format!("{name}.v_proj"),
            x,
        );

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        // Norm
        let q = reshape(builder, Shape(vec![b * s * num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b * s * num_kv_heads, head_dim]), k);
        let q = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.q_norm"), q);
        let k = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.k_norm"), k);
        let q = reshape(builder, Shape(vec![b, num_heads, s, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, num_kv_heads, s, head_dim]), k);

        let q = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let k = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let attn = attn / denom;

        let mask = causal_mask(builder, s, attn.label.dtype);
        let mask = expand(builder, attn.label.shape.clone(), mask);
        let attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape(builder, Shape(vec![b, s, num_heads * head_dim]), x);
        linear_no_bias(
            builder,
            num_heads * head_dim,
            config.hidden_size,
            &format!("{name}.o_proj"),
            x,
        )
    }

    pub fn expert(builder: &Builder, config: &Config, name: &str, n: Var, x: Var) -> Var {
        let gate_up_type = NdArrayType::new(
            Shape(vec![config.moe_intermediate_size, config.hidden_size]),
            x.label.dtype,
        );

        let gate_p = parameter_dynamic(
            builder,
            gate_up_type.clone(),
            n.clone(),
            format!("{name}.gate_proj.weight"),
        );
        let gate = linear_no_bias_param(
            builder,
            config.hidden_size,
            config.moe_intermediate_size,
            gate_p,
            x.clone(),
        );

        let up_p = parameter_dynamic(
            builder,
            gate_up_type,
            n.clone(),
            format!("{name}.up_proj.weight"),
        );
        let up = linear_no_bias_param(
            builder,
            config.hidden_size,
            config.moe_intermediate_size,
            up_p,
            x,
        );
        let x = silu(builder, gate) * up; // SwiGLU

        let down_type = NdArrayType::new(
            Shape(vec![config.hidden_size, config.moe_intermediate_size]),
            x.label.dtype,
        );

        let down_p = parameter_dynamic(builder, down_type, n, format!("{name}.down_proj.weight"));
        linear_no_bias_param(
            builder,
            config.moe_intermediate_size,
            config.hidden_size,
            down_p,
            x,
        )
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

    pub fn moe(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let seq_len = x.label.shape.0[1];

        let routed = linear_no_bias(
            builder,
            config.hidden_size,
            config.num_local_experts,
            &format!("{name}.gate"),
            x.clone(),
        );
        let routed = softmax(builder, routed);

        let vi = topk(builder, config.num_experts_per_tok, routed);
        let values = vi[0].clone();
        let indices = vi[1].clone();

        let indices = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            indices,
        );

        let mut values = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            values,
        );

        if config.norm_topk_prob {
            let sv = sum(builder, values.clone());
            let sv = expand(builder, values.label.shape.clone(), sv);
            values = values / sv;
        }

        let mut xs = x.label.shape.0.clone();
        xs[1] = 0;
        let sumk_type = NdArrayType::new(Shape(xs), x.label.dtype);
        let mut sumk_all = Var::new(builder.clone(), sumk_type);
        let fullx = x;
        for s in 0..seq_len {
            let x = get(builder, 1, s, fullx.clone());
            let idx = get(builder, 0, s, indices.clone());
            let val = get(builder, 0, s, values.clone());
            let mut sumk = constant(builder, x.label.clone(), 0.0);
            for i in 0..config.num_experts_per_tok {
                let n = select(builder, 1, i, idx.clone());
                let x = Model::expert(
                    builder,
                    config,
                    &format!("{name}.experts.{{}}"),
                    n,
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

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.post_attention_layernorm"),
            x,
        );
        let moe_layer = config.num_local_experts > 1
            && (layer_id + 1).is_multiple_of(config.decoder_sparse_step);
        let x = if moe_layer {
            Model::moe(builder, config, &format!("{name}.mlp"), x)
        } else {
            Model::mlp(builder, config, &format!("{name}.mlp"), x)
        };
        res + x
    }
}
