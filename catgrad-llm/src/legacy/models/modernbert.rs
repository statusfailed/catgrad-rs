// ModernBERT decoder model description

use super::utils::{Cache, Config, ModelBuilder};
use crate::legacy::nn::layers::*;
use crate::legacy::nn::rope::rope;
use catgrad_legacy::backend::cpu::eval::Builder;
use catgrad_legacy::core::{NdArrayType, Shape, Var};

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

        result = layernorm_no_bias(builder, config.layer_norm_eps, "model.final_norm", result);

        // Get the logits for the last token only
        if tokens > 1 {
            result = narrow(builder, 1, tokens - 1, 1, result);
        }

        let result = Model::lm_head(builder, config, result);

        linear(
            builder,
            config.hidden_size,
            config.vocab_size,
            "decoder",
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
        let weights = parameter(builder, t, "decoder.weight".to_string());
        let result = embedding(builder, x, weights);
        layernorm_no_bias(
            builder,
            config.layer_norm_eps,
            "model.embeddings.norm",
            result,
        )
    }

    pub fn lm_head(builder: &Builder, config: &Config, x: Var) -> Var {
        let result = linear_no_bias(
            builder,
            config.hidden_size,
            config.hidden_size,
            "lm_head.dense",
            x,
        );
        let result = gelu(builder, result);
        layernorm_no_bias(builder, config.layer_norm_eps, "lm_head.norm", result)
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
        let head_dim = config.get_head_dim();
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim,
            &format!("{name}.q_proj"),
            x.clone(),
        );
        let k = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim,
            &format!("{name}.v_proj"),
            x,
        );

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let theta = if layer_id.is_multiple_of(config.global_attn_every_n_layers) {
            config.global_rope_theta
        } else {
            config.local_rope_theta
        };

        let q = rope(builder, theta, pos, s, q);
        let k = rope(builder, theta, pos, s, k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

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

        linear_no_bias(builder, num_heads * head_dim, dim, &format!("{name}.Wo"), x)
    }

    pub fn mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let x = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size * 2,
            &format!("{name}.Wi"),
            x,
        );

        let input_gate = chunk(builder, 2, 2, x);
        let input = input_gate[0].clone();
        let gate = input_gate[1].clone();
        let x = gelu(builder, input) * gate;

        linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.Wo"),
            x,
        )
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
        let mut x = x;
        if layer_id != 0 {
            x = layernorm_no_bias(
                builder,
                config.layer_norm_eps,
                &format!("{name}.attn_norm"),
                x,
            );
        }
        let x = Model::attention(
            builder,
            layer_id,
            config,
            cache,
            pos,
            &format!("{name}.attn"),
            x,
        );
        let x = res + x;
        let res = x.clone();
        let x = layernorm_no_bias(
            builder,
            config.layer_norm_eps,
            &format!("{name}.mlp_norm"),
            x,
        );
        let x = Model::mlp(builder, config, &format!("{name}.mlp"), x);
        x + res
    }
}
