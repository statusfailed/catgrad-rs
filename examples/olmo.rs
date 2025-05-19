// Example OLMO-2 model inference

use clap::Parser;
use env_logger;
use serde;
use serde_json;
use tokenizers::tokenizer::{Result, Tokenizer};

use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{
        nn::{
            layers::{
                causal_mask, constant, embedding, expand, linear_no_bias, mat_mul, parameter,
                reshape, rmsnorm, silu, softmax, transpose,
            },
            utils::{get_model_files, read_safetensors},
        },
        Dtype, NdArrayType, Shape, Var,
    },
};

#[allow(unused)]
fn show(name: &str, var: &Var) {
    println!("{name} label: {:?}", var.label,);
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

#[derive(Debug)]
struct Model {
    pub state: EvalState,
}

impl Model {
    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType {
            shape: Shape(vec![config.vocab_size, config.hidden_size]),
            dtype: Dtype::F32,
        };
        let weights = parameter(builder, t, format!("model.embed_tokens.weight"));
        embedding(builder, x.clone(), weights)
    }

    pub fn attention(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;
        let b = x.clone().label.shape.0[0];
        let s = x.clone().label.shape.0[1];

        let q = linear_no_bias(builder, dim, dim, &format!("{name}.q_proj"), x.clone());
        let k = linear_no_bias(
            builder,
            dim,
            dim * num_kv_heads / num_heads,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_no_bias(
            builder,
            dim,
            dim * num_kv_heads / num_heads,
            &format!("{name}.v_proj"),
            x.clone(),
        );

        let q = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.q_norm"), q);
        let k = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.k_norm"), k);

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        // Repeat KV
        // let k = expand(builder, Shape(vec![b, num_heads, s, head_dim]), k);
        // let v = expand(builder, Shape(vec![b, num_heads, s, head_dim]), v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q.clone(), tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let attn = attn / denom;

        let mask = causal_mask(builder, s);
        let mask = expand(builder, Shape(vec![b, num_heads, s, s]), mask);
        let attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape(builder, Shape(vec![b, s, dim]), x);
        let o_proj = linear_no_bias(builder, dim, dim, &format!("{name}.o_proj"), x);
        o_proj
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
        let x = linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.down_proj"),
            x,
        );
        x
    }

    pub fn layer(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let res = x.clone();
        let x = Model::attention(builder, config, &format!("{name}.self_attn"), x);
        let x = rmsnorm(
            &builder,
            config.rms_norm_eps,
            &format!("{name}.post_attention_layernorm"),
            x,
        );
        let x = res + x;

        let res = x.clone();
        let x = Model::mlp(builder, config, &format!("{name}.mlp"), x);
        let x = rmsnorm(
            &builder,
            config.rms_norm_eps,
            &format!("{name}.post_feedforward_layernorm"),
            x,
        );
        x + res
    }

    pub fn build(batches: usize, tokens: usize, config: &Config) -> Self {
        let in_type = NdArrayType {
            shape: Shape(vec![batches, tokens]),
            dtype: Dtype::I32,
        };

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let emb = Model::embeddings(&builder, config, x.clone());

            let mut result = emb;

            for i in 0..config.num_hidden_layers {
                result = Model::layer(&builder, config, &format!("model.layers.{i}"), result);
            }

            result = rmsnorm(
                &builder,
                config.rms_norm_eps,
                &format!("model.norm"),
                result,
            );
            let lm_head = linear_no_bias(
                &builder,
                config.hidden_size,
                config.vocab_size,
                "lm_head",
                result,
            );

            (vec![x], vec![lm_head])
        });

        Self { state }
    }

    pub fn run(&mut self, x: &NdArray<i32>, model_path: &str) -> TaggedNdArray {
        let tensors = read_safetensors(model_path);
        println!("Model weights loaded...");
        self.state.set_parameters(tensors);
        let [result] = self.state.eval_with(vec![x.clone().into()])[..] else {
            panic!("unexpected result")
        };

        result.clone()
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to the safetensors model file
    #[arg(short = 'm', long, default_value = "olmo.safetensors")]
    model_path: String,

    /// Number of batches
    #[arg(short = 'b', long, default_value_t = 1)]
    batches: usize,

    /// Number of tokens per sequence
    #[arg(short = 't', long, default_value_t = 1)]
    tokens: usize,

    /// Value to fill input tensor with
    #[arg(short = 'f', long, default_value_t = 0)]
    fill: usize,

    /// Initial prompt
    #[arg(short = 'p', long)]
    prompt: Option<String>,
}

pub fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let (config_path, tokenizer_path) = get_model_files(&args.model_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let config = serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();

    let mut batches = args.batches;
    let mut tokens = args.tokens;
    let fill = args.fill;

    let iv = if fill != 0 {
        vec![fill as i32; batches * tokens]
    } else {
        (0..batches)
            .flat_map(|_| 0..tokens)
            .map(|x| x as i32)
            .collect()
    };

    // let tokenizer = Tokenizer::from_pretrained("allenai/OLMo-2-0425-1B-Instruct", None)?;
    let mut input = NdArray::new(iv, Shape(vec![batches, tokens]));
    if let Some(prompt) = args.prompt {
        let encoding = tokenizer.encode(prompt, true)?;
        // println!("{:?}", encoding.get_tokens());

        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        tokens = ids.len();
        batches = 1;
        input = NdArray::new(ids.clone(), Shape(vec![1, tokens]));
    }

    println!("Input tokens {:?}", &input);
    let mut model = Model::build(batches, tokens, &config);

    println!("Model graph built...");
    let result = model.run(&input, &args.model_path);
    println!("input {:?}", input);
    println!("Result: {:?}", result.len());
    Ok(())
}
