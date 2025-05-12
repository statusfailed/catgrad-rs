// Example GPT-2 model inference
// WIP
// Model used for testing https://huggingface.co/openai-community/gpt2

use clap::Parser;
use env_logger;
use serde;
use serde_json;
use std::path::PathBuf;
use tokenizers::tokenizer::{Result, Tokenizer};

use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{
        nn::{
            layers::{
                arange, causal_mask, constant, embedding, expand, gelu, layernorm, linear_no_bias,
                mat_mul, parameter, reshape, softmax, transpose,
            },
            utils::read_safetensors,
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
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_positions: usize,
    pub layer_norm_epsilon: f32,
    pub vocab_size: usize,
}

#[derive(Debug)]
struct Model {
    pub state: EvalState,
}

pub fn layer(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
    let res = x.clone();
    let x = layernorm(
        &builder,
        config.layer_norm_epsilon,
        &format!("{name}.ln_1"),
        x,
    );
    let x = attention(builder, config, &format!("{name}.attn"), x);
    let x = res + x;
    let res = x.clone();
    let x = layernorm(
        &builder,
        config.layer_norm_epsilon,
        &format!("{name}.ln_2"),
        x,
    );
    let x = mlp(builder, config.n_embd, &format!("{name}.mlp"), x);
    x + res
}

// The original GPT2 checkpoints use a Conv1D layer instead of linear,
// equivalent to a linear layer with weights in transposed order
fn gpt_linear(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    let w_type = NdArrayType {
        shape: Shape(vec![in_dim, out_dim]),
        dtype: x.label.dtype,
    };
    // Bias
    let b_type = NdArrayType {
        shape: Shape(vec![out_dim]),
        dtype: x.label.dtype,
    };

    let w = parameter(builder, w_type.clone(), format!("{name}.weight"));
    let b = parameter(builder, b_type.clone(), format!("{name}.bias"));

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
    let t = NdArrayType {
        shape: Shape(vec![config.vocab_size, config.n_embd]),
        dtype: Dtype::F32,
    };
    let weights = parameter(builder, t, format!("wte.weight"));
    let we = embedding(builder, x.clone(), weights);

    let t = NdArrayType {
        shape: Shape(vec![config.n_positions, config.n_embd]),
        dtype: Dtype::F32,
    };
    let pos = arange(&builder, x.label.clone());
    let weights = parameter(builder, t, format!("wpe.weight"));
    let pe = embedding(builder, pos, weights);

    we + pe
}

pub fn attention(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
    let dim = config.n_embd;
    let num_heads = config.n_head;
    let head_dim = dim / num_heads;

    let b = x.clone().label.shape.0[0];
    let s = x.clone().label.shape.0[1];

    // let c_attn = gpt_linear(builder, dim, 3 * dim, &format!("{name}.c_attn"), x.clone());
    let k = gpt_linear(builder, dim, dim, &format!("{name}.key"), x.clone());
    let q = gpt_linear(builder, dim, dim, &format!("{name}.query"), x.clone());
    let v = gpt_linear(builder, dim, dim, &format!("{name}.value"), x.clone());

    let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
    let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
    let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

    let q = transpose(builder, 1, 2, q);
    let k = transpose(builder, 1, 2, k);
    let v = transpose(builder, 1, 2, v);

    let tk = transpose(builder, 2, 3, k);
    let attn = mat_mul(builder, q.clone(), tk);
    let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
    let attn = attn / denom;

    let mask = causal_mask(builder, s);
    let mask = expand(builder, Shape(vec![b, num_heads, s, s]), mask);
    let attn = attn + mask;

    let attn = softmax(builder, attn);
    let attn = mat_mul(builder, attn, v);

    let attn = transpose(builder, 1, 2, attn);
    let attn = reshape(builder, Shape(vec![b, s, dim]), attn);

    let c_proj = gpt_linear(builder, dim, dim, &format!("{name}.c_proj"), attn);
    c_proj
}

pub fn mlp(builder: &Builder, dim: usize, name: &str, x: Var) -> Var {
    let x = gpt_linear(builder, dim, dim * 4, &format!("{name}.c_fc"), x);
    let x = gelu(builder, x);
    let x = gpt_linear(builder, dim * 4, dim, &format!("{name}.c_proj"), x);
    x
}

impl Model {
    pub fn build(batches: usize, tokens: usize, config: &Config) -> Self {
        let in_type = NdArrayType {
            shape: Shape(vec![batches, tokens]),
            dtype: Dtype::I32,
        };

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let emb = embeddings(&builder, config, x.clone());

            let mut result = emb;

            for i in 0..config.n_layer {
                result = layer(&builder, &config, &format!("h.{i}"), result);
            }

            result = layernorm(
                &builder,
                config.layer_norm_epsilon,
                &format!("ln_f"),
                result,
            );

            // GPT-2 uses weight tying so lm_head is the same as wte
            let lm_head = linear_no_bias(&builder, config.n_embd, config.vocab_size, "wte", result);
            (vec![x], vec![lm_head])
        });

        Self { state }
    }

    fn load(&self, name: &str) -> std::collections::HashMap<String, TaggedNdArray> {
        let mut tensors = read_safetensors(name);

        // TODO: This is needed until there is an operator that slices qkv at runtime.
        // Find all the c_attn weight and bias keys
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

        tensors
    }

    pub fn run(&mut self, x: &NdArray<i32>, model_path: &str) -> TaggedNdArray {
        let tensors = self.load(model_path);
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
    #[arg(short = 'm', long, default_value = "gpt.safetensors")]
    model_path: String,

    /// Number of batches
    #[arg(short = 'b', long, default_value_t = 1)]
    batches: usize,

    /// Number of tokens per sequence
    #[arg(short = 't', long, default_value_t = 1)]
    tokens: usize,

    /// Value to fill input tensor with for testing
    #[arg(short = 'f', long, default_value_t = 0)]
    fill: usize,

    /// Initial prompt
    #[arg(short = 'p', long)]
    prompt: Option<String>,
}

fn get_config(model_path: &str) -> Config {
    let mut model_path = PathBuf::from(model_path);
    if let Ok(link) = model_path.read_link() {
        model_path = link;
    }

    let config_dir = model_path.parent().unwrap();
    let config_path = config_dir.join("config.json");
    serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap()
}

pub fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let config = get_config(&args.model_path);

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

    let mut input = NdArray::new(iv, Shape(vec![batches, tokens]));
    if let Some(prompt) = args.prompt {
        let tokenizer = Tokenizer::from_pretrained("openai-community/gpt2", None)?;
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
    println!("Result shape: {:?}", result.shape());
    // Print just the first values for each token for debugging purposes
    let v = config.vocab_size;
    for t in 0..tokens {
        println!("Token {t}: {:?}", &result.data()[t * v..t * v + 10]);
    }
    Ok(())
}
