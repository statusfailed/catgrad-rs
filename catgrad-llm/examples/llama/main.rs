use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::{chunk, linear_no_bias, rmsnorm, unsqueeze};
use std::io::Write;

use std::collections::HashMap;

use catgrad_llm::models::utils::Config;
use catgrad_llm::utils::get_model_files;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tokenizers::tokenizer::Tokenizer;

mod gpt2;
#[derive(Parser, Debug)]
struct Args {
    /// Model name on Huggingface Hub
    #[arg(
        short = 'm',
        long,
        default_value = "HuggingFaceTB/SmolLM2-135M-Instruct"
    )]
    model_name: String,
    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Category theory is")]
    prompt: String,
    /// Tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    seq_len: usize,
    /// Enable typecheck
    #[arg(short = 't', long)]
    typecheck: bool,
    /// Backend to use
    #[arg(short = 'b', long, value_enum, default_value_t = BackendChoice::Ndarray)]
    backend: BackendChoice,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Candle,
}

/// Construct, shapecheck, and interpret the `GPT2Model` using the selected backend.
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    match args.backend {
        BackendChoice::Ndarray => run_with_backend(&args, NdArrayBackend),
        BackendChoice::Candle => run_with_backend(&args, CandleBackend::new()),
    }
}

fn run_with_backend<B: interpreter::Backend>(args: &Args, backend: B) -> Result<()> {
    let (interpreter_params, parameters, config, tokenizer) =
        load_model(&args.model_name, &backend)?;

    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    let model: Box<dyn Module<1, 1>> = if config.architectures[0].as_str() == "LlamaForCausalLM" {
        Box::new(LlamaModel {
            config: config.clone(),
            max_sequence_length: args.seq_len + token_ids.len(),
        })
    } else {
        Box::new(gpt2::GPT2Model {
            config: config.clone(),
        })
    };

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Shapecheck the model
    if args.typecheck {
        typecheck::check(&env, &parameters, typed_term.clone())
            .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;
    }

    print!("{}", args.prompt);
    let start_gen = std::time::Instant::now();
    let interpreter = interpreter::Interpreter::new(backend, env, interpreter_params);
    // Run interpreter
    for _ in 0..args.seq_len {
        let next_token_id = run_interpreter(&typed_term, &interpreter, &token_ids)?;
        if config.get_eos_token_ids().contains(&(next_token_id as i32)) {
            break;
        }
        let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
        token_ids.push(next_token_id);
        print!("{}", decoded_token);
        std::io::stdout().flush()?;
    }

    let elapsed_gen = start_gen.elapsed();
    let generated_tokens = args.seq_len;
    println!(
        "\n{} tokens generated in {} seconds. ({:.2} tps)",
        generated_tokens,
        elapsed_gen.as_secs(),
        generated_tokens as f64 / elapsed_gen.as_secs_f64(),
    );
    Ok(())
}

fn run_interpreter<B: interpreter::Backend>(
    typed_term: &TypedTerm,
    interpreter: &interpreter::Interpreter<B>,
    input_data: &[u32],
) -> Result<u32> {
    let input_tensor = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, input_data.len()]),
        input_data,
    )
    .expect("Failed to create input tensor");

    // Run the model
    let mut results = interpreter
        .run(typed_term.term.clone(), vec![input_tensor])
        .expect("Failed to run inference");

    // Print info about the main output (should be the last one)
    if let Some(output) = results.pop() {
        match output {
            interpreter::Value::Tensor(arr) => match interpreter.backend.to_vec(arr) {
                interpreter::TaggedVec::U32(v) => Ok(v[v.len() - 1]),
                _ => Err(anyhow::anyhow!("Unexpected output dtype")),
            },
            t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
        }
    } else {
        Err(anyhow::anyhow!("No result"))
    }
}

pub fn repeat_kv(builder: &Builder, rep: usize, x: Var) -> Var {
    let shape = shape(builder, x.clone());
    let [b, num_kv_heads, s, head_dim] = unpack::<4>(builder, shape);

    let sh = shape!(builder, b, num_kv_heads, 1, s, head_dim);
    // equivalent of torch.repeat_interleave across dim 1
    let x = reshape(builder, sh, x);
    let sh = shape!(builder, b, num_kv_heads, rep, s, head_dim);

    let x = broadcast(builder, x, sh);

    let rnkv = num_kv_heads * rep.to_nat(builder);
    let sh = shape!(builder, b, rnkv, s, head_dim);
    reshape(builder, sh, x)
}

// Generate rope tables. This part is usually precomputed
pub fn rope_tables(builder: &Builder, theta: f32, seq_len: Var, head_dim: usize) -> (Var, Var) {
    let half_dim = head_dim / 2;

    let f = arange(builder, half_dim);
    let f = cast(builder, f, Dtype::F32);
    let sh = shape(builder, f.clone());
    let two = constant(builder, 2.0 / (head_dim as f32), &sh);
    let f = f * two;
    let theta = constant(builder, theta, &sh);
    let freq = pow(builder, theta, f);
    let inv_freq = inverse(builder, freq);

    let sh = shape!(builder, seq_len, half_dim);
    let inv_freq = broadcast(builder, inv_freq, sh);

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, pos, sh);
    let pos = pos * inv_freq;
    let cos = cos(builder, pos.clone());
    let sin = sin(builder, pos);

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

fn rotate_half(builder: &Builder, head_dim: usize, x: Var) -> Var {
    let v = chunk(builder, 3, 2, head_dim / 2, x);

    concat(builder, 3, -v[1].clone(), v[0].clone())
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by reusing calculated tables
pub fn apply_rope_embedding(
    builder: &Builder,
    pos: Var,
    head_dim: usize,
    cos: Var,
    sin: Var,
    x: Var,
) -> Var {
    let sh = shape(builder, x.clone());
    let [_, _, seq_len] = unpack::<3>(builder, sh.clone());
    let cos = slice(builder, 0, pos.clone(), seq_len.clone(), cos);
    let sin = slice(builder, 0, pos, seq_len, sin);
    let cos = broadcast(builder, cos, sh.clone());
    let sin = broadcast(builder, sin, sh);

    let rotated_x = rotate_half(builder, head_dim, x.clone());

    cos * x + sin * rotated_x
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by calculating the tables
pub fn rope(
    builder: &Builder,
    theta: f32,
    pos: impl IntoNatVar,
    seq_len: impl IntoNatVar,
    head_dim: usize,
    x: Var,
) -> Var {
    let pos = pos.to_nat(builder);
    let seq_len = seq_len.to_nat(builder);
    let (cos, sin) = rope_tables(builder, theta, pos.clone() + seq_len, head_dim);

    apply_rope_embedding(builder, pos, head_dim, cos, sin, x)
}

/// Type signature for LLM Modules
fn llm_type() -> ([Type; 1], [Type; 1]) {
    use catgrad::typecheck::*;
    let batch_size = NatExpr::Var(0);
    let seq_len = NatExpr::Var(1);

    // Input shape B×S
    let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len]),
    }));

    // Output shape B×1
    let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(1)]),
    }));

    ([t_x], [t_y])
}

struct Cache {
    pub cos: Var,
    pub sin: Var,
}

impl Cache {
    pub fn init(builder: &Builder, config: &Config, positions: usize) -> Self {
        let (cos, sin) = rope_tables(
            builder,
            config.rope_theta,
            positions.to_nat(builder),
            config.get_head_dim(),
        );

        Self { cos, sin }
    }
}

pub struct LlamaModel {
    config: Config,
    max_sequence_length: usize,
}

impl LlamaModel {
    pub fn info(&self) {
        println!("Config: {:#?}", self.config);
    }

    pub fn embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let wte = param(
            builder,
            &p.extend(vec!["model", "embed_tokens", "weight"]).unwrap(),
        );
        let dim = 0.to_nat(builder);
        let te = index(builder, wte, dim, x);

        unsqueeze::<2, 3>(builder, 0, te)
    }

    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let gate = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            p.extend(["gate_proj"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            p.extend(["up_proj"]).unwrap(),
            x,
        );
        let x = nn::silu(builder, gate) * up;
        linear_no_bias(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    fn attention(
        &self,
        builder: &Builder,
        _layer_id: usize,
        cache: &mut Cache,
        pos: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.hidden_size / num_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());

        let k = linear_no_bias(
            builder,
            dim,
            dim / rep,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );

        let v = linear_no_bias(builder, dim, dim / rep, p.extend(["v_proj"]).unwrap(), x);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);

        let v = reshape(builder, sh, v);
        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let q = apply_rope_embedding(
            builder,
            pos.to_nat(builder),
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            q,
        );
        let k = apply_rope_embedding(
            builder,
            pos.to_nat(builder),
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            k,
        );

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = nn::causal_mask(builder, s.clone());
        let mask = broadcast(builder, mask, sh);
        attn = attn + mask;

        let attn = nn::softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["o_proj"]).unwrap(), attn)
    }

    fn layer(
        &self,
        builder: &Builder,
        _layer_id: usize,
        cache: &mut Cache,
        pos: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = self.attention(
            builder,
            _layer_id,
            cache,
            pos,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = res + x;
        let res = x.clone();
        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        x + res
    }
}

impl Module<1, 1> for LlamaModel {
    fn path(&self) -> Path {
        path(vec!["llama"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let root = self.path();

        let mut cache = Cache::init(builder, &self.config, self.max_sequence_length);

        let mut x = self.embeddings(builder, root.clone(), x);

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                &mut cache,
                0,
                root.extend(["model", "layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            root.extend(["model", "norm"]).unwrap(),
            x,
        );

        let lm_head_weights = if self.config.tie_word_embeddings {
            vec!["model", "embed_tokens"]
        } else {
            vec!["lm_head"]
        };

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(lm_head_weights).unwrap(),
            x,
        );

        x = argmax(builder, x);
        [x]
    }

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        llm_type()
    }
}

fn load_model<B: interpreter::Backend>(
    model_name: &str,
    backend: &B,
) -> Result<(
    interpreter::Parameters<B>,
    typecheck::Parameters,
    Config,
    Tokenizer,
)> {
    let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, "main")?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| anyhow::anyhow!("tokenizer load error {:?}", err))?;

    // Read each tensor
    let mut type_map = HashMap::new();
    let mut data_map = HashMap::new();

    for file_path in model_paths {
        let file = std::fs::File::open(file_path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&data)?;

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let tensor_data = view.data();

            use catgrad::typecheck::*;
            // Convert dtype and load tensor data
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => tensor_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                safetensors::Dtype::BF16 => tensor_data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                _ => {
                    panic!("Unsupported dtype: {:?}", view.dtype());
                }
            };

            let tensor = interpreter::tensor(backend, Shape(shape.clone()), &data)
                .expect("failed to create tensor");
            let key = path(name.split(".").collect()).expect("invalid param path");
            data_map.insert(key.clone(), tensor);

            let vne = shape.into_iter().map(NatExpr::Constant).collect();
            let tensor_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vne),
            }));
            type_map.insert(key, tensor_type);
        }
    }

    Ok((
        interpreter::Parameters::from(data_map),
        typecheck::Parameters::from(type_map),
        config,
        tokenizer,
    ))
}
