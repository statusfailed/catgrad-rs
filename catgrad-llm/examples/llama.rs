use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::{chunk, linear_no_bias, rmsnorm, unsqueeze};

use std::collections::HashMap;

use catgrad_llm::models::utils::Config;
use catgrad_llm::utils::get_model_files;

use anyhow::Result;

/// Construct, shapecheck, and interpret the `GPT2Model` using the ndarray backend.
fn main() -> Result<()> {
    // Create parameters for the model
    let backend = NdArrayBackend;
    let (interpreter_params, parameters, config) =
        load_model("HuggingFaceTB/SmolLM2-135M-Instruct", &backend)?;

    let model = LlamaModel { config };

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Shapecheck the model
    typecheck::check(&env, &parameters, typed_term.clone())
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    // Run interpreter
    run_interpreter(&typed_term, env, interpreter_params)?;

    Ok(())
}

fn run_interpreter(
    typed_term: &TypedTerm,
    env: Environment,
    interpreter_params: interpreter::Parameters<NdArrayBackend>,
) -> Result<()> {
    let backend = NdArrayBackend;

    // Create interpreter
    let interpreter = interpreter::Interpreter::new(backend, env, interpreter_params);

    // Llama encoding for 'Category theory is'
    let input_data = [27348, 3108, 314];
    let input_tensor = interpreter::tensor(&interpreter.backend, Shape(vec![1, 3]), &input_data)
        .expect("Failed to create input tensor");

    // Run the model
    let results = interpreter
        .run(typed_term.term.clone(), vec![input_tensor])
        .expect("Failed to run inference");

    // Print info about the main output (should be the last one)
    if let Some(output) = results.last() {
        use catgrad::interpreter::{TaggedTensor, Value};
        match output {
            Value::Tensor(TaggedTensor::U32([arr])) => {
                println!("Output shape: {:?}", arr.shape());
                println!(
                    "Output sample: {:?}",
                    &arr.as_slice().unwrap()[..10.min(arr.len())]
                );
            }
            _ => println!("Unexpected output type: {:?}", output),
        }
    }

    Ok(())
}

pub fn repeat_kv(builder: &Builder, rep: usize, x: Var) -> Var {
    let shape = shape(builder, x.clone());
    let [b, num_kv_heads, s, head_dim] = unpack::<4>(builder, shape);

    let sh = shape!(builder, b, num_kv_heads, 1, s, head_dim);
    // equivalent of torch.repeat_interleave across dim 1
    let x = reshape(builder, sh, x);
    let sh = shape!(builder, b, num_kv_heads, rep, s, head_dim);

    let x = broadcast(builder, x, sh);

    let rnkv = num_kv_heads * lit(builder, nat(rep as u32));
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
    let pos = pos.to_var(builder);
    let seq_len = seq_len.to_var(builder);
    let (cos, sin) = rope_tables(builder, theta, pos.clone() + seq_len, head_dim);

    apply_rope_embedding(builder, pos, head_dim, cos, sin, x)
}

pub struct LlamaModel {
    config: Config,
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
        let dim = lit(builder, nat(0));
        let te = index(builder, wte, dim, x);

        unsqueeze::<2, 3>(builder, 0, te)
    }

    fn mlp(&self, builder: &Builder, config: &Config, p: Path, x: Var) -> Var {
        let gate = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size,
            p.extend(["gate_proj"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size,
            p.extend(["up_proj"]).unwrap(),
            x,
        );
        let x = nn::silu(builder, gate) * up;
        linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    fn attention(
        &self,
        builder: &Builder,
        _layer_id: usize,
        config: &Config,
        pos: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads;
        let num_kv_heads = config.num_key_value_heads;
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

        let q = rope(builder, config.rope_theta, pos, s.clone(), head_dim, q);
        let k = rope(builder, config.rope_theta, pos, s.clone(), head_dim, k);

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
        config: &Config,
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
            &self.config,
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
        let x = self.mlp(builder, config, p.extend(["mlp"]).unwrap(), x);
        x + res
    }
}

impl Module<1, 1> for LlamaModel {
    fn path(&self) -> Path {
        path(vec!["llama"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let root = self.path();

        // self.info();

        let mut x = self.embeddings(builder, root.clone(), x);

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                &self.config,
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

        [x]
    }

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use catgrad::typecheck::*;
        let batch_size = NatExpr::Var(0);
        let seq_len = NatExpr::Var(1);

        // Input shape B×S
        let t_x = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len]),
        }));

        // Output shape B×1
        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(1)]),
        }));

        ([t_x], [t_y])
    }
}

fn load_model<B: interpreter::Backend>(
    model_name: &str,
    backend: &B,
) -> Result<(interpreter::Parameters<B>, typecheck::Parameters, Config)> {
    let (model_paths, config_path, _tokenizer_path, _) = get_model_files(model_name, "main")?;
    // let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let file = std::fs::File::open(&model_paths[0])?;
    let data = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    // Read each tensor
    let mut type_map = HashMap::new();
    let mut data_map = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();
        let tensor_data = view.data();

        // Convert dtype and load tensor data
        match view.dtype() {
            safetensors::Dtype::BF16 => {
                use catgrad::typecheck::*;
                let data: Vec<f32> = tensor_data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect();
                let tensor =
                    interpreter::TaggedTensor::from_slice(backend, &data, Shape(shape.clone()))
                        .expect("Failed to create tensor");
                let key = path(name.split(".").collect()).expect("invalid param path");
                data_map.insert(key.clone(), tensor);

                let vne = shape.into_iter().map(NatExpr::Constant).collect();
                let tensor_type = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                    dtype: DtypeExpr::Constant(Dtype::F32),
                    shape: ShapeExpr::Shape(vne),
                }));
                type_map.insert(key, tensor_type);
            }
            // Add other dtype conversions as needed
            _ => {
                panic!("Unsupported dtype: {:?}", view.dtype());
            }
        }
    }
    Ok((
        interpreter::Parameters::from(data_map),
        typecheck::Parameters::from(type_map),
        config,
    ))
}
