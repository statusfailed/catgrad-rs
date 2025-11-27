use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad_mlir::{compile::CompiledModel, runtime::LlvmRuntime};

use anyhow::Result;
use std::collections::HashMap;

use catgrad_llm::helpers::*;
use catgrad_llm::models::utils::Config;
use catgrad_llm::utils::get_model_files;
use tokenizers::tokenizer::Tokenizer;

use nn::*;

pub fn main() -> Result<()> {
    let (param_values, parameters, config, _tokenizer) =
        load_model("HuggingFaceTB/SmolLM2-135M-Instruct")?;

    ////////////////////////////////////////
    // Setup model and environment
    let model = LlamaModel {
        config,
        max_sequence_length: 10, //todo
    };

    let typed_term = model.term().expect("Failed to create typed term");

    // TODO: this is a lot of work for the user...?
    let mut env = stdlib();
    env.definitions.extend([(model.path(), typed_term)]);
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    ////////////////////////////////////////
    // Compile and set up runtime with compiled code
    println!("Compiling {}...", model.path());
    let compiled_model = CompiledModel::new(&env, &parameters, model.path());

    ////////////////////////////////////////
    // Execute with example data
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 10.0, 11.0, 12.0, // 2
    ];

    let input_tensor = LlvmRuntime::tensor(input_data, vec![3, 4], vec![4, 1]);
    println!("Input tensor: {}", input_tensor);

    // Call the function using the CompiledModel API
    let prefix = Path::new(["llama"]).unwrap();
    let param_values = param_values
        .into_iter()
        .map(|(k, v)| (prefix.concat(&k), v))
        .collect();

    println!("calling...");
    let results = compiled_model.call(model.path(), &param_values, vec![input_tensor])?;

    // Print each result using Display
    for (i, result) in results.iter().enumerate() {
        println!("Output tensor {}: {}", i, result);
    }

    Ok(())
}

fn prefix_product(xs: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(xs.len() + 1);
    let mut acc = 1;
    out.push(acc);
    for &x in xs {
        acc *= x;
        out.push(acc);
    }
    out
}

fn load_model(
    model_name: &str,
) -> Result<(
    HashMap<Path, catgrad_mlir::runtime::MlirValue>,
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

            let strides: Vec<usize> = prefix_product(&shape);
            let tensor = LlvmRuntime::tensor(data, shape.clone(), strides);
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
        data_map,
        typecheck::Parameters::from(type_map),
        config,
        tokenizer,
    ))
}

pub struct LlamaModel {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl LlamaModel {
    pub fn embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let wte = param(
            builder,
            &p.extend(vec!["model", "embed_tokens", "weight"]).unwrap(),
        );

        // flatten::<N>()
        let [d0, d1] = unpack::<2>(builder, shape(builder, x.clone()));
        let shape = pack::<1>(builder, [d0 * d1]);
        let x = reshape(builder, shape, x);

        let te = index(builder, 0, x, wte);
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
        let x = silu(builder, gate) * up;
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
            pos,
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            q,
        );
        let k = apply_rope_embedding(
            builder,
            pos,
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

        let mask = causal_mask(builder, s.clone());
        let mask = broadcast(builder, mask, sh);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["o_proj"]).unwrap(), attn)
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
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
            layer_id,
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
    pos: impl IntoNatVar,
    head_dim: usize,
    cos: Var,
    sin: Var,
    x: Var,
) -> Var {
    let sh = shape(builder, x.clone());
    let [_, _, seq_len, _] = unpack::<4>(builder, sh.clone());
    let pos = pos.to_nat(builder);
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
    seq_len: &impl IntoNatVar,
    head_dim: usize,
    x: Var,
) -> Var {
    let pos = pos.to_nat(builder);
    let seq_len = seq_len.to_nat(builder);
    let (cos, sin) = rope_tables(builder, theta, pos.clone() + seq_len, head_dim);

    apply_rope_embedding(builder, pos, head_dim, cos, sin, x)
}

pub fn causal_mask(builder: &Builder, size: Var) -> Var {
    let i = arange(builder, size.clone());
    let sh = pack::<2>(builder, [size.clone(), size.clone()]);
    let i = broadcast(builder, i, sh.clone());

    let one = 1.to_nat(builder);
    let shr = pack::<2>(builder, [size.clone(), one]);
    let j = arange(builder, size);
    let j = reshape(builder, shr, j);
    let j = broadcast(builder, j, sh);

    let mask = lt(builder, j, i);

    let mask = cast(builder, mask, Dtype::F32);
    let sh = shape(builder, mask.clone());
    let ninf = constant(builder, f32::MIN, &sh);

    mask * ninf
}

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
