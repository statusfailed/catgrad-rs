use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

use std::collections::HashMap;

use catgrad_llm::models::utils::Config;
use catgrad_llm::utils::get_model_files;

use anyhow::Result;

/// Construct, shapecheck, and interpret the `GPT2Model` using the ndarray backend.
fn main() -> Result<()> {
    // Create parameters for the model
    let backend = NdArrayBackend;
    let (interpreter_params, parameters, config) = load_model("openai-community/gpt2", &backend)?;

    let model = GPT2Model { config };

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

    // GPT-2 encoding for 'Category theory is'
    let input_data = [27313, 4583, 318];
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

////////////////////////////////////////////////////////////////////////////////
// Define the GPT2Model model

pub struct GPT2Model {
    config: Config,
}

impl GPT2Model {
    pub fn info(&self) {
        println!("Config: {:#?}", self.config);
    }

    pub fn embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let wte = param(builder, &p.extend(["wte", "weight"]).unwrap());
        let dim = lit(builder, nat(0));
        let te = index(builder, wte, dim.clone(), x);

        // add back batch size dim
        let sh = shape(builder, te.clone());
        let [seq_len, hidden_dim] = unpack::<2>(builder, sh);
        let sh = shape!(builder, 1, seq_len, hidden_dim);

        let te = reshape(builder, sh.clone(), te);

        let wpe = param(builder, &p.extend(["wpe", "weight"]).unwrap());
        let r = arange(builder, seq_len);
        let pe = index(builder, wpe, dim, r);
        let pe = reshape(builder, sh, pe);
        te + pe
    }

    fn gpt_linear(
        &self,
        builder: &Builder,
        _in_dim: usize,
        _out_dim: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let w = param(builder, &p.extend(["weight"]).unwrap());
        let b = param(builder, &p.extend(["bias"]).unwrap());

        // w is already transposed in GPT-2 checkpoints
        let w_t = w;

        let w_t = nn::unsqueeze::<2, 3>(builder, 0, w_t);
        let m = matmul(builder, x, w_t);
        let sh = shape(builder, m.clone());
        let bb = broadcast_to(builder, b, sh);
        m + bb
    }

    fn mlp(&self, builder: &Builder, dim: usize, p: Path, x: Var) -> Var {
        let x = self.gpt_linear(builder, dim, dim * 4, p.extend(["c_fc"]).unwrap(), x);
        // let x = nn::gelu(builder, x);
        let x = nn::Gelu.call(builder, [x]);
        self.gpt_linear(builder, dim * 4, dim, p.extend(["c_proj"]).unwrap(), x)
    }

    fn attention(
        &self,
        builder: &Builder,
        _layer_id: usize,
        config: &Config,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = dim / num_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let c_attn = self.gpt_linear(builder, dim, 3 * dim, p.extend(["c_attn"]).unwrap(), x);

        let a = nn::chunk(builder, 2, 3, config.hidden_size, c_attn);
        let q = a[0].clone();
        let k = a[1].clone();
        let v = a[2].clone();

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh.clone(), q);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        // TODO: check for seqlen > 1
        // if s > 1 {
        let mask = nn::causal_mask(builder, s.clone());
        let mask = broadcast_to(builder, mask, sh);
        attn = attn + mask;
        // }

        let attn = nn::softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        self.gpt_linear(builder, dim, dim, p.extend(["c_proj"]).unwrap(), attn)
    }

    fn layer(&self, builder: &Builder, _layer_id: usize, p: Path, x: Var) -> Var {
        // Params
        let ln_1 = p.extend(["ln_1"]).unwrap();
        let attn = p.extend(["attn"]).unwrap();
        let ln_2 = p.extend(["ln_2"]).unwrap();
        let mlp = p.extend(["mlp"]).unwrap();

        // layers
        let res = x.clone();
        let x = nn::layernorm(builder, self.config.layer_norm_epsilon, ln_1, x);
        let x = self.attention(builder, _layer_id, &self.config, attn, x);
        let x = res + x;

        let res = x.clone();
        let x = nn::layernorm(builder, self.config.layer_norm_epsilon, ln_2, x);
        let x = self.mlp(builder, self.config.hidden_size, mlp, x);
        x + res
    }
}

// Implement `Def`: this is like torch's `Module`.
impl Module<1, 1> for GPT2Model {
    fn path(&self) -> Path {
        path(vec!["gpt2"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let root = self.path();

        // self.info();

        let mut x = self.embeddings(builder, root.clone(), x);

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                root.concat(&path(vec!["h", &i.to_string()]).unwrap()),
                x,
            );
        }

        x = nn::layernorm(
            builder,
            self.config.layer_norm_epsilon,
            root.concat(&path(vec!["ln_f"]).expect("invalid param path")),
            x,
        );

        // weight tied lm_head
        x = nn::linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.concat(&path(vec!["wte"]).expect("invalid param path")),
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
            safetensors::Dtype::F32 => {
                use catgrad::typecheck::*;
                let data: Vec<f32> = tensor_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
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
