//! A stripped-down version of ModelRunner from catgrad examples, intended for serving
use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::nn::layers::{argmax, cast, reshape},
    core::{Dtype, NdArrayType, Shape, Var},
};
use hf_hub::api::sync::Api;
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use tokenizers::tokenizer::{Result, Tokenizer};

use crate::models::gemma::Model as GemmaModel;
use crate::models::gpt2::Model as GPT2Model;
use crate::models::llama::Model as LlamaModel;
use crate::models::olmo::Model as OlmoModel;
use crate::models::phi::Model as PhiModel;
use crate::models::qwen::Model as QwenModel;
use crate::models::utils::{Cache, Config, ModelBuilder};

use crate::utils::read_safetensors_multiple;

use crate::serve;

/// Load model
pub struct ModelLoader {
    config: Config,
    model_paths: Vec<PathBuf>,
    tokenizer_path: PathBuf,
    tokenizer_config_path: PathBuf,
    use_kv_cache: bool,
}

fn read_to_value<V: for<'a> serde::Deserialize<'a>>(path: PathBuf) -> Result<V> {
    let config_str = &std::fs::read_to_string(path).map_err(|e| serve::Error::IO(e.to_string()))?;
    let result: V =
        serde_json::from_str(config_str).map_err(|e| serve::Error::IO(e.to_string()))?;
    Ok(result)
}

impl ModelLoader {
    pub fn new(model_name: &str, use_kv_cache: bool) -> serve::Result<Self> {
        let (model_paths, config_path, tokenizer_path, tokenizer_config_path) =
            get_model_files(model_name);

        let config: Config = read_to_value(config_path)?;

        Ok(Self {
            config,
            model_paths,
            tokenizer_path,
            tokenizer_config_path,
            use_kv_cache,
        })
    }
}

pub struct ModelTokenizer {
    pub tokenizer: Tokenizer,
    pub chat_template: String,
}

impl ModelTokenizer {
    fn new(tokenizer_path: PathBuf, tokenizer_config_path: PathBuf) -> serve::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        let tokenizer_config: serde_json::Value = read_to_value(tokenizer_config_path)?;
        let chat_template = tokenizer_config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(Self {
            tokenizer,
            chat_template,
        })
    }

    fn render_context(&self, messages: &[serve::Message]) -> String {
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template("chat", &self.chat_template).unwrap();
        let tmpl = env.get_template("chat").unwrap();
        let message_context: Vec<_> = messages
            .iter()
            .map(|msg| context!(role => msg.role, content => msg.content))
            .collect();
        tmpl.render(context!(
            messages => message_context,
            add_generation_prompt => true,
            enable_thinking => false
        ))
        .expect("template failed to render")
    }
}

pub struct ModelRunner {
    pub tensors: Rc<HashMap<String, TaggedNdArray>>,
    pub state: Option<EvalState>,
    pub model: Box<dyn ModelBuilder>,
    pub use_kv_cache: bool,
    pub config: Config,
    pub context: Vec<i32>,
}

impl ModelRunner {
    pub fn new(
        model_paths: Vec<PathBuf>,
        config: Config,
        use_kv_cache: bool,
    ) -> Result<ModelRunner> {
        env_logger::init();

        let arch = &config.architectures[0];

        let mut model: Box<dyn ModelBuilder> = match arch.as_str() {
            "LlamaForCausalLM" => Box::new(LlamaModel {}),
            "Olmo2ForCausalLM" => Box::new(OlmoModel {}),
            "Qwen3ForCausalLM" => Box::new(QwenModel {}),
            "Gemma3ForCausalLM" => Box::new(GemmaModel {}),
            "Phi3ForCausalLM" => Box::new(PhiModel {}),
            "GPT2LMHeadModel" => Box::new(GPT2Model {}),
            _ => return Err("Unknown architecture {arch}".into()),
        };

        let mut tensors = read_safetensors_multiple(model_paths);
        model.post_load(&mut tensors);

        Ok(Self {
            tensors: Rc::new(tensors),
            state: None, // TODO?
            model,
            use_kv_cache,
            config,
            context: vec![],
        })
    }

    fn next_token(&self, builder: &Builder, logits: Var) -> Var {
        let batches = logits.label.shape.0[0];
        let am = argmax(builder, logits);
        let am = reshape(builder, Shape(vec![batches, 1]), am);
        cast(builder, Dtype::I32, am)
    }

    fn build(&mut self, tokens: usize) {
        let batches = 1;
        let in_type = NdArrayType::new(Shape(vec![batches, tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let positions = x.label.shape.0[1];
            let mut cache = Cache::init(builder, &self.config, positions, self.use_kv_cache);
            let result = self
                .model
                .build(builder, &self.config, &mut cache, 0, x.clone());
            let new_token = self.next_token(builder, result);
            (vec![x], vec![new_token])
        });

        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors));
    }

    // Make a forward pass given a list of tokens
    fn run(&mut self, x: &NdArray<i32>) -> TaggedNdArray {
        let [result] = self
            .state
            .as_mut()
            .unwrap()
            .eval_with(vec![x.clone().into()])[..]
        else {
            panic!("unexpected result")
        };

        result.clone()
    }

    pub fn generate(&mut self, tokens: Vec<i32>) -> Option<i32> {
        let num_tokens = tokens.len();
        let batches = 1;
        let input = NdArray::new(tokens, Shape(vec![batches, num_tokens / batches]));
        self.build(num_tokens);
        let result = self.run(&input);

        let token = result.data()[0] as i32;
        if self.config.get_eos_token_ids().contains(&token) {
            return None;
        }
        Some(token)
    }
}

fn get_model_files(model: &str) -> (Vec<PathBuf>, PathBuf, PathBuf, PathBuf) {
    let api = Api::new().unwrap();

    let repo = api.model(model.to_string());

    // Get the model.safetensor file(s)
    let m = if let Ok(index) = repo.get("model.safetensors.index.json") {
        let index = std::fs::File::open(index).unwrap();
        let json: serde_json::Value = serde_json::from_reader(&index).unwrap();
        let mut set = std::collections::HashSet::new();
        if let Some(weight_map) = json.get("weight_map").unwrap().as_object() {
            for v in weight_map.values() {
                set.insert(v.as_str().unwrap().to_string());
            }
        }
        set.iter().map(|p| repo.get(p).unwrap()).collect()
    } else {
        vec![repo.get("model.safetensors").unwrap()]
    };

    let c = repo.get("config.json").unwrap();
    let t = repo.get("tokenizer.json").unwrap();
    let tc = repo.get("tokenizer_config.json").unwrap();

    (m, c, t, tc)
}

////////////////////////////////////////////////////////////////////////////////
// Trait impls

impl Iterator for ModelRunner {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        let next_token = self.generate(self.context.clone());
        if let Some(token) = next_token {
            self.context.push(token);
        }
        next_token
    }
}

impl serve::LM<i32> for ModelRunner {
    fn set_context(&mut self, context: Vec<i32>) {
        self.context = context;
    }
}

impl serve::Tokenizer<i32> for ModelTokenizer {
    fn encode(&self, content: String) -> serve::Result<Vec<i32>> {
        let tokens = self.tokenizer.encode(content, true)?;
        Ok(tokens.get_ids().iter().map(|&x| x as i32).collect())
    }

    fn decode(&self, tokens: Vec<i32>) -> serve::Result<String> {
        // TODO: efficiency?
        // TODO: support u32 in interpreter to remove try_into().unwrap().
        let tokens_u32: Vec<u32> = tokens.into_iter().map(|i| i.try_into().unwrap()).collect();
        Ok(self.tokenizer.decode(&tokens_u32, false)?)
    }
}

impl serve::ChatTokenizer<i32> for ModelTokenizer {
    fn encode_messages(&self, messages: Vec<serve::Message>) -> serve::Result<Vec<i32>> {
        // initialize context
        let content = self.render_context(&messages);
        use serve::Tokenizer;
        self.encode(content)
    }
}

impl serve::Loader<i32, ModelRunner, ModelTokenizer> for ModelLoader {
    fn get_runner(&self) -> serve::Result<ModelRunner> {
        Ok(ModelRunner::new(
            self.model_paths.clone(),
            self.config.clone(),
            self.use_kv_cache,
        )?)
    }

    fn get_translator(&self) -> serve::Result<ModelTokenizer> {
        ModelTokenizer::new(
            self.tokenizer_path.clone(),
            self.tokenizer_config_path.clone(),
        )
    }
}
