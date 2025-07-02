use crate::{
    backend::cpu::{eval::Builder, ndarray::TaggedNdArray},
    core::nn::layers::rope_tables,
    core::Var,
};

use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i32),
    Multiple(Vec<i32>),
}

// This configuration contains the union of relevant fields from all supported models.
// Models ignore fields they don't need. The aliases are for GPT-2 alternative names.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub sliding_window_pattern: usize,
    pub rope_local_base_freq: f32,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub eos_token_id: Option<EosTokenId>,
    pub vocab_size: usize,
    pub architectures: Vec<String>,
}

impl Config {
    // Sometimes the head_dim fields is missing
    fn get_head_dim(&self) -> usize {
        if self.head_dim == 0 {
            self.hidden_size / self.num_attention_heads
        } else {
            self.head_dim
        }
    }

    pub fn get_eos_token_ids(&self) -> Vec<i32> {
        match self.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![id],
            Some(EosTokenId::Multiple(ref ids)) => ids.clone(),
            None => vec![],
        }
    }
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub use_kv_cache: bool,
    pub kv_cache: Vec<Option<(Var, Var)>>,
}

impl Cache {
    pub fn init(builder: &Builder, config: &Config, positions: usize, use_kv_cache: bool) -> Self {
        let (cos, sin) = rope_tables(builder, config.rope_theta, positions, config.get_head_dim());
        Self {
            cos,
            sin,
            use_kv_cache,
            kv_cache: vec![None; config.num_hidden_layers],
        }
    }
}

// Trait for model builders for various architectures (llama, qwen, gpt2, etc.)
pub trait ModelBuilder {
    // Build the model architecture graph for a given input shape
    fn build(
        &self,
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        x: Var,
    ) -> Var;
    // Optional post-processing of loaded weights (renaming, reshaping, etc.)
    fn post_load(&mut self, _tensors: &mut HashMap<String, TaggedNdArray>) {}
}
