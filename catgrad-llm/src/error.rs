use thiserror::Error;

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("IO error occurred: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unsupported tensor dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Unsupported model architecture: {0}")]
    UnsupportedModel(String),

    #[error("Failed to parse JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Failed to deserialize safetensors: {0}")]
    SafetensorsError(#[from] safetensors::SafeTensorError),

    #[error("Tokenizer Error: {0}")]
    TokenizerError(#[from] tokenizers::tokenizer::Error),

    #[error("Unknown error occurred: {0}")]
    HuggingFaceAPIError(#[from] hf_hub::api::sync::ApiError),
}
