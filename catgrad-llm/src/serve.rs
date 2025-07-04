//! Abstract interfaces for serving LLMs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Tokenizer Error: {0}")]
    Tokenizer(String),
}

impl From<tokenizers::tokenizer::Error> for Error {
    fn from(err: tokenizers::tokenizer::Error) -> Self {
        Error::Tokenizer(err.to_string())
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// [`LanguageModel`] as a stateful iterator over tokens.
pub trait LM<Token> {
    // TODO: &mut self in iter() is a problem; it means we can't do mutable things to self inside
    // the loop!
    /// Iterate through tokens generated given a context.
    fn iter(&mut self, context: Vec<Token>) -> impl Iterator<Item = Token>;

    /// Tokenize a string
    fn tokenize(&self, content: String) -> Result<Vec<Token>>;

    /// Stringify a token
    fn untokenize(&self, token: Token) -> Result<String>;
}

/// Message type for use with instruct models
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Iterate through stringified tokens given a context provided as [`Message`]
pub trait ChatLM {
    fn chat(&mut self, context: Vec<Message>) -> Result<impl Iterator<Item = Result<String>>>;
}
