//! Abstract interfaces for serving LLMs

/// [`LanguageModel`] as a stateful iterator over tokens.
pub trait LM<Token> {
    // TODO: &mut self in iter() is a problem; it means we can't do mutable things to self inside
    // the loop!
    /// Iterate through tokens generated given a context.
    fn iter(&mut self, context: Vec<Token>) -> impl Iterator<Item = Token>;

    /// Tokenize a string
    fn tokenize(&self, content: String) -> Vec<Token>;

    /// Stringify a token
    fn untokenize(&self, token: Token) -> String;
}

/// Message type for use with instruct models
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Iterate through stringified tokens given a context provided as [`Message`]
pub trait ChatLM {
    fn chat(&mut self, context: Vec<Message>) -> impl Iterator<Item = String>;
}
