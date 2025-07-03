// Two interfaces:
//
// 1. Raw LLM - takes String context produces Vec<Token>. Iterator.
// 2. Runner: message templates,

// A [`LanguageModel`] is a stateful iterator over tokens
// whose internal context can be managed.
pub trait LM<Token> {
    // TODO: &mut self in iter() is a problem; it means we can't do mutable things to self inside
    // the loop!
    fn iter(&mut self, context: Vec<Token>) -> impl Iterator<Item = Token>;

    // tokenize some string content
    fn tokenize(&self, content: String) -> Vec<Token>;

    // Stringify a token
    fn untokenize(&self, token: Token) -> String;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub trait ChatLM<Token>: LM<Token> {
    fn chat(&mut self, context: Vec<Message>) -> impl Iterator<Item = String>;
}
