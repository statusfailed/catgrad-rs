use catgrad_llm::run::*;
use catgrad_llm::serve::*;
use std::io::Write;

fn main() {
    let mut lm = ModelRunner::new("qwen/qwen3-0.6B", true).unwrap();

    let system_message = Message {
        role: "system".to_string(),
        content: "You are a helpful chat assistant".to_string(),
    };

    let prompt_message = Message {
        role: "user".to_string(),
        content: "What is 2+2?".to_string(),
    };

    let messages = vec![system_message, prompt_message];
    for chunk in lm.chat(messages) {
        print!("{chunk}");
        let _ = std::io::stdout().flush();
    }
}
