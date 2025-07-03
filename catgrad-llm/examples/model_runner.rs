use catgrad_llm::model_runner::*;
use catgrad_llm::traits::*;
use std::io::Write;

fn main() {
    let mut lm = ModelRunner::new("qwen/qwen3-0.6B", true).unwrap();

    let system_message = Message {
        role: "system".to_string(),
        content: "You are a helpful chat assistant".to_string(),
    };

    let prompt_message = Message {
        role: "user".to_string(),
        content: "Category theory is ".to_string(),
    };

    let messages = vec![system_message, prompt_message];
    for chunk in lm.chat(messages) {
        print!("{chunk}");
        let _ = std::io::stdout().flush();
    }
}
