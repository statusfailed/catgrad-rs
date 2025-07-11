import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-community/gpt2",
    )
    parser.add_argument("-p", "--prompt", type=str, default="Hello world")
    parser.add_argument("-s", "--seq-len", type=int, default=10)
    parser.add_argument("-t", "--thinking", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    prompt = args.prompt

    if tokenizer.chat_template is not None:
        chat = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )

    # Remove model settings so generate does not warn about sampling parameters
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    inputs = tokenizer(prompt, return_tensors="pt")
    logits = model.generate(**inputs, max_new_tokens=args.seq_len, do_sample=False)
    output = tokenizer.decode(logits[0])

    print(output)
