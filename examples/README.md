## Testing model inference with Catgrad

These examples work with safetensor weights as found on [Huggingface Hub](https://huggingface.co/models)

**They run but the implementations are still incomplete.**

### Supported architectures (WIP): ###

**BERT**, **GPT-2**, **Llama-3**, **Qwen-3**, **OLMo-2**

### LLM example ###

The `llm` example uses `model.safetensors`, `tokenizer.json` and  `config.json` files from models under  `~/.cache/huggingface/hub/`.
It either downloads the files or reuses the ones already in the cache (maybe previously downloaded by other frameworks like Candle, Transformers or vLLM).

```
cargo run --release --example llm -- -m openai-community/gpt2 -p 'The capital of France' -s 14
The capital of France, Paris, is home to the world's largest concentration of the world

```

Some models to test the `llm` example with:

<https://huggingface.co/openai-community/gpt2>

<https://huggingface.co/meta-llama/Llama-3.2-1B>

<https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>

<https://huggingface.co/Qwen/Qwen3-0.6B>

<https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct>
