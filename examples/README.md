## Testing model inference with Catgrad


These examples work with safetensor weights as found on [Huggingface Hub](https://huggingface.co/models)

**They run but the implementations are still incomplete.**

### Supported architectures (WIP): ###

**BERT**, **GPT-2**, **Llama-3**, **Qwen-3**, **OLMo-2**

Example models to test with:

<https://huggingface.co/BAAI/bge-base-en-v1.5>

<https://huggingface.co/meta-llama/Llama-3.2-1B>

<https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct> (Llama architecture)

<https://huggingface.co/Qwen/Qwen3-0.6B>

<https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct>

### Files required ###

The example apps load `model.safetensors` along with `tokenizer.json` and  `config.json`.

Either download them manually for the model you want to test from links like the above, or use [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)
which will place the files  into ~/.cache/huggingface/hub/models--$MODEL_ID:

```
MODEL_ID=HuggingFaceTB/SmolLM2-135M-Instruct
huggingface-cli download $MODEL_ID model.safetensors config.json tokenizer.json
```

You need to pass the model file to the app and it will look for `config.json` and `tokenizer.json` in the same dir. Either pass the full path or create a symbolic link to the model (or copy all three files above into the current directory).

```
ln -s ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/a91318be21aeaf0879874faa161dcb40c68847e9/model.safetensors smol
cargo run --release --example llama -- -p Hello -m smol
```
