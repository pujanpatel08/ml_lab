# Module 04: Hugging Face

New dependencies (run from `ML_Lab`):

```bash
uv add transformers datasets
```

Hugging Face is where the open-model world lives. Nearly every open-weight model, from GPT-2 to Llama and Qwen, is published there, along with most public datasets. LM Studio and Ollama download from it under the hood, and so will your fine-tuning code. This module teaches you to use it from Python, and it opens up a real model so you can watch it predict the next token.

## The pieces

The Hub (huggingface.co) is GitHub for models and datasets. Every model is a git repository of files. A typical model repo contains:

| File | What it is |
|---|---|
| `config.json` | The architecture: number of layers, hidden size, attention heads, vocabulary size |
| `model.safetensors` | The weights, often split into several shards for big models |
| `tokenizer.json` | The tokenizer's vocabulary and merge rules |
| `tokenizer_config.json` | Tokenizer settings, usually including the chat template |
| `generation_config.json` | Default sampling settings |
| `README.md` | The model card: who made it, training data, intended use, license |

Each library does one job:

| Library | Job |
|---|---|
| `huggingface_hub` | Download and upload files, manage your account and the local cache. Gives you the `hf` CLI |
| `transformers` | The model code. Turns `config.json` and the weights into a working PyTorch model |
| `tokenizers` | Fast tokenizers, written in Rust. `transformers` uses it automatically |
| `datasets` | Load, process, and stream datasets efficiently |

`mlx-lm` (module 07) reads the same repos. The `mlx-community` organization on the Hub publishes MLX conversions of popular models.

## Do this in the browser first (10 minutes)

1. Open huggingface.co/Qwen/Qwen2.5-1.5B-Instruct. Read the model card and find the license.
2. Open the "Files and versions" tab. Click `config.json` and find `num_hidden_layers`, `hidden_size`, and `vocab_size`.
3. Find the model tree on the right side of the model page. It shows fine-tunes and quantizations other people built from this model. Click into a quantized one.
4. Search for "Qwen2.5-1.5B-Instruct" under the `mlx-community` organization and compare its files to the original.
5. Open a gated model like meta-llama/Llama-3.2-1B-Instruct. You'll see you have to accept a license before downloading. Accept it if you want to use Llama models later.

## The cache

Everything you download lands in `~/.cache/huggingface/hub`, no matter which folder you run code from. Two commands to know:

```bash
uv run hf cache ls                 # what's cached and how big
uv run hf cache rm model/gpt2      # delete one repo from the cache
```

This module downloads about 2 GB (GPT-2, a 0.5B Qwen model, a small sentiment model, and a small dataset).

## Then

Open `hugging_face.ipynb`.

## Why this matters

When you build on open models, the Hub is your supply chain. You'll pick base models by reading model cards and licenses (some forbid commercial use, which matters for Darvis), you'll pull datasets from it, and you'll publish your own fine-tuned models to it, privately at first.

You're done when you can explain what each file in a model repo does, and you can turn a list of chat messages into the exact token ids a model sees.
