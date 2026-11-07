# Module 07: Fine-tuning a real LLM with MLX and LoRA

No new dependencies. `mlx-lm` is already installed.

You'll fine-tune Qwen2.5-1.5B-Instruct, a real open-weight LLM, on your Mac to do one specific job for Darvis: turn messy student questions into a structured JSON query that a backend can route. For example:

```
"who's easier for data structures, alvarez or chen? fall '26"
-> {"intent": "professor_compare", "courses": ["CS 3114"], "professors": ["Alvarez", "Chen"], "term": "Fall 2026"}
```

That's a realistic piece of a production system. Darvis's chatbot already has a multi-route question handler, and this is the model-based version of its first step.

## Fine-tuning vs. prompting vs. RAG

Fine-tuning changes how a model behaves. It's the right tool when you need a consistent format, a specific style, or a narrow skill done reliably, cheaply, and fast by a small model. Here, the model has to learn VT-specific mappings ("systems" means CS 3214, "lin alg" means MATH 2114) and output strict JSON every time.

It's the wrong tool for teaching facts that change, like this semester's grade distributions. Facts go in the prompt at request time, retrieved from your database. That's retrieval-augmented generation (RAG), which you tried in miniature in module 01. Production systems usually combine both: a fine-tuned model that knows the format and the domain's vocabulary, fed fresh facts through retrieval.

The notebook makes the trade-off concrete. You'll measure the base model with a long, detailed prompt against your fine-tuned model with a one-line prompt, on the same test set.

## LoRA in one paragraph

Full fine-tuning updates all 1.5 billion weights, which needs several times the model's size in memory for gradients and optimizer state. LoRA (low-rank adaptation) freezes the original weights and learns a small correction for each weight matrix, written as the product of two thin matrices. You train well under 1% of the parameters, it fits easily on your Mac, and the result is a small adapter file of tens of megabytes that you can attach to the base model or merge into it. QLoRA is the same idea with the frozen base stored in 4-bit, which saves even more memory. The notebook builds LoRA by hand in PyTorch before using MLX's implementation.

## What happens, and how long it takes

1. Build LoRA by hand in PyTorch on a toy problem (a minute).
2. Generate a synthetic dataset of 900 question/JSON pairs into `data/` (seconds).
3. Download the base model (about 3 GB, once) and measure its baseline accuracy (a few minutes).
4. Fine-tune with `mlx_lm lora` (roughly 5 to 15 minutes on your Mac). Loss curves go to W&B project `ml-lab-lora`.
5. Measure the fine-tuned model on the same test set.
6. Fuse the adapter into the base model and quantize it to 4-bit, ready for serving in module 08.

`data/`, `adapters/`, and `models/` in this folder are ignored by git. The fused models are several GB and belong on the Hugging Face Hub (privately) if you want to keep them, not in git.

Close other heavy apps (LM Studio with a model loaded, Docker) before training. Training uses roughly 8 to 12 GB of your 24 GB.

## Why this matters

This is the skill behind your stated Darvis goal of replacing a hosted model with your own. A small fine-tuned model running on hardware you control can be faster and much cheaper per request than a large general model behind an API, and the data that makes it good (your users' real questions) is something competitors don't have. The catch is that synthetic training data only gets you started. Real usage logs, handled with your users' privacy in mind, are what make it good, which is why module 08 includes logging.

You're done when you can explain what the rank of a LoRA adapter controls, why the B matrix starts at zero, and when you'd fine-tune instead of improving the prompt.
