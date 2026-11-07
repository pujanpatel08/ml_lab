# Module 06: GPT from scratch

No new dependencies.

You'll build a GPT-style transformer language model from an empty file and train it on the complete works of Shakespeare (about 1 MB of text) on your Mac. By the end it writes new, fake Shakespeare. The architecture is the same one inside GPT-2, Llama, and Qwen, just much smaller: about 2 million parameters instead of billions.

This follows Andrej Karpathy's video "Let's build GPT: from scratch, in code, spelled out." (YouTube, about 2 hours). Watch it alongside the notebook the same way you did for micrograd: watch a section, write that piece yourself, then compare.

## What you'll build, in order

1. A character-level tokenizer and the data pipeline
2. A bigram model, the simplest possible language model, as a baseline
3. The averaging trick that makes attention efficient
4. A single self-attention head, then multi-head attention
5. Transformer blocks with residual connections and LayerNorm
6. The full model, trained on your GPU, with W&B logging
7. Sampling with temperature and top-k (the same knobs you used in LM Studio in module 01)

The notebook downloads Tiny Shakespeare into `data/` inside this folder (git ignores it) and saves the trained model to `checkpoints/`.

Training takes about 3 to 6 minutes on your Mac with the default settings. There's a bigger configuration at the top of the training section if you want better text and can wait longer.

## Afterwards

Karpathy's follow-up video, "Let's build the GPT Tokenizer," goes deep on byte-pair encoding, which exercise 3 introduces. His `nanoGPT` repository on GitHub is this notebook's model, cleaned up and scaled to reproduce GPT-2. Reading through `model.py` there after this module is a good test of how much you now understand.

## Why this matters

After this module, an LLM is no longer a black box. When you read about context length, KV caches, attention heads, or parameter counts in a model card, you'll know exactly which lines of code those words refer to. That's the understanding you'll need to make real decisions about model size, cost, and fine-tuning for your products.

You're done when you can draw the transformer block from memory and explain why attention scores are divided by the square root of the head size.
