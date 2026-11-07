# ML Lab course

A hands-on path from "I can call an API" to "I trained and served my own model." Every module lives in its own folder here. Work through them in order.

## How to use this

1. Open `ML_Lab` in VS Code (File > Open Folder).
2. Read the module's `README.md` first. It explains the concepts and anything you do outside Python.
3. If the module has a notebook (`.ipynb`), open it, click "Select Kernel" in the top right, and pick the `.venv` inside `ML_Lab`. Run cells top to bottom with Shift+Enter.
4. Do the exercises before you look at the solutions at the bottom of each notebook. The struggle is where the learning happens.
5. Commit after each module: `git add . && git commit -m "Finish module 02"`.

Put your own experiments in `experiments/` (create it when you need it), not inside `lessons/`, so the course stays clean and your work stays yours.

## Modules

| # | Module | What you'll be able to do | Status |
|---|---|---|---|
| 00 | uv | Manage Python projects, dependencies, and environments without breaking anything | Ready |
| 01 | LM Studio | Run open-weight LLMs locally, understand quantization, call a local model from Python | Ready |
| 02 | PyTorch basics | Work with tensors, use your Mac's GPU, understand autograd, write a training loop | Ready |
| 03 | micrograd | Build backpropagation from scratch and understand what `loss.backward()` actually does | Ready |
| 04 | Hugging Face | Pull models and datasets from the Hub, use tokenizers and `transformers`, push your own models | Next |
| 05 | Training + W&B | Train a real neural network on a real dataset and track every run in Weights & Biases | Next |
| 06 | GPT from scratch | Build and train a small transformer language model end to end | Later |
| 07 | MLX + LoRA | Fine-tune a real open-weight LLM on your Mac with your own data | Later |
| 08 | Serving | Load your fine-tuned model in LM Studio or Ollama and call it from a FastAPI backend | Later |

Modules 04 onward get built as you finish the earlier ones, so they match where you actually are.

## Why there's no TensorFlow module

PyTorch is the default for research and for nearly all LLM work. Most papers, Hugging Face models, and open training code (nanoGPT, Llama fine-tuning, LoRA tooling) are PyTorch first. You already use JAX at the Hume Center, and MLX covers Apple Silicon. TensorFlow support on Mac GPUs is also weak. If a class or job ever requires TensorFlow, the concepts from module 02 carry over and you'll pick up the API in a day.

## Watch alongside

- Andrej Karpathy, "Neural Networks: Zero to Hero" (YouTube). Module 03 follows his micrograd video, and module 06 follows his GPT video.
- 3Blue1Brown, "Essence of Linear Algebra" and "Neural Networks" (YouTube). Worth a weekend before module 02 if your linear algebra feels rusty.
- Stanford CS336, "Language Modeling from Scratch" (public lectures). Good for depth after module 06.
