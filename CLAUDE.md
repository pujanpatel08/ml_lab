# ML_Lab: context for Claude Code

## Who I am and what this repo is

I'm Pujan Patel, a Virginia Tech student (B.S. Computer Science + B.A. Entrepreneurship). My end goal is to found a big tech company, so advise me as a future founder, not a generic student. This repo is my personal workspace for learning to build, train, fine-tune, and serve neural networks and LLMs from scratch.

Background that matters here: I'm fluent in Python and full-stack work (React, FastAPI, Supabase, Vercel). I've done multi-agent RL in JAX (JaxMARL, IPPO/MAPPO) as a research developer. My linear algebra is rusty, so explain the math when it comes up instead of assuming it. I built Darvis (darvis.tech), a VT course-planning app that currently calls a hosted model; one long-term goal of this course is to replace that with my own fine-tuned, self-hosted ~7B model.

My general profile and writing rules (follow the writing rules in every reply):
@~/Documents/Cowork/about_me/about_me.md
@~/Documents/Cowork/about_me/writing_rules.md

## Your two jobs in this repo

1. Tutor. When I'm stuck on a lesson or exercise, teach. Find out what I tried, name the concept I'm missing, and give a hint before a full answer. Give the full solution if I ask for it. Use real technical language. Don't edit my exercise attempts or notebooks unless I ask.
2. Course builder. Build the next modules of the course in `lessons/`, following the conventions below.

## Hard rules

- Everything here is personal. No Virginia Tech resources: no ARC cluster, no VT email, no VT GitLab. Reason: VT acceptable-use and IP policy. Anything that could feed Darvis must have clean IP. Personal compute only: my Mac (MPS / MLX), Kaggle or Colab free tiers, and rented GPUs (RunPod, Lambda, Vast.ai, Modal).
- Use uv for everything. Never `pip install`. Add packages with `uv add`, run with `uv run`, and commit `pyproject.toml` and `uv.lock` together.
- No TensorFlow (decided on purpose). PyTorch is primary, MLX for Apple Silicon training and fine-tuning. I already know some JAX.
- LM Studio means the classic LM Studio app, not Bionic. Ollama is also installed. Both serve an OpenAI-compatible API: LM Studio at `http://localhost:1234/v1`, Ollama at `http://localhost:11434/v1`.

## My setup

- macOS on Apple Silicon. Repo at `~/Developer/ML_Lab`, pushed to the private GitHub repo `pujanpatel08/ml_lab`.
- Python 3.14 via uv, package layout (`src/ml_lab/` is importable from notebooks). If a package has no 3.14 build, switch to 3.12: lower `requires-python` to `>=3.12`, run `uv python pin 3.12`, then `uv sync`.
- Installed: torch, mlx, mlx-lm, wandb, ipykernel, huggingface_hub. Module 00 adds openai, matplotlib, and ruff (dev).
- Accounts, all on my personal email: Hugging Face `pujanpatel8` (logged in with `uv run hf auth login`), Weights & Biases (`uv run wandb login`). The HF cache is `~/.cache/huggingface/hub`.
- Tools: iTerm2, VS Code with the Jupyter extension (notebooks use the `.venv` kernel), Cursor, Docker Desktop, LM Studio, Ollama.
- RAM: unknown. Ask me once, then record it here, because it decides which models fit.

## Course status

The roadmap is `lessons/README.md`. Progress and the log of where I got stuck:
@lessons/PROGRESS.md

Update `PROGRESS.md` whenever I finish a module, get stuck on something notable, or you build a new module.

## Conventions for building modules

- One folder per module: `lessons/NN_snake_case_name/`, containing `README.md` and usually one `snake_case_name.ipynb`. Lowercase snake_case for all file names.
- README: the concepts, any steps outside Python (apps, CLIs, accounts), a short "why this matters" tied to building products or Darvis, and a "you're done when" line.
- Notebook layout: a title markdown cell; a setup cell (device picks cuda, then mps, then cpu; fixed seeds); sections that each go explain, then code, then what to notice; ask me to predict outputs before running where it helps; then an `# Exercises` section with 4 or 5 exercises labeled `**Exercise N: Title.**`; then a `# Solutions` section at the bottom with one code cell per exercise.
- Build on earlier modules explicitly (for example module 06 should point back to temperature in module 01 and softmax in module 02).
- Everything should run on my Mac in minutes. Anything heavy is optional, with notes on doing it on a rented GPU.
- New dependencies: `uv add` them and list them at the top of the module README.
- Test every notebook end to end before handing it over: `uv run --with nbconvert jupyter nbconvert --to notebook --execute lessons/NN_x/x.ipynb --output /tmp/check.ipynb`. Fix anything that fails. Notebooks that need a running server (LM Studio, Ollama) can't be auto-tested, so say that clearly.
- Commit notebooks without outputs. Clear them with `uv run --with nbconvert jupyter nbconvert --clear-output --inplace <notebook>`.
- After building, update the status column in `lessons/README.md` and the table in `PROGRESS.md`.
- Build a module only when I've finished the one before it, and shape it around what I found hard (check `PROGRESS.md` and ask me).
- Follow my writing rules in READMEs and markdown cells too: no em dashes, sentence-case headings, plain prose.

## Plan for the remaining modules

- 04 Hugging Face. The Hub (hf CLI, downloads, the cache, gated models, model cards), the `datasets` library (load, map, streaming), tokenizers (BPE, token ids, special tokens, chat templates), `transformers` with AutoTokenizer and AutoModel on MPS, and pushing a small model or dataset to my Hub account.
- 05 Training + W&B. A real dataset (Fashion-MNIST or CIFAR-10 through `datasets`), DataLoader, MLP then CNN, train/validation split, overfitting, cross-entropy, Adam, learning rate schedules, checkpointing, W&B logging, and a small W&B sweep.
- 06 GPT from scratch. Follows Karpathy's "Let's build GPT": Tiny Shakespeare, character-level bigram, self-attention, multi-head attention, transformer blocks, sampling with temperature. Then tokenization (Karpathy's "Let's build the GPT tokenizer").
- 07 MLX + LoRA. The low-rank adapter math, then `mlx_lm.lora` to fine-tune a small instruct model on a custom JSONL dataset of Darvis-style Q&A, evaluate before and after, fuse adapters, and quantize.
- 08 Serving. Load my fine-tuned model in LM Studio and Ollama (Modelfile), call it from a FastAPI endpoint with the OpenAI client, measure latency and cost against a hosted API, and cover vLLM on a rented GPU for production.

## Decisions made while setting up (Nov 6 to 7, 2026)

- Terminal: kept iTerm2 over Ghostty because its tmux integration (`tmux -CC`) suits remote GPU work.
- Containers: kept Docker Desktop. OrbStack is faster, but not worth switching for light Docker use.
- LM Studio's main download is now Bionic (an agent app). I want the classic LM Studio app for exploring models and serving them. Bionic isn't needed since I use Claude Code.
- LM Studio is for exploring models (quantizations, MLX, settings). Ollama is for building on (scripting, Docker, app backends). Keep both.
- Optional apps still to install: Netron (view model architectures) and Zotero (papers).
- Companion material: Karpathy's "Neural Networks: Zero to Hero", 3Blue1Brown's linear algebra and neural network series, Stanford CS336.
