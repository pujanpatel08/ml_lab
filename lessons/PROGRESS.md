# Progress

| Module | Status | Notes |
|---|---|---|
| 00 uv | Built, not started | |
| 01 LM Studio | Built, not started | Notebook needs the LM Studio server running |
| 02 PyTorch basics | Built, not started | |
| 03 micrograd | Built, not started | Pair with Karpathy's micrograd video |
| 04 Hugging Face | Built, not started | `uv add transformers datasets`; downloads about 2 GB |
| 05 Training + W&B | Built, not started | `uv add pillow` |
| 06 GPT from scratch | Built, not started | Pair with Karpathy's "Let's build GPT" video |
| 07 MLX + LoRA | Built, not started | Downloads about 3 GB; close LM Studio before training |
| 08 Serving | Built, not started | `uv add fastapi uvicorn`; needs module 07's model |
| 09 Modal | Built, not started | `modal` is already in `pyproject.toml`; the cloud cells spend a little money, see the notebook |

Statuses: Not built, Built not started, In progress, Done.

## Course changes

- 2026-11-07: every lesson now lives entirely in its notebook. Module READMEs were folded in and deleted, and `00_uv/uv_basics.ipynb` is new. Gaps filled:
  - 02, sections 9 to 11: geometry of gradient descent, saddles and mini-batch noise, momentum/Adam/AdamW built by hand and checked against `torch.optim`.
  - 03, sections 8 and 9: backprop with matrices, the softmax + cross-entropy gradient, a spiral classifier trained with hand-written gradients.
  - 05, section 4: initialization, vanishing and exploding gradients, clipping, and `train/grad_norm` logging.
  - 06: gradient clipping in the training loop, plus a note on GPT-2 initialization.
- Testing (scratch copies of the repo, W&B offline, outputs cleared): every notebook has now run top to bottom on the M5 with 0 error outputs in its final run. Timings: 04 about 4.5 minutes (with downloads), 05 about 100 seconds, 06 about 6 minutes, 07 about 20 minutes (run twice, the second time after the fixes below), 08 about 1 minute. 01 ran against a stand-in server (`mlx_lm.server` with Qwen2.5-0.5B-Instruct) because LM Studio has no chat model installed here. Not tested: the optional W&B sweep in 05, and the Hub-push snippets in 04 and 07.
- Fixed because of the real runs: 01 (the Ollama exercise crashed when Ollama has no models, and now suggests `qwen2.5:1.5b`, which is Apache 2.0); 04 (wrong tokenizer and base-model claims); 05 (the run takes about 2 minutes, not 15 to 25, plus several section 4 numbers and the exercise 4 and 5 expectations); 06 (initial loss, top-k is not in module 01, scale claims, a wrong "th" vs "ht" example); 07 (training peaks at about 5 GB, not 8 to 12; `wandb sync adapters/wandb`; the training cell no longer floods the notebook with progress-bar redraws); 08 (the cost cell now uses measured prompt tokens, the exercise 3 load test no longer causes fallbacks, and exercise 4 can't send a placeholder prompt to a hosted model).
- Known quirk, left as is: `app.py` retries at temperature 0, so the retry repeats the identical request and only helps with transient failures.
- 2026-11-08: everything the course downloads now lives in `ML_Lab/.cache` (Hugging Face models and datasets, W&B and torch caches, uv's package cache; about 4.7 GB so far). `uv run python -m ml_lab` shows where downloads go, and `uv run python -m ml_lab setup` installs the settings: a startup file in `.venv` plus a git-ignored `uv.toml`. Run setup after a fresh clone, a rebuilt `.venv`, or moving the folder, then restart notebook kernels. Notebooks 04, 05, and 07 also configure themselves in their first cell, and notebook 00 has a new section 8 on all of this. Stays outside on purpose: the Hugging Face and W&B logins, LM Studio's and Ollama's model folders, and a bare `uvx` typed in a terminal (uv ignores `uv.toml` for tool commands). Checked in scratch clones: plain Python, `uv run`, the `uv run --with` overlay, a VS Code-style kernel, subprocesses, exported overrides, and offline loading of the moved downloads.
- 2026-11-10: new module 09, `09_modal/modal_cloud_gpus.ipynb` (plus `endpoint_demo.py`): what Modal is, App/Image/Function/Volume/Secret, a first function, a GPU benchmark against the M5, `map`, Volumes and secrets, a model that stays loaded, deploying, and cost control with live prices from `modal billing rates`. Every cloud cell sits behind `RUN_CLOUD`, and the A100/H100 exercises behind `RUN_BIGGER_GPUS`. Checked so far: the notebook runs top to bottom with `RUN_CLOUD = False`, and its API names were checked against `modal` 1.6.3 and Modal's docs. Not yet run live in Modal's cloud.
- Optional next steps: run the 05 sweep online, and try module 01 with a real LM Studio chat model (pick a plain instruct model, not a reasoning model).
- Kernel: notebooks use the `ml-lab (3.14.2)` kernel from `ML_Lab/.venv`. If a notebook can't connect, open `ML_Lab` in its own local VS Code window, then Select Kernel > Python Environments > `.venv`.

## Stuck log

Short notes on what was hard and what fixed it, newest first.

- (nothing yet)
