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
- Optional next steps: run the 05 sweep online, and try module 01 with a real LM Studio chat model (pick a plain instruct model, not a reasoning model).
- Kernel: notebooks use the `ml-lab (3.14.2)` kernel from `ML_Lab/.venv`. If a notebook can't connect, open `ML_Lab` in its own local VS Code window, then Select Kernel > Python Environments > `.venv`.

## Stuck log

Short notes on what was hard and what fixed it, newest first.

- (nothing yet)
