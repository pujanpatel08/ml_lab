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
- Testing so far (scratch copies, outputs cleared): 00, 02, 03 ran top to bottom with no errors. 05 and 06 ran with shortened training (1 epoch, 300 iterations). 01, 04, 07, 08 were checked for unchanged code only, not executed. Still to do: full-length runs of 05 and 06 on the M5, and a first real run of 04, 07, 08.
- Kernel: notebooks use the `ml-lab (3.14.2)` kernel from `ML_Lab/.venv`. If a notebook can't connect, open `ML_Lab` in its own local VS Code window, then Select Kernel > Python Environments > `.venv`.

## Stuck log

Short notes on what was hard and what fixed it, newest first.

- (nothing yet)
