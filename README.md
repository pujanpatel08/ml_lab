# ML Lab

Personal workspace for learning to build, train, fine-tune, and serve neural networks and LLMs.

- `lessons/` is the course, one module per folder. Start at `lessons/README.md`.
- `experiments/` (create when needed) is for my own projects and tests.
- `src/ml_lab/` is for reusable code I want to import from notebooks.

Machine: Apple M5, 24 GB unified memory. The GPU can use about 17.8 GB by default, which is the real budget for model weights plus context.
Fits well: 7 to 8B at 4-bit or 8-bit, 14B at 4-bit, and QLoRA on a 4-bit 7 to 8B. Tight: 7 to 8B in bf16, 24B at 4-bit.
Rent a GPU for: 32B and up, LoRA on a bf16 7B, and full fine-tuning above about 1B.

Setup on a new machine:

```bash
git clone https://github.com/pujanpatel08/ml_lab.git
cd ml_lab
uv sync
```
