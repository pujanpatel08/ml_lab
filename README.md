# ML Lab

Personal workspace for learning to build, train, fine-tune, and serve neural networks and LLMs.

- `lessons/` is the course, one notebook per module folder. Start at `lessons/00_uv/uv_basics.ipynb`.
- `experiments/` (create when needed) is for my own projects and tests.
- `src/ml_lab/` is for reusable code I want to import from notebooks.
- `.cache/` (git-ignored) holds everything the course downloads: models, datasets, and package caches.

Machine: Apple M5, 24 GB unified memory. The GPU can use about 17.8 GB by default, which is the real budget for model weights plus context.
Fits well: 7 to 8B at 4-bit or 8-bit, 14B at 4-bit, and QLoRA on a 4-bit 7 to 8B. Tight: 7 to 8B in bf16, 24B at 4-bit.
Rent a GPU for: 32B and up, LoRA on a bf16 7B, and full fine-tuning above about 1B.

Setup on a new machine:

```bash
git clone https://github.com/pujanpatel08/ml_lab.git
cd ml_lab
uv sync
uv run python -m ml_lab setup
```

The last command keeps every download inside this folder. Run it again after rebuilding `.venv` or moving the folder, and restart any open notebook kernels. Section 8 of `lessons/00_uv/uv_basics.ipynb` explains it.
