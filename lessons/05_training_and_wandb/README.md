# Module 05: Training a real network, tracked in Weights & Biases

New dependency (run from `ML_Lab`):

```bash
uv add pillow
```

Saved models go in a `checkpoints/` folder, which `.gitignore` already excludes, so weights never end up in git.

In module 02 you trained toy models on made-up data. Here you train real image classifiers on Fashion-MNIST: 70,000 small grayscale photos of clothing in 10 classes. The dataset is small enough to train in minutes on your Mac and hard enough that architecture and training choices visibly matter.

You'll learn the habits that separate real training from toy loops:

- Train, validation, and test splits, and why the test set gets touched once
- Cross-entropy loss and the sanity check every classifier should pass at step 0
- MLPs vs. convolutional networks
- Overfitting, and the two standard fixes (dropout and weight decay)
- Learning rate schedules with warmup
- Saving and reloading checkpoints
- Logging every run to Weights & Biases so you can compare them, plus a small hyperparameter sweep

## Weights & Biases in two minutes

You logged in during setup (`uv run wandb login`). Every `wandb.init(...)` starts a run, every `wandb.log({...})` adds a point to its charts, and `wandb.finish()` closes it. Runs are grouped into projects. This module logs to a project called `ml-lab-fashion`, which you'll find at wandb.ai under your account.

The workflow that matters: change one thing, run, compare against the previous run in the W&B dashboard. When you have 30 runs, the dashboard is how you remember what you tried and what worked. You can tick runs on and off in the runs table and the charts overlay them.

If you're ever offline, set `WANDB_MODE=offline` before starting Jupyter and sync later with `uv run wandb sync wandb/offline-run-*`.

## Then

Open `training_with_wandb.ipynb`. The full notebook takes roughly 15 to 25 minutes of training time on your Mac. The sweep at the end is optional and adds about 10 more.

## Why this matters

Every model you'll train later, including the LoRA fine-tune in module 07, uses this loop and these diagnostics. When a run goes wrong, the checks you learn here (initial loss, overfitting one batch, train vs. validation gap, learning rate) are how you find out why. Experiment tracking is also how ML teams work: when you have ML engineers, "show me the W&B report" will be how decisions get made.

You're done when you can look at a pair of training and validation curves and say whether the model is underfitting, overfitting, or has a learning rate problem.
