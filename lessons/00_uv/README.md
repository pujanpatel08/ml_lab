# Module 00: uv

uv manages three things for you: which Python version a project uses, which packages it depends on, and the isolated environment those packages get installed into. Everything else in this course sits on top of it, so it's worth 30 minutes to understand properly.

## The problem it solves

If you `pip install` everything into one global Python, projects end up fighting. Darvis wants one version of a library, a class project wants another, and upgrading one breaks the other. The fix is a virtual environment: a private folder of packages per project. uv creates and manages that folder for you, and it's 10 to 100 times faster than pip.

## The three files that matter

Open `ML_Lab` and look at these.

`pyproject.toml` is what you asked for. It lists your direct dependencies with minimum versions (`torch>=2.15.0`). You edit it through `uv add` and `uv remove`, rarely by hand.

`uv.lock` is what you actually got. It records the exact version of every package, including the dependencies of your dependencies. You listed 6 packages, but the lock file is over 300KB because `torch`, `wandb`, and the rest pull in dozens more. Commit this file. It's what lets someone else (or you on a new laptop) recreate the identical environment.

`.venv/` is the installed environment itself. Never commit it (it's in `.gitignore`). It's fully rebuildable from `uv.lock`, so if it ever gets weird, delete it and run `uv sync`.

Mental model: `pyproject.toml` is the recipe, `uv.lock` is the exact shopping receipt, `.venv` is the cooked meal.

## Your project's layout

You created the project with the package layout, so there's a `src/ml_lab/` folder and a `[project.scripts]` entry in `pyproject.toml`. That means `uv run ml-lab` calls the `main()` function in `src/ml_lab/__init__.py`. You won't need this for the course, but it's how you'd turn reusable code (a data loader, a model class) into an importable package later: anything in `src/ml_lab/` can be imported in any notebook with `from ml_lab import ...`.

## Commands you'll actually use

| Command | What it does |
|---|---|
| `uv add <pkg>` | Adds a dependency to `pyproject.toml`, updates `uv.lock`, installs it into `.venv` |
| `uv remove <pkg>` | The reverse |
| `uv add --dev <pkg>` | Adds a tool you need while developing (linters, test runners) but your code doesn't import |
| `uv sync` | Makes `.venv` match `uv.lock` exactly. Run it after `git pull` or after deleting `.venv` |
| `uv run <cmd>` | Runs a command inside the environment, syncing first if needed. No activating required |
| `uv tree` | Shows the full dependency tree |
| `uv lock --upgrade` | Bumps every package to the newest version your `pyproject.toml` allows |
| `uv python list` | Shows Python versions installed or available |
| `uv python pin 3.12` | Switches the project's Python version (writes `.python-version`) |
| `uvx <tool>` | Runs a command-line tool in a throwaway environment without adding it to your project |

You'll see people run `source .venv/bin/activate` and then plain `python`. That works too, but `uv run` is safer because it always syncs first, so you can't accidentally run against a stale environment.

## Exercises

Run these from inside `ML_Lab` in iTerm2.

1. Run `uv tree | head -40`. Find what `torch` depends on. Then answer for yourself why `uv.lock` is so much bigger than your 6-line dependency list.

2. Add the packages later modules need:
   ```bash
   uv add openai matplotlib
   ```
   Then run `git diff pyproject.toml` and look at exactly what changed. Notice `uv.lock` changed too (`git diff --stat`).

3. Practice removing. Add and remove a package you don't need:
   ```bash
   uv add rich
   uv run python -c "from rich import print; print('[bold green]it works[/]')"
   uv remove rich
   ```

4. Add a dev tool and use it:
   ```bash
   uv add --dev ruff
   uv run ruff check .
   ```
   Look at `pyproject.toml` again. Ruff went into a separate `[dependency-groups]` section, not `dependencies`.

5. Run a tool without installing it into the project:
   ```bash
   uvx pycowsay "uvx runs tools in a throwaway env"
   ```
   Check `pyproject.toml`. Nothing changed.

6. Prove the environment is reproducible. Delete it and rebuild it:
   ```bash
   rm -rf .venv
   uv sync
   ```
   Time how long that takes. That speed is why everyone is moving to uv.

7. Check that PyTorch can see your GPU and that MLX works:
   ```bash
   uv run python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
   uv run python -c "import mlx.core as mx; print(mx.array([1, 2, 3]) * 2)"
   ```

8. Commit:
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Add openai, matplotlib, ruff"
   git push
   ```
   Always commit `pyproject.toml` and `uv.lock` together.

## If a package won't install

Your project uses Python 3.14, which is very new. Most of the ML stack supports it, but occasionally a package won't have a prebuilt build for the newest Python yet, and the install fails. If that happens, switch to 3.12:

1. In `pyproject.toml`, change `requires-python = ">=3.14"` to `requires-python = ">=3.12"`.
2. Run `uv python pin 3.12` and then `uv sync`.

uv downloads Python 3.12 for you if you don't have it.

## Why this matters past this course

When you hire your first engineer, onboarding to your codebase should be `git clone` and `uv sync`, and they're running the exact environment you are. Reproducible environments stop "works on my machine" bugs, which waste a lot of time on small teams.
