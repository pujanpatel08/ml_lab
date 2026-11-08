"""Keep everything this course downloads inside the ML_Lab folder.

By default, Hugging Face models and datasets, W&B caches, and uv's package cache
live in hidden folders in your home directory. This module sends them to
ML_Lab/.cache instead, so deleting ML_Lab removes all of it.

How it works:
  * configure() sets environment variables, skipping any you have set yourself.
  * `uv run python -m ml_lab setup` installs a one-line startup file into this
    project's virtual environment, so every Python process that uses it (notebook
    kernels, `uv run`, scripts) calls configure() when it starts. It also writes
    uv.toml, which points uv's own package cache at ML_Lab/.cache/uv.
  * Run `setup` again after you rebuild .venv or move the ML_Lab folder.

Logins stay in your home folder (the Hugging Face token and the W&B key). They are
credentials shared with your other projects, not downloads.
"""

import os
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"

# "zz" sorts after ml_lab.pth (the editable install), so ml_lab is importable when this runs.
# The try/except keeps a broken checkout from printing errors on every Python start.
PTH_NAME = "zz_ml_lab_cache.pth"
PTH_LINE = (
    'import os; exec("try:\\n import ml_lab.cache as c\\n c.configure()\\nexcept Exception:\\n pass")'
)


def settings() -> dict[str, str]:
    """Environment variables that move downloads into ML_Lab/.cache."""
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return {
        "HF_HOME": str(CACHE / "huggingface"),       # models, datasets, xet chunks, assets
        "HF_TOKEN_PATH": str(Path(hf_home) / "token"),  # keep the login where it already is
        "TORCH_HOME": str(CACHE / "torch"),
        "WANDB_CACHE_DIR": str(CACHE / "wandb" / "cache"),
        "WANDB_DATA_DIR": str(CACHE / "wandb" / "data"),
        "WANDB_ARTIFACT_DIR": str(CACHE / "wandb" / "artifacts"),
    }


def configure() -> bool:
    """Point downloads at ML_Lab/.cache. Does nothing outside an ML_Lab checkout."""
    if not (ROOT / "pyproject.toml").is_file():
        return False
    for key, value in settings().items():
        os.environ.setdefault(key, value)  # setdefault: your own exports always win
    return True


def _uv_cache_dir() -> str:
    import subprocess  # imported here, not at the top: this module loads on every Python start

    try:
        # --color never: notebook kernels set FORCE_COLOR, which would wrap the path in ANSI codes
        cmd = ["uv", "--color", "never", "cache", "dir"]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "(unknown)"
    except (OSError, subprocess.SubprocessError):
        return "(uv not found)"


def _fresh_settings() -> dict[str, str | None]:
    """What a brand-new Python process from this environment gets.

    Runs a child Python with our variables removed, so the answer comes from the
    startup file and not from this process's own (possibly self-configured) environment.
    """
    import json
    import subprocess

    keys = list(settings())
    env = {k: v for k, v in os.environ.items() if k not in keys}
    code = f"import os, json; print(json.dumps({{k: os.environ.get(k) for k in {keys!r}}}))"
    try:
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {k: None for k in keys}


def _inside(value: str) -> bool:
    return Path(value).resolve().is_relative_to(ROOT)


def status(warn_stale: bool = True) -> None:
    pth = Path(sysconfig.get_paths()["purelib"]) / PTH_NAME
    fix = "NO, run: uv run python -m ml_lab setup"
    print(f"ML_Lab folder:           {ROOT}")
    print(f"downloads folder:        {CACHE}")
    print(f"startup file installed:  {'yes' if pth.is_file() else fix}")
    print(f"uv.toml present:         {'yes' if (ROOT / 'uv.toml').is_file() else fix}")
    print("\nA new Python process from this environment gets:")
    fresh = _fresh_settings()
    for key in settings():
        value = fresh.get(key)
        if value is None:
            note, value = "NOT SET", "(unset)"
        elif key == "HF_TOKEN_PATH":
            note = "login, stays in your home folder"
        else:
            note = "inside ML_Lab" if _inside(value) else "OUTSIDE ML_Lab"
        print(f"{key:20} {value}   [{note}]")
    uv_dir = _uv_cache_dir()
    note = "unknown" if uv_dir.startswith("(") else ("inside ML_Lab" if _inside(uv_dir) else "OUTSIDE ML_Lab")
    print(f"{'uv cache':20} {uv_dir}   [{note}]")
    stale = [k for k in settings() if os.environ.get(k) != fresh.get(k)]
    if warn_stale and stale:
        print(f"\nThis running process differs from a new one for: {', '.join(stale)}.")
        print("If you did not set those yourself, it started before setup: restart the kernel or terminal.")


def setup() -> None:
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        sys.exit(f"Run this with the project environment: uv run python -m ml_lab setup (this Python is {sys.prefix})")
    CACHE.mkdir(exist_ok=True)
    (Path(sysconfig.get_paths()["purelib"]) / PTH_NAME).write_text(PTH_LINE + "\n")
    _write_uv_toml()
    print("Setup done. Restart any running notebook kernels so they pick it up.\n")
    status(warn_stale=False)


def _write_uv_toml() -> None:
    """Point uv's package cache at ML_Lab/.cache/uv, keeping any other settings you added to uv.toml."""
    import json
    import re

    toml = ROOT / "uv.toml"
    line = f"cache-dir = {json.dumps(str(CACHE / 'uv'))}"  # a JSON string is a valid TOML string
    text = toml.read_text() if toml.is_file() else ""
    if re.search(r"(?m)^cache-dir\s*=", text):
        text = re.sub(r"(?m)^cache-dir\s*=.*$", lambda _: line, text)
    else:
        header = "# Machine-specific and git-ignored. Written by `uv run python -m ml_lab setup`.\n" if not text else ""
        text = header + line + "\n" + text  # top-level keys must come before any [table]
    toml.write_text(text)


def cli(args: list[str]) -> None:
    command = args[0] if args else "status"
    if command == "setup":
        setup()
    elif command == "status":
        status()
    else:
        sys.exit("usage: python -m ml_lab [status|setup]")
