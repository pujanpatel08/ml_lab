"""`uv run python -m ml_lab` shows where downloads go. `... setup` installs the settings."""

import sys

from ml_lab import cache

cache.cli(sys.argv[1:])
