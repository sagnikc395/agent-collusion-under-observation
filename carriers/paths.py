"""Locate the Colosseum clone and make its `experiments` package importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def colosseum_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the Colosseum checkout root.

    Resolution order: explicit argument, $COLOSSEUM_ROOT, ./colosseum.
    """
    candidate = explicit or os.environ.get("COLOSSEUM_ROOT") or (PROJECT_ROOT / "colosseum")
    root = Path(candidate).expanduser().resolve()
    if not (root / "experiments" / "collusion" / "run.py").exists():
        raise FileNotFoundError(
            f"Colosseum checkout not found at {root}. Clone it "
            "(github.com/umass-ai-safety/colosseum) or set COLOSSEUM_ROOT."
        )
    return root


def ensure_colosseum_on_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Insert the Colosseum root on sys.path (idempotent) and return it."""
    root = colosseum_root(explicit)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
