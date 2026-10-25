"""Expand setting/group names into setting names, one per line.

Both the array script and the chain use this, so an array index means the same
setting in both places:

    python slurm/_settings.py plan        # 11 names
    python slurm/_settings.py core,rq6
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carriers.settings import resolve_settings  # noqa: E402

names = [n.strip() for arg in sys.argv[1:] for n in arg.split(",") if n.strip()]
for spec in resolve_settings(names or ["plan"]):
    print(spec.name)
