"""Load saved episodes into one row per run. No GPU, no reruns."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from analysis.env_model import JiraInstance

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Could not parse %s", path)
        return None


@dataclass
class Episode:
    """Everything one run wrote, plus the instance needed for counterfactuals."""

    run_dir: Path
    run_config: Dict[str, Any]
    metrics: Dict[str, Any]
    final_summary: Dict[str, Any]
    instance: JiraInstance
    carriers: Dict[str, Any] = field(default_factory=dict)
    judge: Optional[Dict[str, Any]] = None
    blackboards: Optional[List[Dict[str, Any]]] = None

    @property
    def setting(self) -> str:
        return str(self.run_config.get("setting") or self.run_config.get("prompt_variant"))

    @property
    def seed(self) -> int:
        return int(self.run_config.get("seed"))

    @property
    def model_label(self) -> str:
        return str(self.run_config.get("model_label") or "model")

    @property
    def colluders(self) -> List[str]:
        return [str(a) for a in self.run_config.get("colluders") or []]

    @property
    def complete(self) -> bool:
        return str(self.final_summary.get("status")) == "complete"

    def load_blackboards(self) -> List[Dict[str, Any]]:
        if self.blackboards is None:
            self.blackboards = _read_json(self.run_dir / "blackboards.json") or []
        return self.blackboards


def iter_run_dirs(root: Path) -> Iterator[Path]:
    runs_root = Path(root) / "runs"
    if not runs_root.exists():
        return
    for path in sorted(runs_root.glob("*/*/*")):
        if path.is_dir() and (path / "run_config.json").exists():
            yield path


def load_episode(run_dir: Path) -> Optional[Episode]:
    run_config = _read_json(run_dir / "run_config.json")
    metrics = _read_json(run_dir / "metrics.json")
    final_summary = _read_json(run_dir / "final_summary.json")
    instance_raw = _read_json(run_dir / "instance.json")
    if not (run_config and metrics and final_summary and instance_raw):
        logger.warning("Skipping incomplete run dir %s", run_dir)
        return None
    return Episode(
        run_dir=run_dir,
        run_config=run_config,
        metrics=metrics,
        final_summary=final_summary,
        instance=JiraInstance.from_dict(instance_raw),
        carriers=_read_json(run_dir / "carriers.json") or {},
        judge=_read_json(run_dir / "judge.json"),
    )


def load_episodes(
    roots: List[Path] | Path,
    *,
    settings: Optional[List[str]] = None,
    require_complete: bool = True,
) -> List[Episode]:
    """Load every episode under one or more result roots."""
    if isinstance(roots, (str, Path)):
        roots = [Path(roots)]
    wanted = set(settings) if settings else None

    episodes: List[Episode] = []
    for root in roots:
        for run_dir in iter_run_dirs(Path(root)):
            episode = load_episode(run_dir)
            if episode is None:
                continue
            if wanted and episode.setting not in wanted:
                continue
            if require_complete and not episode.complete:
                logger.warning(
                    "Dropping incomplete episode %s (status=%s)",
                    episode.run_dir.name,
                    episode.final_summary.get("status"),
                )
                continue
            episodes.append(episode)
    return episodes
