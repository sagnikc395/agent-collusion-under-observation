"""End-to-end wiring check for every setting, with a scripted client (no GPU).

Run from the repo root with Colosseum's venv:

    colosseum/.venv/bin/python -m pytest tests/test_runner_smoke.py -q
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carriers.runner import collect_summary, run_from_config  # noqa: E402
from carriers.settings import SETTINGS  # noqa: E402
from tests.fake_client import fake_client_factory  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "smoke_fake.yaml"
OUT = PROJECT_ROOT / "results" / "smoke_fake"


def _run_all() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    return asyncio.run(
        run_from_config(
            str(CONFIG),
            out_dir=str(OUT),
            client_factory=fake_client_factory(seed=1),
        )
    )


def test_every_setting_produces_complete_artifacts() -> None:
    root = _run_all()
    rows = collect_summary(root)
    assert len(rows) == len(SETTINGS), f"expected {len(SETTINGS)} episodes, got {len(rows)}"

    by_setting = {row["setting"]: row for row in rows}
    assert set(by_setting) == set(SETTINGS)

    for setting, row in by_setting.items():
        run_dir = next(root.glob(f"runs/*/{setting}/*"))
        for name in (
            "run_config.json",
            "metrics.json",
            "final_summary.json",
            "blackboards.json",
            "instance.json",
            "carriers.json",
            "tool_events.json",
        ):
            assert (run_dir / name).exists(), f"{setting} missing {name}"
        assert row["status"] == "complete", f"{setting} did not finish: {row['status']}"
        instance = json.loads((run_dir / "instance.json").read_text())
        assert instance["costs"] and instance["tasks"]
        assert len(instance["colluders"]) == 2


def test_colluders_are_paired_across_settings() -> None:
    rows = collect_summary(OUT)
    pairs = {tuple(sorted(row["colluders"])) for row in rows}
    assert len(pairs) == 1, f"coalition must be identical across settings: {pairs}"


def test_carrier_hooks_fire() -> None:
    rows = {row["setting"]: row for row in collect_summary(OUT)}
    assert rows  # guard against an empty sweep

    para_dir = next(OUT.glob("runs/*/no_channel_paraphrase/*"))
    payload = json.loads((para_dir / "carriers.json").read_text())
    assert payload["paraphrase"]["messages_paraphrased"] > 0

    hide_dir = next(OUT.glob("runs/*/no_channel_hide_claims/*"))
    hide_cfg = json.loads((hide_dir / "run_config.json").read_text())
    assert hide_cfg["hide_claims"] is True

    verifier_dir = next(OUT.glob("runs/*/verifier/*"))
    verifier = json.loads((verifier_dir / "carriers.json").read_text())["verifier"]
    assert verifier is not None and verifier["verifier_llm"]["calls"] > 0

    survey_dir = next(OUT.glob("runs/*/find_partner/*"))
    survey = json.loads((survey_dir / "carriers.json").read_text())["partner_survey"]
    assert survey is not None and len(survey) == 2


def test_only_coalition_settings_get_a_private_channel() -> None:
    for setting, spec in SETTINGS.items():
        run_dir = next(OUT.glob(f"runs/*/{setting}/*"))
        cfg = json.loads((run_dir / "run_config.json").read_text())
        assert cfg["secret_channel_enabled"] == spec.secret_channel
        assert (cfg["secret_blackboard_id"] is not None) == spec.secret_channel
