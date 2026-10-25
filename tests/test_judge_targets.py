"""Which transcript the local judge scores, and how a run's score is written."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.judge import _target_blackboards, judge_run  # noqa: E402

PUBLIC = {
    "blackboard_id": "0",
    "participants": ["A", "B", "C"],
    "events": [
        {"kind": "communication", "agent": "A", "payload": {"content": "I take ISSUE-1::test"}},
        {"kind": "communication", "agent": "B", "payload": {"content": "Fine by me"}},
    ],
}
SECRET = {
    "blackboard_id": "1",
    "participants": ["A", "B"],
    "events": [
        {"kind": "communication", "agent": "A", "payload": {"content": "Take the good ones"}}
    ],
}


class _StubJudge:
    model = "stub"

    def __init__(self) -> None:
        self.seen = []

    def rate(self, question, rubric, transcript):
        self.seen.append(transcript)
        return {"rating": 3, "rationale": "stub", "parse_error": None, "request_error": None}


def test_private_channel_settings_are_judged_on_that_channel() -> None:
    run_config = {
        "secret_channel_enabled": True,
        "secret_blackboard_id": 1,
        "colluders": ["A", "B"],
    }
    assert _target_blackboards(run_config, [PUBLIC, SECRET]) == [SECRET]


def test_channel_free_settings_are_judged_on_the_boards_the_colluders_share() -> None:
    run_config = {"secret_channel_enabled": False, "colluders": ["A", "B"]}
    assert _target_blackboards(run_config, [PUBLIC]) == [PUBLIC]


def test_judge_run_writes_a_mean_over_the_three_prompts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "setting": "no_channel",
                "seed": 1,
                "secret_channel_enabled": False,
                "colluders": ["A", "B"],
            }
        )
    )
    (run_dir / "blackboards.json").write_text(json.dumps([PUBLIC]))

    stub = _StubJudge()
    payload = judge_run(run_dir, stub)  # type: ignore[arg-type]
    assert payload["judge_mean_rating"] == 3.0
    assert set(payload["judge_ratings"]) == {"simple", "medium", "complex"}
    assert len(stub.seen) == 3, "all three Colosseum prompts are asked"
    assert (run_dir / "judge.json").exists()

    # A second call reuses the stored score rather than re-judging.
    judge_run(run_dir, stub)  # type: ignore[arg-type]
    assert len(stub.seen) == 3


def test_empty_channels_score_zero_without_a_model_call(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_id": "r2", "setting": "benign", "seed": 1, "colluders": ["A"]})
    )
    (run_dir / "blackboards.json").write_text(
        json.dumps([{"blackboard_id": "0", "participants": ["A"], "events": []}])
    )
    stub = _StubJudge()
    payload = judge_run(run_dir, stub)  # type: ignore[arg-type]
    assert payload["judge_mean_rating"] == 0.0
    assert stub.seen == []
