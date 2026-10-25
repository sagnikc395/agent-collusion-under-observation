"""The Slurm scripts pass the vLLM endpoint on the CLI; check that it lands.

Array tasks can share a node, so each picks a free port at job start and the
port in the YAML is only a laptop fallback. If this override ever stops
reaching the model config, every episode on the cluster talks to the wrong
server (or tries to start its own), so it is worth a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carriers.runner import apply_endpoint_overrides  # noqa: E402

UNITY_CONFIG = PROJECT_ROOT / "configs" / "unity.yaml"


def _unity_cfg() -> dict:
    return yaml.safe_load(UNITY_CONFIG.read_text(encoding="utf-8"))


def test_override_rewrites_host_port_and_disables_auto_start() -> None:
    cfg = apply_endpoint_overrides(_unity_cfg(), host="10.0.0.5", port=41234)
    vllm = cfg["llm_models"][0]["llm"]["vllm"]
    assert vllm["auto_start_server"] is False
    assert vllm["persistent_server"] is True
    entry = vllm["models"][0]
    assert entry["host"] == "10.0.0.5"
    assert entry["port"] == 41234
    # The checkpoint is untouched unless asked for.
    assert entry["checkpoint"] == "Qwen/Qwen2.5-3B-Instruct"


def test_override_can_swap_the_checkpoint() -> None:
    cfg = apply_endpoint_overrides(_unity_cfg(), port=1, checkpoint="Qwen/Qwen2.5-7B-Instruct")
    assert cfg["llm_models"][0]["llm"]["vllm"]["models"][0]["checkpoint"] == (
        "Qwen/Qwen2.5-7B-Instruct"
    )


def test_no_arguments_leaves_the_config_alone() -> None:
    before = _unity_cfg()
    after = apply_endpoint_overrides(_unity_cfg())
    assert after == before


def test_unity_config_never_lets_an_episode_start_its_own_server() -> None:
    # The batch script owns the server; two episodes racing to launch vLLM on one
    # GPU is an out-of-memory failure halfway through a sweep.
    vllm = _unity_cfg()["llm_models"][0]["llm"]["vllm"]
    assert vllm["auto_start_server"] is False
