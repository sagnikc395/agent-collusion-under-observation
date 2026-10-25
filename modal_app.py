"""Run the sweep on Modal: a replication model, or overflow from Colab.

    modal run modal_app.py::run --settings core --seeds 1-10
    modal run modal_app.py::judge
    modal volume get carriers-results modal ./results/modal   # pull results down

The container holds one A100, serves the model with vLLM inside the container and
writes every artifact to the `carriers-results` volume, so a timeout or a crash
never loses finished episodes (the runner skips runs that already have
`metrics.json`).

Budget note: at roughly $2/GPU-hour, a $50 credit is about 24 GPU-hours. The
10-seed `core` replication below is the cheapest useful use of it; `--settings
plan` is the full 220-episode grid and will not fit in the credit at 14B.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

import modal

APP_NAME = "agent-collusion-carriers"
COLOSSEUM_REPO = "https://github.com/umass-ai-safety/colosseum.git"
GPU = os.environ.get("CARRIERS_GPU", "A100-40GB")

results_volume = modal.Volume.from_name("carriers-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("carriers-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "terrarium-agents[providers,science,plots,vllm]==0.2.1",
        "names",
        "tqdm",
        "pyyaml",
        "numpy",
        "scipy",
        "pandas",
        "requests",
    )
    .run_commands(f"git clone --recurse-submodules --depth 1 {COLOSSEUM_REPO} /opt/colosseum")
    .env(
        {
            "COLOSSEUM_ROOT": "/opt/colosseum",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Our code is uploaded at run time, so edits do not rebuild the image.
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "results/*",
            "colosseum/*",
            "papers/*",
            "format/*",
            "**/.git/*",
            "**/__pycache__/*",
            "**/.venv/*",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
VOLUMES = {"/results": results_volume, "/cache": hf_cache}


def _setup() -> None:
    os.chdir("/root/project")
    if "/root/project" not in sys.path:
        sys.path.insert(0, "/root/project")


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes=VOLUMES)
def run_sweep(
    config: str = "configs/modal_replication.yaml",
    settings: Optional[str] = None,
    seeds: Optional[str] = None,
    max_concurrent: int = 4,
    overwrite: bool = False,
) -> str:
    """Run episodes on the GPU; artifacts land on the results volume."""
    import asyncio

    _setup()
    from carriers.runner import run_from_config

    setting_list = [s.strip() for s in settings.split(",")] if settings else None
    seed_list = None
    if seeds:
        seed_list = []
        for part in seeds.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                seed_list.extend(range(int(lo), int(hi) + 1))
            elif part:
                seed_list.append(int(part))

    root = asyncio.run(
        run_from_config(
            config,
            settings=setting_list,
            seeds=seed_list,
            max_concurrent_runs=int(max_concurrent),
            overwrite=bool(overwrite),
        )
    )
    results_volume.commit()
    return str(root)


def _start_vllm(model: str, port: int, util: float = 0.90) -> subprocess.Popen:
    """Serve a model inside the container and wait for its health endpoint."""
    import requests

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--max-model-len",
            "16000",
            "--gpu-memory-utilization",
            str(util),
            "--trust-remote-code",
            "--download-dir",
            "/cache/huggingface",
        ]
    )
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + 1800
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {process.returncode}")
        try:
            if requests.get(url, timeout=5).ok:
                return process
        except Exception:
            pass
        time.sleep(10)
    process.terminate()
    raise TimeoutError(f"vLLM did not become ready at {url}")


@app.function(gpu=GPU, timeout=4 * 60 * 60, volumes=VOLUMES)
def judge(
    results: str = "/results/modal",
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    overwrite: bool = False,
) -> int:
    """Score saved runs with the local judge (Colosseum's prompts, our transport)."""
    from pathlib import Path

    _setup()
    server = _start_vllm(model, 8021)
    try:
        from analysis.judge import LocalJudge, judge_root

        payloads = judge_root(
            Path(results),
            LocalJudge(base_url="http://127.0.0.1:8021/v1", model=model),
            overwrite=bool(overwrite),
        )
    finally:
        server.terminate()
    results_volume.commit()
    return len(payloads)


@app.function(timeout=30 * 60, volumes=VOLUMES)
def report(results: str = "/results/modal", out: str = "/results/modal_tables") -> str:
    """Build the tables on CPU from whatever is already on the volume."""
    _setup()
    from analysis.report import main as report_main

    report_main(["--results", results, "--out", out])
    results_volume.commit()
    return out


@app.local_entrypoint()
def run(
    config: str = "configs/modal_replication.yaml",
    settings: Optional[str] = None,
    seeds: Optional[str] = None,
    max_concurrent: int = 4,
    overwrite: bool = False,
):
    print(run_sweep.remote(config, settings, seeds, max_concurrent, overwrite))


@app.local_entrypoint()
def score(results: str = "/results/modal", model: str = "Qwen/Qwen2.5-14B-Instruct"):
    print(f"judged {judge.remote(results, model)} runs")


@app.local_entrypoint()
def tables(results: str = "/results/modal", out: str = "/results/modal_tables"):
    print(report.remote(results, out))
