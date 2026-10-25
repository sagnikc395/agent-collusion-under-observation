# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A hackathon research project (Track 3: Collusion Carriers and Countermeasures) studying whether
monitoring or removing a private channel actually stops LLM-agent collusion, or merely displaces the
coordination onto another carrier (public-message wording, action ordering).

**It is scoped to a one-day sprint on a single 16GB GPU** (README, "Scope"): RQ1-RQ4 only, the
`sprint` setting group (6 settings) x 8 paired seeds = 48 episodes, with `Qwen2.5-3B-Instruct`
agents and a 4-bit `Qwen2.5-7B-Instruct-AWQ` judge (~12GB of weights in total). RQ5 (`find_partner`),
RQ6 (`verifier`), the `hide_claims` / `_both` ablations and the Modal replication stay implemented
and tested but are stretch goals. Do not widen the default grid or raise the model sizes without
being asked; `configs/main.yaml` and `configs/unity.yaml` still hold the 220-episode plan.

`README.md` is the specification and the primary source of truth: it defines the threat model, the
six research questions (RQ1–RQ6), the sprint scope, the pre-registered evidence standard and the
metrics. The
experiment code implementing it now exists; `RUNNING.md` is the operational guide.

What exists:
- `README.md` — the full plan; `RUNNING.md` — how to run it.
- `carriers/` — settings table, coalition prompts, carrier ablations, verifier, episode runner.
- `analysis/` — scoring model, regret, λ_emp fit, paired tests, local judge, report tables.
- `configs/` — `sprint.yaml` (the default 48-episode grid), `pilot.yaml` (RQ2), `unity.yaml` and
  `main.yaml` (the 220-episode plan), `modal_replication.yaml`, `smoke_fake.yaml` (offline).
- `slurm/` — the primary way to run it: `chain.sh` submits pilot, sweep, judge and report as a
  dependent chain; `gen.sbatch` is one array task per setting; `env.sh` is the shared preamble.
- `notebooks/colab_runner.ipynb`, `modal_app.py`, `scripts/start_vllm.sh`, `tests/`.
- `papers/2602.15198v2.pdf` — the Colosseum paper (Nakamura et al., NeurIPS 2026), which this project builds on.
- `format/AuthorKit27/` — AAAI 2027 LaTeX author kit; the project draft is `main.tex`, and the stock
  `AnonymousSubmission2027.tex` and kit support files are left untouched.

## Working on this repo

**Read `README.md` before implementing anything.** The RQ structure is sequential: each stage's design
depends on the previous stage's result, and the README states what changes if a stage comes back
negative. Do not re-derive the experimental design; implement what is specified.

**The evidence standard in RQ1 is pre-registered.** A setting counts as collusive only if all three
tests pass (action gain, non-coalition externality, intent via λ_emp or judge delta). Do not relax,
reorder, or substitute these tests when writing analysis code — the pre-registration is the point.

**Colosseum is an external dependency, used unmodified.** Clone it separately
(`github.com/umass-ai-safety/colosseum`, with `git submodule update --init --recursive` for CoLLAB).
Its environment (`JiraTicketEnvironment`), metrics, and judge prompts are taken as-is; this project's
contribution is new configs, prompt variants, carrier ablations, and analysis on top. Changes to
coalition behavior go in this repo's `carriers/` module (prompts, verifier, paraphraser,
claim-hiding), not in Colosseum's tree.

**Seeds are paired across settings.** In Colosseum a seed fixes the problem instance, agent order and
topology, so every comparison is within-seed (paired permutation tests, Holm-corrected). Any analysis
code must preserve seed pairing rather than comparing pooled means.

**Setting names are fixed** by the README's config table: `benign`, `emergent`, `coalition`,
`watched`, `no_channel`, `find_partner`, `verifier`, plus `*_paraphrase`, `*_hide_claims`, `*_both`.
Use these exact labels for configs, result directories, and analysis keys. The `sprint` group in
`carriers/settings.py` is the default grid; groups are not part of the pre-registration, settings
are.

## How the code hangs off Colosseum

`carriers/runner.py` mirrors Colosseum's `experiments/collusion/run.py` episode loop and imports its
metrics, prompt wrapper, blackboard logger and run utilities. It exists because Colosseum forces the
prompt variant to `control` whenever the secret channel is off, which rules out `no_channel` and
`find_partner`. Run artifacts keep Colosseum's file names and key names, so its judge script still
reads our runs.

Carrier ablations are read-time hooks in `CarrierProtocol` (a subclass): the paraphraser rewrites
other agents' public messages once per event, claim hiding drops `action_executed` events and the
other agents' entries in `joint_assignment`. The verifier is not an environment agent — it holds no
tickets, so the team objective is unchanged.

`analysis/env_model.py` re-implements the JIRA scoring rule in pure Python so counterfactuals run on
CPU from `instance.json`. It is checked against Terrarium in
`tests/test_env_model_matches_terrarium.py`; if you touch it, that test must stay green.

## Environment

`pyproject.toml` now targets Python >= 3.11. Experiment runs use Colosseum's own venv (`uv sync
--no-install-project --extra vllm` inside the clone); analysis only needs numpy/scipy/pandas.
Set `COLOSSEUM_ROOT` if the clone is not at `./colosseum`.

```bash
# On Slurm (the primary path): the whole plan as one dependent chain
sbatch slurm/setup.sbatch        # once: clone, venv, weights, tests
bash slurm/chain.sh              # pilot -> sweep -> judge -> report
bash slurm/chain.sh --fast       # core x 4 seeds; --full for the 220-episode plan
bash slurm/status.sh

# Tests: every setting end to end with a scripted client, no GPU and no API
colosseum/.venv/bin/python -m pytest tests -q

# Generation (one 16GB card; on a T4 also export VLLM_ATTENTION_BACKEND=XFORMERS)
bash scripts/start_vllm.sh
python -m carriers.runner --config configs/sprint.yaml --settings pilot --seeds 1-4
python -m carriers.runner --config configs/sprint.yaml

# Judge, then tables (tables are CPU-only)
bash scripts/start_vllm.sh --judge
python -m analysis.judge --results results/sprint --base-url http://127.0.0.1:8021/v1
python -m analysis.report --results results/sprint --out results/sprint/tables
```

Generation and the local judge run on one 16GB GPU with vLLM, one server at a time
(`Qwen/Qwen2.5-3B-Instruct` for agents, `Qwen/Qwen2.5-7B-Instruct-AWQ` for the judge) — on Colab T4
via `notebooks/colab_runner.ipynb`, or on Slurm via `bash slurm/chain.sh`. Every model
is open-weight and served locally; **no stage uses a hosted API key**, so do not add a provider
client. `AGENT_MODEL` / `JUDGE_MODEL` swap the checkpoints and the vLLM tool-call parser follows the
model family. All analysis runs on CPU from saved JSON outputs
(`metrics.json`, `blackboards.json`, `agent_turns.json`, `tool_events.json`), so analysis work never
needs a GPU or a rerun of generation.

Judge scores come from a local 4-bit 7B model, so absolute values are not comparable to Colosseum's
published numbers — only within-project differences are meaningful. Do not compare against the
paper's figures.

λ_emp is identified only where the team-best and coalition-best choice differ, which in this
environment needs a contested ticket; `lambda_fits.csv` reports the count per setting. When it is
near zero the intent test falls back to the judge delta, which RQ1 permits.

Tests: `colosseum/.venv/bin/python -m pytest tests -q` (needs pytest in that venv:
`VIRTUAL_ENV=colosseum/.venv uv pip install pytest`). Lint with
`colosseum/.venv/bin/python -m ruff check analysis carriers tests scripts modal_app.py`.
