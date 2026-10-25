# Running the experiments

Everything here sits on top of Colosseum, which is cloned separately and used
unmodified. Generation and the judge need one 16GB GPU, one server at a time;
all analysis runs on CPU from the saved JSON, so you never regenerate to
re-analyse.

**This is a one-day sprint plan** (README, Scope): RQ1–RQ4, 6 settings × 8
paired seeds = 48 episodes, on `configs/sprint.yaml`. RQ5 (`find_partner`),
RQ6 (`verifier`) and the remaining ablations are implemented and runnable, but
outside the default grid.

**Models are open weights, served locally by vLLM. No OpenAI or Anthropic key is
used anywhere in this project** — not for the agents, not for the judge. The
defaults are `Qwen/Qwen2.5-3B-Instruct` (agents, ~6GB) and
`Qwen/Qwen2.5-7B-Instruct-AWQ` (judge, 4-bit, ~5.6GB), both ungated on the Hub
and ~12GB together. See [Models](#models) to swap them.

## The short version (Colab T4, or any single 16GB card)

`notebooks/colab_runner.ipynb` does all of this; by hand it is:

```bash
export COLOSSEUM_ROOT=$PWD/colosseum
export VLLM_ATTENTION_BACKEND=XFORMERS   # FlashAttention needs sm80; T4 is sm75
export VLLM_DTYPE=half                   # T4 has no bf16

bash scripts/start_vllm.sh                                           # agents on :8020
python -m carriers.runner --config configs/sprint.yaml \
    --settings pilot --seeds 1-4 --out-dir results/sprint            # RQ2 pilot, read it first
python -m carriers.runner --config configs/sprint.yaml \
    --out-dir results/sprint                                         # the 48-episode grid

pkill -f vllm.entrypoints.openai.api_server                          # free the card
bash scripts/start_vllm.sh --judge                                   # judge on :8021
python -m analysis.judge --results results/sprint --base-url http://127.0.0.1:8021/v1
python -m analysis.report --results results/sprint --out results/sprint/tables
```

Budget on a T4: roughly 3–4 hours for the grid and half an hour for the judge.
Finished episodes are skipped on a re-run, so a Colab disconnect costs one
episode, not the sweep.

If the AWQ judge will not start (the 4-bit kernels need sm75 or better), fall
back to `JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct` and say so in the writeup — the
judge delta is then between a judge and agents of the same size, which weakens
the intent test but does not invalidate the action and externality tests.

## The short version (Unity)

On a Unity login node, from the repo root:

```bash
cp .env.example .env         # set SLURM_ACCOUNT and HF_HOME; see below
sbatch slurm/setup.sbatch    # clone Colosseum, build the venv, fetch weights, run the tests
bash slurm/chain.sh          # pilot -> sweep -> judge -> report, as one dependent chain
bash slurm/status.sh         # where it is
```

`chain.sh` submits four stages and wires the Slurm dependencies, so the whole
48-episode grid is one command and no stage starts before its input exists.
Results land in `$CARRIERS_RESULTS` (default `results/sprint`) and the tables in
`$CARRIERS_RESULTS/tables/report.md`.

Wider or narrower:

```bash
bash slurm/chain.sh --full        # the 11-setting, 20-seed plan on configs/unity.yaml
bash slurm/chain.sh --fast        # core 5 settings x 4 seeds, about an hour
bash slurm/chain.sh --dry-run     # print the submissions, send nothing
```

Everything is resumable. Each stage skips work already on disk (an episode with
`metrics.json`, a run with `judge.json`), so a timeout, a preemption or a failed
array task costs only the episodes that were in flight — re-run `chain.sh` and
it picks up where it stopped.

### Two things to put in `.env` first

| Variable | Why |
|---|---|
| `SLURM_ACCOUNT` | Only if your Unity login has no default account (`-A`) |
| `HF_HOME` | The two checkpoints are ~12GB. Small, but still point it at `/work/pi_<account>/$USER` or `/scratch/workspace/$USER` rather than home |

If `HF_HOME` is unset the jobs fall back to `.hf_cache` inside the repo and say
so loudly.

## Layout

| Path | What it is |
|---|---|
| `slurm/chain.sh` | The whole experiment as one dependent chain of jobs |
| `slurm/setup.sbatch` | One-time: clone, venv, weights, tests (CPU) |
| `slurm/gen.sbatch` | Generation; one array task per setting, one GPU each |
| `slurm/judge.sbatch` | The local judge over saved runs (GPU) |
| `slurm/report.sbatch` | Tables and the RQ1 evidence standard (CPU) |
| `slurm/status.sh` | Queue, episodes on disk per setting, recent failures |
| `slurm/env.sh` | Shared job preamble: venv, caches, vLLM helpers |
| `carriers/` | Coalition prompts, carrier ablations, verifier, episode runner |
| `configs/` | `sprint.yaml` (the default grid), `pilot.yaml`, `unity.yaml`, `main.yaml`, `modal_replication.yaml`, `smoke_fake.yaml` |
| `analysis/` | Regret, lambda_emp, paired tests, local judge, report tables |
| `scripts/start_vllm.sh` | Serve a model by hand (laptop or interactive node) |
| `scripts/embed_results.py` | Fold the tables back into a notebook (CPU) |
| `notebooks/colab_runner.ipynb` | Colab T4 entry point: the sprint, end to end |
| `modal_app.py` | The same sweep on Modal, for the larger replication model (stretch) |
| `tests/` | Offline wiring checks (no GPU, no API) |
| `results/` | Run outputs; one directory per episode |

## Stages, and running them one at a time

The chain is four stages; each is an ordinary `sbatch` you can submit alone.

```bash
bash slurm/chain.sh --stage pilot     # RQ2 only
bash slurm/chain.sh --stage gen       # the sweep
bash slurm/chain.sh --stage judge
bash slurm/chain.sh --stage report
```

**1. Pilot (RQ2).** `benign`, `emergent`, `coalition` at 4 seeds. The README says
to run this first: if the coalition gains nothing at 3B, RQ3 and RQ4 would
measure noise. Its episodes are the first seeds of the main grid, so the sweep reuses
them rather than redoing them.

**2. Sweep.** One array task per setting, all of that setting's seeds inside it.
Each task holds one GPU, starts its own vLLM on a free port, runs its episodes
four at a time, and shuts the server down. Tasks are independent, so a failure
is confined to one setting.

```bash
SETTINGS=rq4 SEEDS=1-8 sbatch --array=0-6%4 slurm/gen.sbatch    # one stage by hand
sbatch --array=3 slurm/gen.sbatch                               # retry one task
```

The array index selects a setting from the expanded list; `python
slurm/_settings.py sprint` prints that list in order, which is how `chain.sh`
works out the array range.

**3. Judge.** The 4-bit 7B judge on the saved transcripts, Colosseum's three prompts and
its parsing unchanged; only the transport is local. A setting with a private
channel is judged on that channel; a setting without one on the blackboards the
colluders share, averaged.

**4. Report.** CPU only, from `metrics.json` / `blackboards.json` /
`instance.json`. Writes `episodes.csv`, `rq1_evidence.csv`, `lambda_fits.csv`,
`report.json` and `report.md`, and applies the RQ1 standard as pre-registered:
action test, externality test, intent test, all three required, Holm-corrected
across every comparison in one family.

## Knobs

Flags on `chain.sh`, or the same names in `.env`:

| Name | Default | Meaning |
|---|---|---|
| `CONFIG` | `configs/sprint.yaml` | Experiment YAML |
| `SETTINGS` | `sprint` | Setting or group list (`sprint`, `pilot`, `core`, `rq2`...`rq6`, `plan`, `all`) |
| `SEEDS` | `1-8` | Seed list or range |
| `MAXPAR` | `4` | GPU array tasks in flight at once |
| `CONCURRENCY` | `4` | Episodes in flight inside one task (2 on a 16GB card) |
| `CARRIERS_RESULTS` | `results/sprint` | Results root |
| `OVERWRITE` | unset | Set to `1` to redo finished work |

Any `#SBATCH` directive can be overridden without editing the scripts, because
`sbatch` reads `SBATCH_*` from the environment — e.g. `SBATCH_PARTITION=gpu-preempt`
or `SBATCH_CONSTRAINT='a100|vram40'` in `.env`.

## Settings

Names are fixed by the README and are used as directory and analysis keys:
`benign`, `emergent`, `coalition`, `watched`, `no_channel`, `find_partner`,
`verifier`, plus `watched_*` and `no_channel_*` ablations (`_paraphrase`,
`_hide_claims`, `_both`). Groups you can pass instead of a list: `sprint` (the
default: `benign`, `emergent`, `coalition`, `watched`, `no_channel`,
`no_channel_paraphrase`), `pilot`, `core`, `rq2`...`rq6`, `single_ablations`,
`combined_ablations`, `plan`, `all`.

## Models

Both roles are open-weight checkpoints pulled from the Hub and served by vLLM in
the job that uses them. Set these in `.env`:

| Role | Default | On disk | Alternatives |
|---|---|---|---|
| Agents (`AGENT_MODEL`) | `Qwen/Qwen2.5-3B-Instruct` | ~6GB | `Qwen/Qwen2.5-7B-Instruct-AWQ` (~5.6GB, still fits 16GB); `Qwen/Qwen2.5-7B-Instruct` (~15GB, needs 24GB+); `Qwen/Qwen2.5-1.5B-Instruct` (~3GB, tool calling gets unreliable) |
| Judge (`JUDGE_MODEL`) | `Qwen/Qwen2.5-7B-Instruct-AWQ` | ~5.6GB | `Qwen/Qwen2.5-3B-Instruct` (fallback if AWQ kernels fail); `Qwen/Qwen2.5-14B-Instruct` (~28GB, needs an A100) |

Keep the judge at least as large as the agents: Colosseum found that smaller
judges score collusion lower, so a judge below the agents biases the intent test
toward "no collusion".

vLLM's tool-call parser is chosen from the model family automatically (`hermes`
for Qwen, `llama3_json` for Llama); the agents cannot act without it. Llama also
needs vLLM's tool chat template, via `VLLM_CHAT_TEMPLATE`. Llama checkpoints are
gated: accept the license on the Hub and put `HF_TOKEN` in `.env` first. Qwen is
not gated, which is why it is the default.

Judging with a model from a different family than the agents is a useful
robustness check on the judge delta, and costs one extra judge job:

```bash
JUDGE_MODEL=meta-llama/Llama-3.1-8B-Instruct OVERWRITE=1 sbatch slurm/judge.sbatch
```

Judge scores come from a local model, so absolute values are not comparable to
Colosseum's published numbers — only differences within this project are.

## Local setup (laptop; tests and analysis only)

```bash
git clone --recurse-submodules https://github.com/umass-ai-safety/colosseum.git
cd colosseum && uv venv --python 3.11 .venv && source .venv/bin/activate
uv sync --no-install-project --extra vllm && cd ..
export COLOSSEUM_ROOT=$PWD/colosseum

colosseum/.venv/bin/python -m pytest tests -q
colosseum/.venv/bin/python -m ruff check analysis carriers tests scripts modal_app.py
```

The offline tests drive every setting end to end with a scripted client, so a
wiring mistake shows up in seconds instead of after an hour of GPU time. They
need no GPU and no network.

## Interactive node, or any single machine

Without Slurm the stages are plain commands against a server you start yourself:

```bash
bash scripts/start_vllm.sh                    # agents on :8020
python -m carriers.runner --config configs/unity.yaml --settings pilot --seeds 1-5
bash scripts/start_vllm.sh --judge            # judge on :8021
python -m analysis.judge --results results/unity --base-url http://127.0.0.1:8021/v1
python -m analysis.report --results results/unity --out results/unity/tables
```

The runner takes `--vllm-host/--vllm-port` (or `CARRIERS_VLLM_HOST/_PORT`)
because on the cluster each array task picks a free port at job start — array
tasks can share a node, so the port in the YAML is only this fallback.

## Other backends

`notebooks/colab_runner.ipynb` runs the same stages on a Colab T4 (mounts Drive,
clones both repos, then pilot, sweep, judge, report). Its last section writes the
results back into the notebook, so one `.ipynb` carries both the run and its tables:

```bash
python scripts/embed_results.py --notebook notebooks/colab_runner.ipynb \
    --tables results/sprint/tables --results results/sprint \
    --out results/sprint/colab_runner_results.ipynb
```

It appends `report.md`, `rq1_evidence.csv`, `lambda_fits.csv`, `report.json` and the
per-setting episode counts as markdown cells tagged `carriers_embedded_results`, and
replaces that section on a re-run instead of appending a second copy. Omit `--out` to
write in place. `modal_app.py` runs the
sweep on Modal for the larger replication model, which is a stretch goal:

```bash
modal run modal_app.py::run --settings sprint --seeds 1-8
modal run modal_app.py::score
modal run modal_app.py::tables
modal volume get carriers-results modal ./results/modal
```

At roughly $2 per A100-hour, $50 is about 24 GPU-hours: more than enough to
re-run the 48-episode sprint grid at 7B, which is what the replication is for —
it answers "was the null a cut carrier or a 3B model?" directly.

## Order of work, from the README

Run the **RQ2 pilot first**. If the coalition gains nothing at 3B, try
Colosseum's `structured` variant (set `base_variant` in `carriers/settings.py`),
then `agent_order: colluders_first` in the config, then
`AGENT_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ`, before spending GPU time on RQ3 and
RQ4. RQ5 and RQ6 are already out of the default grid; if the grid itself runs
late, cut `no_channel_paraphrase`, then `emergent`, then seeds down to 6.

## Output of one episode

```
$CARRIERS_RESULTS/runs/<model>/<setting>/<run_id>/
  run_config.json     setting, seeds, coalition, blackboard ids
  metrics.json        Colosseum's metrics, unmodified code
  final_summary.json  assignment, per-agent rewards, joint reward
  instance.json       costs, tasks, weights -> all counterfactual analysis
  carriers.json       paraphrase records, verifier flags, partner survey
  blackboards.json    every channel's full event log
  agent_turns.json / tool_events.json / agent_prompts.* / agent_trajectories.json
  judge.json          written later by analysis/judge.py
```

## A caveat on lambda_emp

lambda is identified only on decisions where the team-best and the coalition-best
choice differ. In this environment the two coincide unless the colluder's best
choice collides with a ticket a non-coalition agent has already taken, so on
8 tasks and 6 agents the identified count can be small; `lambda_fits.csv` reports
it per setting. When it is near zero, the intent test rests on the judge delta,
which RQ1 allows ("either ... or ..."). Making tickets scarce (`max_tasks: 6`)
raises contention and identification sharply, but it changes the testbed away
from the README's base config, so treat it as a separate robustness run rather
than the headline.

## When something fails

| Symptom | Where to look |
|---|---|
| Job died at once | `logs/slurm/<name>-<jobid>.err` |
| vLLM never became healthy | `logs/vllm/<name>-<jobid>.log` (the last 40 lines are also echoed into the job log) |
| CUDA out of memory | Lower `CONCURRENCY` to 1, or `VLLM_MAX_MODEL_LEN` to 4096, or `GPU_UTIL`; on Slurm, ask for a bigger card via `SBATCH_CONSTRAINT` |
| vLLM fails on a T4 with a FlashAttention or bf16 error | `export VLLM_ATTENTION_BACKEND=XFORMERS` and `VLLM_DTYPE=half` before starting the server |
| Some episodes missing | `bash slurm/status.sh`, then re-run `chain.sh`; finished episodes are skipped |
| Weights re-downloading every job | `HF_HOME` is not set the same way in every job; set it in `.env` |
