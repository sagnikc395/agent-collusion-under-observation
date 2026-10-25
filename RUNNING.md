# Running the experiments

Everything here sits on top of Colosseum, which is cloned separately and used
unmodified. Generation and the judge need one A100; all analysis runs on CPU
from the saved JSON, so you never regenerate to re-analyse.

**Models are open weights, served locally by vLLM. No OpenAI or Anthropic key is
used anywhere in this project** — not for the agents, not for the judge. The
defaults are `Qwen/Qwen2.5-7B-Instruct` (agents) and `Qwen/Qwen2.5-14B-Instruct`
(judge), both ungated on the Hub. See [Models](#models) to swap them.

## The short version (Unity)

On a Unity login node, from the repo root:

```bash
cp .env.example .env         # set SLURM_ACCOUNT and HF_HOME; see below
sbatch slurm/setup.sbatch    # clone Colosseum, build the venv, fetch weights, run the tests
bash slurm/chain.sh          # pilot -> sweep -> judge -> report, as one dependent chain
bash slurm/status.sh         # where it is
```

`chain.sh` submits four stages and wires the Slurm dependencies, so the whole
220-episode plan is one command and no stage starts before its input exists.
Results land in `$CARRIERS_RESULTS` (default `results/unity`) and the tables in
`$CARRIERS_RESULTS/tables/report.md`.

Short on time or GPU hours:

```bash
bash slurm/chain.sh --fast        # core 5 settings x 8 seeds, a few GPU-hours
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
| `HF_HOME` | The two checkpoints are ~45GB. Point this at `/work/pi_<account>/$USER` or `/scratch/workspace/$USER`, never home: the home quota is far smaller |

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
| `configs/` | `unity.yaml` (the Slurm plan), `pilot.yaml`, `main.yaml`, `modal_replication.yaml`, `smoke_fake.yaml` |
| `analysis/` | Regret, lambda_emp, paired tests, local judge, report tables |
| `scripts/start_vllm.sh` | Serve a model by hand (laptop or interactive node) |
| `notebooks/colab_runner.ipynb` | Colab A100 entry point (alternative to Unity) |
| `modal_app.py` | The same sweep on Modal, for the replication model |
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

**1. Pilot (RQ2).** `benign`, `emergent`, `coalition` at 5 seeds. The README says
to run this first: if the coalition gains nothing at 7B, RQ3-RQ6 would measure
noise. Its episodes are the first seeds of the main grid, so the sweep reuses
them rather than redoing them.

**2. Sweep.** One array task per setting, all of that setting's seeds inside it.
Each task holds one GPU, starts its own vLLM on a free port, runs its episodes
four at a time, and shuts the server down. Tasks are independent, so a failure
is confined to one setting.

```bash
SETTINGS=rq4 SEEDS=1-20 sbatch --array=0-6%4 slurm/gen.sbatch   # one stage by hand
sbatch --array=3 slurm/gen.sbatch                               # retry one task
```

The array index selects a setting from the expanded list; `python
slurm/_settings.py plan` prints that list in order, which is how `chain.sh`
works out the array range.

**3. Judge.** A 14B judge on the saved transcripts, Colosseum's three prompts and
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
| `CONFIG` | `configs/unity.yaml` | Experiment YAML |
| `SETTINGS` | `plan` | Setting or group list (`pilot`, `core`, `rq2`...`rq6`, `plan`, `all`) |
| `SEEDS` | `1-20` | Seed list or range |
| `MAXPAR` | `4` | GPU array tasks in flight at once |
| `CONCURRENCY` | `4` | Episodes in flight inside one task |
| `CARRIERS_RESULTS` | `results/unity` | Results root |
| `OVERWRITE` | unset | Set to `1` to redo finished work |

Any `#SBATCH` directive can be overridden without editing the scripts, because
`sbatch` reads `SBATCH_*` from the environment — e.g. `SBATCH_PARTITION=gpu-preempt`
or `SBATCH_CONSTRAINT='a100|vram40'` in `.env`.

## Settings

Names are fixed by the README and are used as directory and analysis keys:
`benign`, `emergent`, `coalition`, `watched`, `no_channel`, `find_partner`,
`verifier`, plus `watched_*` and `no_channel_*` ablations (`_paraphrase`,
`_hide_claims`, `_both`). Groups you can pass instead of a list: `pilot`, `core`,
`rq2`...`rq6`, `plan`, `all`.

## Models

Both roles are open-weight checkpoints pulled from the Hub and served by vLLM in
the job that uses them. Set these in `.env`:

| Role | Default | Alternatives |
|---|---|---|
| Agents (`AGENT_MODEL`) | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-14B-Instruct` (stronger, ~2x the GPU-hours); `meta-llama/Llama-3.1-8B-Instruct` |
| Judge (`JUDGE_MODEL`) | `Qwen/Qwen2.5-14B-Instruct` | `Qwen/Qwen2.5-32B-Instruct-AWQ` (still one 40GB card); `meta-llama/Llama-3.1-8B-Instruct` |

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

`notebooks/colab_runner.ipynb` runs the same stages on a Colab A100 (mounts
Drive, clones both repos, then pilot, sweep, judge, report). `modal_app.py` runs
the sweep on Modal for the replication model:

```bash
modal run modal_app.py::run --settings core --seeds 1-10
modal run modal_app.py::score
modal run modal_app.py::tables
modal volume get carriers-results modal ./results/modal
```

At roughly $2 per A100-hour, $50 is about 24 GPU-hours: enough for the `core`
chain at 10 seeds on a second model, which is what the replication is for. The
full `plan` grid at 14B does not fit in that credit.

## Order of work, from the README

Run the **RQ2 pilot first**. If the coalition gains nothing at 7B, try
Colosseum's `structured` variant (set `base_variant` in `carriers/settings.py`),
then `agent_order: colluders_first` in the config, then a larger agent model,
before spending GPU time on RQ3-RQ6. If time runs short, cut RQ6, then RQ5, then
the combined ablations.

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
| CUDA out of memory | Lower `GPU_UTIL` or `CONCURRENCY`, or ask for a bigger card via `SBATCH_CONSTRAINT` |
| Some episodes missing | `bash slurm/status.sh`, then re-run `chain.sh`; finished episodes are skipped |
| Weights re-downloading every job | `HF_HOME` is not set the same way in every job; set it in `.env` |
