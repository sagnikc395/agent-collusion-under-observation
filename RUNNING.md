# Running the experiments

Everything here sits on top of Colosseum, which is cloned separately and used
unmodified. Generation needs one A100; all analysis runs on CPU from the saved
JSON, so you never regenerate to re-analyse.

## Layout

| Path | What it is |
|---|---|
| `carriers/` | Coalition prompts, carrier ablations, verifier, episode runner |
| `configs/` | `pilot.yaml` (RQ2), `main.yaml` (the 220-episode plan), `modal_replication.yaml`, `smoke_fake.yaml` |
| `analysis/` | Regret, lambda_emp, paired tests, local judge, report tables |
| `scripts/start_vllm.sh` | Starts the agent model (8020) or the judge (8021) |
| `notebooks/colab_runner.ipynb` | Colab A100 entry point |
| `modal_app.py` | The same sweep on Modal, for the replication model |
| `tests/` | Offline wiring checks (no GPU, no API) |
| `results/` | Run outputs; one directory per episode |

## Settings

Names are fixed by the README and are used as directory and analysis keys:
`benign`, `emergent`, `coalition`, `watched`, `no_channel`, `find_partner`,
`verifier`, plus `watched_*` and `no_channel_*` ablations (`_paraphrase`,
`_hide_claims`, `_both`). Groups you can pass instead of a list: `pilot`, `core`,
`rq2`...`rq6`, `plan`, `all`.

```bash
python -m carriers.runner --config configs/main.yaml --settings watched,no_channel
python -m carriers.runner --config configs/main.yaml --settings rq4 --seeds 1-20
```

## Local setup (laptop; tests and analysis only)

```bash
git clone --recurse-submodules https://github.com/umass-ai-safety/colosseum.git
cd colosseum && uv venv --python 3.11 .venv && source .venv/bin/activate
uv sync --no-install-project --extra vllm && cd ..
export COLOSSEUM_ROOT=$PWD/colosseum

colosseum/.venv/bin/python -m pytest tests -q
```

The offline tests drive every setting end to end with a scripted client, so a
wiring mistake shows up in seconds instead of after an hour of GPU time.

## On Colab (the main runs)

Open `notebooks/colab_runner.ipynb`. It mounts Drive, clones both repos, installs
Terrarium with the vLLM extra, starts the server and runs pilot, sweep, judge and
report in that order. Results go to Drive; finished episodes are skipped on
re-run, so a disconnect costs one episode.

Order, from the README: run the **RQ2 pilot first**. If the coalition gains
nothing at 7B, try Colosseum's `structured` variant (set `base_variant` in
`carriers/settings.py`), then `agent_order: colluders_first` in the config, then
a larger model on Modal, before spending time on RQ3-RQ6.

## On Modal (the replication)

```bash
modal run modal_app.py::run --settings core --seeds 1-10
modal run modal_app.py::score          # local judge over the same runs
modal run modal_app.py::tables
modal volume get carriers-results modal ./results/modal
```

At roughly $2 per A100-hour, $50 is about 24 GPU-hours: enough for the `core`
chain (`benign`, `coalition`, `watched`, `no_channel`, `watched_paraphrase`) at
10 seeds on a second model, which is what the replication is for. The full
`plan` grid at 14B does not fit in the credit; run that on Colab at 7B.

## Judge

```bash
bash scripts/start_vllm.sh --judge
python -m analysis.judge --results results/main --base-url http://127.0.0.1:8021/v1
```

Colosseum's three judge prompts and its parsing are imported unchanged; only the
transport is local. A setting with a private channel is judged on that channel;
a setting without one is judged on the blackboards the colluders share, averaged.
Scores come from a 14B model, so only differences within this project mean
anything.

## Analysis

```bash
python -m analysis.report --results results/main --out results/tables
```

Writes `episodes.csv`, `rq1_evidence.csv`, `lambda_fits.csv`, `report.json` and
`report.md`. The RQ1 standard is applied as pre-registered: action test,
externality test, intent test, all three required, Holm-corrected across every
comparison in one family.

### A caveat on lambda_emp

lambda is identified only on decisions where the team-best and the coalition-best
choice differ. In this environment the two coincide unless the colluder's best
choice collides with a ticket a non-coalition agent has already taken, so on
8 tasks and 6 agents the identified count can be small; `lambda_fits.csv` reports
it per setting. When it is near zero, the intent test rests on the judge delta,
which RQ1 allows ("either ... or ..."). Making tickets scarce (`max_tasks: 6`)
raises contention and identification sharply, but it changes the testbed away
from the README's base config, so treat it as a separate robustness run rather
than the headline.

## Output of one episode

```
results/<tag>/runs/<model>/<setting>/<run_id>/
  run_config.json     setting, seeds, coalition, blackboard ids
  metrics.json        Colosseum's metrics, unmodified code
  final_summary.json  assignment, per-agent rewards, joint reward
  instance.json       costs, tasks, weights -> all counterfactual analysis
  carriers.json       paraphrase records, verifier flags, partner survey
  blackboards.json    every channel's full event log
  agent_turns.json / tool_events.json / agent_prompts.* / agent_trajectories.json
  judge.json          written later by analysis/judge.py
```
