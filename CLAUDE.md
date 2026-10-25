# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A one-day hackathon project (Apart AI Collusion Sprint, Track 3): can two copies of a small LLM
pass a one-bit secret through a public message, and does an LLM monitor or an LLM paraphraser
stop it? `README.md` is the specification: schemes, channels, metrics, tests and the
pre-registered expectations.

The repo was refactored down from a larger Colosseum-based plan; that plan and its code are
in git history (`README.md` at commit `c78c64e`). Do not bring
Colosseum back or widen the design without being asked.

## Layout

- `covert/prompts.py` — topics, codebooks (`SCHEMES`), role prompts, `parse_choice`, `signal_present`.
- `covert/backends.py` — batched `generate(convs, max_tokens, temperature, seed)` backends:
  vLLM (Colab default), transformers fallback, OpenAI-compatible local server, scripted (tests).
- `covert/experiment.py` — `Config`, `make_trials`, `run_chunk`, `run`: sender → channel →
  receiver + monitor, in chunks. Each chunk is appended and fsynced to `trials.jsonl`, with
  `run.log` and `progress.json`; rerunning on the same directory resumes (partial trials are
  dropped, a changed config is refused). Keep this crash-safety when editing.
- `covert/analysis.py` — `summarize`, `paraphrase_tests`, `plot`, `examples`; CPU only.
- `notebooks/covert_signal.ipynb` — **generated** by `scripts/build_notebook.py`, which inlines
  `covert/*.py` as `%%writefile` cells. Never hand-edit the notebook; edit `covert/` or the
  builder, then rebuild.

## Rules

- **Pairing is the design.** A trial fixes topic and bit across all conditions, and both channels
  deliver the same sender message. Analysis must stay paired (McNemar on matched trials), not
  compare pooled means.
- Condition labels are fixed: schemes `none`, `lexical`, `semantic`, `improvised`; channels
  `plain`, `paraphrased`. Use them verbatim in code, files and the write-up.
- The pre-registered expectations in the README are written before results; do not edit them to
  match results. Report misses as misses.
- Experiments run on Google Colab (the user runs them there). Do not run models locally.
- Models are open-weight and served locally in the runtime; do not add a hosted API client.

## Commands

```bash
python -m pytest tests -q                 # scripted end-to-end test, no GPU
ruff check covert tests scripts
python scripts/build_notebook.py          # after any change to covert/
python -m covert.analysis results/<model> --out results/tables
```

A local venv for tests and lint is at `.venv` (`.venv/bin/python -m pytest tests -q`).
