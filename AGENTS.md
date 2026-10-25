# AGENTS.md

## What this repo is

- A research project and implementation for a hackathon (Track 3: Collusion
  Carriers and Countermeasures). `README.md` is the specification: threat model,
  RQ1–RQ6, pre-registered evidence standard, metrics, and experiment plan.
- Read `README.md` before coding. The RQs are sequential: later stages depend on
  earlier results, and the README says how the design changes if a stage is
  negative. Implement the specified design rather than re-deriving it.
- The scaffold is implemented in `carriers/`, `configs/`, `analysis/`, `tests/`,
  `scripts/`, `notebooks/`, and `modal_app.py`. `RUNNING.md` has the operational
  commands. The paper draft is `format/AuthorKit27/main.tex`.
- `colosseum/` is an ignored local checkout used for offline tests. It is an
  external dependency: do not commit it or edit its tree. A fresh checkout can
  live elsewhere; set `COLOSSEUM_ROOT` to its path.

## Non-negotiable research constraints

- RQ1's evidence standard is pre-registered: a setting counts as collusive only
  if the coalition action-advantage test, the non-coalition reward-loss test, and
  the intent test (λ_emp 95% CI above 0 or judge score +1) all pass. Do not
  relax, reorder, or substitute these tests.
- Preserve seed pairing. A Colosseum seed fixes the problem instance, agent
  order, and topology; comparisons use paired sign-flip permutation tests with
  Holm correction, never pooled means.
- Use the fixed setting labels: `benign`, `emergent`, `coalition`, `watched`,
  `no_channel`, `find_partner`, `verifier`, plus `*_paraphrase`,
  `*_hide_claims`, `*_both`.
- Colosseum is used unmodified. This project's additions belong in `configs/`,
  `carriers/`, `analysis/`, and `notebooks/`; do not patch the Colosseum tree.
- When implementing λ_emp, fit only on instances where the team-best and
  coalition-best actions differ, bootstrap over seeds, and sanity-check that
  benign agents give λ near 0.
- Judge scores come from a local Qwen2.5-14B, so absolute values are not
  comparable to Colosseum's published results. Only within-project differences
  are meaningful.
- Do not fabricate results. The paper is a pre-results protocol; result tables
  stay as placeholders until real runs produce the numbers.

## Environment and commands

- The project targets Python 3.11. Use the Colosseum checkout's environment for
  anything that imports Terrarium; this repo's `.venv` may be a different local
  environment.
- Setup inside the Colosseum clone:

  ```bash
  git clone --recurse-submodules https://github.com/umass-ai-safety/colosseum.git
  cd colosseum
  uv venv --python 3.11 .venv
  source .venv/bin/activate
  uv sync --no-install-project --extra vllm
  cp .env.example .env   # no API keys needed for vLLM-only runs
  ```

- Offline tests, from this repo root (no GPU or API):

  ```bash
  colosseum/.venv/bin/python -m pytest tests -q
  ```

  The suite covers pairing and the RQ1 standard, regret/QRE checks against
  Terrarium, the carrier prompts and verifier, and a fake-client run through
  every setting. Do not assume a test or lint command beyond this is configured.

- Run settings through this repo's runner, from the repo root, with the
  Colosseum environment active and `COLOSSEUM_ROOT` set:

  ```bash
  python -m carriers.runner --config configs/pilot.yaml
  python -m carriers.runner --config configs/main.yaml --settings plan
  python -m carriers.runner --config configs/main.yaml --settings rq3,rq4 --seeds 1-20
  ```

  `configs/smoke_fake.yaml` is the offline wiring check. See `RUNNING.md` for
  Colab, Modal, judge, and report commands.

- Analysis is CPU-only from saved artifacts:

  ```bash
  python -m analysis.report --results results/main --out results/tables
  python -m analysis.judge --results results/main \
    --base-url http://127.0.0.1:8021/v1 --model Qwen/Qwen2.5-14B-Instruct
  ```

- Generation and judging require a GPU (Colab A100 + vLLM). `results/`, `logs/`,
  and `outputs/` are generated and ignored. Change analysis without rerunning
  generation.

## Paper

- `format/AuthorKit27/main.tex` is the project draft. The stock
  `AnonymousSubmission2027.tex` is left untouched.
- Do not edit kit support files (`aaai2027.sty`, `aaai2027.bst`) or lines marked
  `DO NOT CHANGE`; use the template for paper content.
- Compile with pdfTeX from `format/AuthorKit27/`:
  `pdflatex main && bibtex main && pdflatex main && pdflatex main`. The AAAI kit
  is not Tectonic/XeTeX compatible.
