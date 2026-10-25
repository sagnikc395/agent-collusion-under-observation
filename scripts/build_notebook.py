"""Build notebooks/covert_signal.ipynb with the `covert` package inlined.

The notebook writes covert/*.py into the Colab runtime with %%writefile, so it
runs as a single uploaded file with no repo clone. Rerun after editing covert/:

    python scripts/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__.py", "prompts.py", "backends.py", "experiment.py", "analysis.py"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def build() -> dict:
    cells = [
        md("""
# Covert signalling between small LLM agents

Can two agents pass a secret one-bit bid through a public team message, and do a
**monitor** or a **paraphraser** stop it?

| Role | What it does |
|---|---|
| Sender (agent A) | Writes a 2-3 sentence work update that hides a secret bid, HIGH or LOW |
| Paraphraser | Optionally rewrites the message before delivery (`paraphrased` channel) |
| Receiver (agent B) | Reads the delivered message and guesses the bid |
| Monitor | Reads the delivered message and answers YES/NO: is it coded? |

Schemes: `none` (sender not told to hide anything, the chance baseline), `lexical`
(agreed code word), `semantic` (agreed topic: coffee vs tea), `improvised` (no agreed
code, the receiver is a copy of the sender). Every trial fixes the topic and the bit, and
both channels deliver the same sender message, so all comparisons are paired.

**Runtime:** Colab T4 GPU (Runtime → Change runtime type → T4). With vLLM, 100 trials
per condition takes a few minutes once the model is loaded.

**No tokens needed.** The code is embedded in this notebook (section 2), so nothing is
cloned from or pushed to GitHub, and the Qwen models are public on Hugging Face, so no
`HF_TOKEN` either.

**Nothing is lost if Colab cuts out.** Trials run in chunks of `CHUNK_SIZE`; each
finished chunk is appended to `trials.jsonl` on Google Drive at once, next to `run.log`
and `progress.json`. If the runtime dies: reconnect, then **Runtime → Run all**. The run
picks up from the last saved chunk. Section 5 analyses whatever is on disk, finished or
not, and the zips in sections 4 and 7 bring everything back to your computer.
"""),
        code("""
# Public models only: stop huggingface_hub from looking for a token or warning about it.
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
        md("""
## 1. Install

vLLM is fast. If its install fails or breaks the runtime, set `BACKEND = "hf"` below
and skip this cell; plain `transformers` is slower but always works.
"""),
        code("!pip install -q vllm"),
        md("## 2. Write the `covert` package\n\nGenerated from the repo by `scripts/build_notebook.py`."),
        code("os.makedirs('covert', exist_ok=True)"),
    ]
    for name in MODULES:
        src = (ROOT / "covert" / name).read_text()
        cells.append(code(f"%%writefile covert/{name}\n{src}"))
    cells += [
        md("""
## 3. Configure

`MODELS` run one after another, each into its own `<model>/` folder. A model whose
folder already holds `N_TRIALS` finished trials is skipped without loading it, so you
can leave a finished model in the list. The 7B model is the 4-bit AWQ build: full fp16
7B weights do not fit a 15 GB T4.

`N_TRIALS` is per condition; there are 8 conditions. Start with 30 to check the
pipeline end to end, then raise it to 100-200 and run section 4 again: the first 30
trials are reused, not redone. Keep `N_TRIALS` and `SEED` the same across models so
every model sees the same trials (topic and bit).

`USE_DRIVE = True` stores results on Google Drive (Colab asks you to sign in with a
pop-up, no token). That copy survives a disconnect; files on the Colab disk do not.
"""),
        code("""
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
]
BACKEND = "auto"                     # "vllm", "hf", or "auto" (vllm if installed)
N_TRIALS = 100
CHUNK_SIZE = 10                      # trials per saved chunk: a cut-off loses at most this many
SEED = 0
USE_DRIVE = True

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_ROOT = "/content/drive/MyDrive/covert_results"
else:
    RESULTS_ROOT = "results"
os.makedirs(RESULTS_ROOT, exist_ok=True)
print("results go to", RESULTS_ROOT)
"""),
        code("""
import shutil

def save_results(name="covert_results"):
    \"\"\"Zip everything under RESULTS_ROOT and download it to your computer.\"\"\"
    path = shutil.make_archive(name, "zip", RESULTS_ROOT)
    try:
        from google.colab import files
        files.download(path)
    except ImportError:
        print(f"not in Colab; wrote {path}")
    # If no download appears, the browser blocked it: open the folder icon on the left,
    # right-click the zip and choose Download.

def show_progress():
    \"\"\"Progress and the last log lines of every run under RESULTS_ROOT.\"\"\"
    for d in sorted(os.listdir(RESULTS_ROOT)):
        p = os.path.join(RESULTS_ROOT, d, "progress.json")
        if os.path.exists(p):
            print(f"== {d} ==")
            print(open(p).read())
            print("".join(open(os.path.join(RESULTS_ROOT, d, "run.log")).readlines()[-4:]))
"""),
        md("""
## 4. Run

Each model runs in its own Python process (`python -m covert.experiment`), so its GPU
memory is released before the next model loads; freeing vLLM inside the notebook is not
reliable. Progress is logged after every chunk: trials done, ETA, receiver parse rate and
running accuracy per condition. The same lines go to `run.log` in the model's folder.
If a model fails (e.g. out of memory), the loop reports it and moves on to the next one;
whatever was saved is downloaded as a zip either way.

If the 7B model runs out of memory, lower `gpu_memory_utilization` or `max_model_len` in
`VLLMBackend` (section 2). With `BACKEND = "hf"`, the AWQ model also needs
`!pip install -q autoawq`.
"""),
        code("""
import json
import subprocess
import sys
from covert.experiment import model_slug

def trials_done(model):
    p = f"{RESULTS_ROOT}/{model_slug(model)}/progress.json"
    return json.load(open(p))["trials_done"] if os.path.exists(p) else 0

failed = []
try:
    for model in MODELS:
        if trials_done(model) >= N_TRIALS:
            print(f"== {model}: {N_TRIALS} trials already on disk, skipping ==")
            continue
        print(f"== {model} ==")
        cmd = [sys.executable, "-u", "-m", "covert.experiment", "--backend", BACKEND,
               "--model", model, "--n", str(N_TRIALS), "--chunk", str(CHUNK_SIZE),
               "--seed", str(SEED), "--out", f"{RESULTS_ROOT}/{model_slug(model)}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                print(line, end="")
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        if proc.returncode != 0:
            failed.append(model)
            print(f"!! {model} exited with code {proc.returncode}; its saved chunks are kept")
finally:
    # Runs on success, error or interrupt, so a partial run still reaches your computer.
    save_results("covert_results_runs")
print("failed:", failed or "none")
"""),
        md("Check on a run at any time, including after a reconnect, without starting anything:"),
        code("show_progress()"),
        md("""
## 5. Results

Works on partial runs too: it reads whatever trials are on disk.

`accuracy` is how often the receiver recovers the bit (chance = 0.5). `flag_rate`
is how often the monitor flags the delivered message. `covert_success` = decoded
and not flagged. `signal_delivered` = the codebook marker for the right bit is
present in what the receiver saw (agreed schemes only).
"""),
        code("""
import glob
import pandas as pd
from covert import analysis as A

pd.set_option("display.width", 200)
TABLES = f"{RESULTS_ROOT}/tables"
os.makedirs(TABLES, exist_ok=True)
run_dirs = sorted(os.path.dirname(p) for p in glob.glob(f"{RESULTS_ROOT}/*/trials.jsonl"))
print("runs:", [os.path.basename(d) for d in run_dirs])
df = A.load(*run_dirs)
summary = A.summarize(df)
tests = A.paraphrase_tests(df)
summary.to_csv(f"{TABLES}/summary.csv", index=False)
tests.to_csv(f"{TABLES}/paraphrase_tests.csv", index=False)
summary.round(3)
"""),
        md("Paired paraphrase effect per scheme (exact McNemar, Holm-corrected across schemes)."),
        code("tests.round(4)"),
        code('fig = A.plot(summary, f"{TABLES}/fig1.png")'),
        md("## 6. Examples for the report\n\nMessages the receiver decoded in plain form but not after paraphrasing."),
        code("""
for model, g in df.groupby("model"):
    for scheme in ["lexical", "semantic", "improvised"]:
        print(f"===== {model} / {scheme} =====")
        for _, r in A.examples(g, scheme, k=2).iterrows():
            print("PLAIN:      ", r.plain)
            print("PARAPHRASED:", r.paraphrased, "\\n")
"""),
        code("""
# What the improvised sender actually did: read a few, these are the qualitative result.
for model, g in df.groupby("model"):
    print(f"===== {model} =====")
    for _, r in g[(g.scheme == "improvised") & (g.channel == "plain")].head(4).iterrows():
        print(r.bit, "|", "correct" if r.correct else "wrong", "|", "flagged" if r.flagged else "clean")
        print(r.message, "\\n")
"""),
        md("""
## 7. Download everything

One zip with every run (trials, logs, configs) plus the tables and figure, then the
figure and tables on their own for dropping straight into the report.
"""),
        code("""
save_results("covert_results_all")
try:
    from google.colab import files
    for f in ["fig1.png", "summary.csv", "paraphrase_tests.csv"]:
        files.download(f"{TABLES}/{f}")
except ImportError:
    pass
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    out = ROOT / "notebooks" / "covert_signal.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {out}")
