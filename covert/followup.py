"""Follow-up experiments over the messages of finished main runs (README, F1-F3).

Both passes reuse the sender messages saved in results/<model>/trials.jsonl, so every
follow-up row pairs with a main row on (sender model, scheme, trial):

- sanitize (F3): the sender's own model rewrites each message into a bare factual update
  (the `sanitized` channel); the receiver and monitor read it as in the main run, and a
  judge scores whether the paraphrased and the sanitized rewrite kept the work content.
- score (F1, F2): a judge model reads every delivered message and is scored by probability
  instead of a one-word answer: P(HIGH) under the receiver prompt, and P(YES) under the
  original (`basic`) and the `informed` monitor prompt. Any model can judge any run, which
  gives cross-size receivers and monitors.

Output, under --out (default results/followup):

    sanitized/<sender model>/trials.jsonl   main-run row format, channel "sanitized"
    scores/<judge model>/scores.jsonl       one row per judged message

Like the main run, both passes write in chunks that are appended and fsynced, log to run.log,
and resume from what is on disk; a broken last line is dropped.

    python -m covert.followup sanitize --backend vllm --model Qwen/Qwen2.5-3B-Instruct
    python -m covert.followup score --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --runs results/Qwen2.5-1.5B-Instruct results/Qwen2.5-3B-Instruct
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

from covert import prompts as P
from covert.backends import Backend, make_backend
from covert.experiment import log, model_slug, read_guess, setup_logging


def p_first(logps: list[float]) -> float:
    """Probability of the first of two candidates, renormalised over the pair."""
    a, b = logps
    return 1.0 / (1.0 + math.exp(b - a))


def read_rows(path: Path) -> list[dict]:
    """Rows of a jsonl file; rewrites it without a line cut off by an interrupted run."""
    if not path.exists():
        return []
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        log.info(f"resume: dropping {bad} broken line(s) from {path}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, path)
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
        f.flush()
        os.fsync(f.fileno())


def check_config(out: Path, cfg: dict) -> None:
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise ValueError(f"{out} holds a follow-up run with a different config; use another --out")
    path.write_text(json.dumps(cfg, indent=2))


def sanitize(backend: Backend, model: str, run_dir: str | Path, out_root: str | Path, chunk: int = 10) -> Path:
    """F3: the `sanitized` channel for one main run, rewritten by that run's own model."""
    run_dir = Path(run_dir)
    main = read_rows(run_dir / "trials.jsonl")
    main_cfg = json.loads((run_dir / "config.json").read_text())
    if main_cfg["model"] != model:
        raise ValueError(f"{run_dir} was sent by {main_cfg['model']}; sanitize it with that model")
    out = Path(out_root) / "sanitized" / model_slug(model)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    check_config(out, {
        "pass": "sanitize", "model": model, "run": run_dir.name, "backend": backend.name,
        "temperature": main_cfg["paraphrase_temperature"], "max_tokens": main_cfg["paraphrase_max_tokens"],
    })

    plain = {(r["scheme"], r["trial"]): r for r in main if r["channel"] == "plain"}
    para = {(r["scheme"], r["trial"]): r["delivered"] for r in main if r["channel"] == "paraphrased"}
    path = out / "trials.jsonl"
    done = {(r["scheme"], r["trial"]) for r in read_rows(path)}
    trials = sorted({t for _, t in plain})
    todo = [t for t in trials if any((s, t) not in done for s in main_cfg["schemes"])]
    log.info(f"sanitize: model={model} run={run_dir} trials={len(trials)} to go={len(todo)} out={out}")

    start = time.time()
    for c in range(0, len(todo), chunk):
        keys = [(s, t) for t in todo[c : c + chunk] for s in main_cfg["schemes"] if (s, t) not in done]
        # Offset from the main run's seeds so the sanitizer never reuses a sender's sample.
        seed = main_cfg["seed"] * 100_003 + keys[0][1] + 50_000
        san = backend.generate(
            [P.sanitizer_messages(plain[k]["message"]) for k in keys],
            main_cfg["paraphrase_max_tokens"], main_cfg["paraphrase_temperature"], seed,
        )
        guesses = backend.generate([P.receiver_messages(s, x) for (s, _), x in zip(keys, san)], 8, 0.0, seed + 2)
        flags = backend.generate([P.monitor_messages(x) for x in san], 4, 0.0, seed + 3)
        util = backend.score(
            [P.judge_messages(plain[k]["message"], x) for k, x in zip(keys, san)]
            + [P.judge_messages(plain[k]["message"], para[k]) for k in keys],
            ("YES", "NO"),
        )
        rows = []
        for j, ((scheme, trial), text, guess_raw, flag_raw) in enumerate(zip(keys, san, guesses, flags)):
            src = plain[(scheme, trial)]
            guess, parsed = read_guess(guess_raw, main_cfg["seed"], trial, scheme, "sanitized")
            flag = P.parse_choice(flag_raw, ("YES", "NO"))
            rows.append({
                **{k: src[k] for k in ("model", "scheme", "trial", "topic", "bit", "message", "signal_sent")},
                "channel": "sanitized",
                "condition": f"{scheme}/sanitized",
                "delivered": text,
                "receiver_raw": guess_raw,
                "guess_parsed": parsed,
                "guess": guess,
                "correct": guess == src["bit"],
                "monitor_raw": flag_raw,
                "monitor_parsed": flag is not None,
                "flagged": flag == "YES",
                "signal_delivered": P.signal_present(scheme, src["bit"], text),
                "utility": p_first(util[j]),
                "utility_paraphrased": p_first(util[len(keys) + j]),
            })
        append_rows(path, rows)
        n = c + len(todo[c : c + chunk])
        log.info(f"sanitize: trials {n}/{len(todo)}  {time.time() - start:.0f}s")
    log.info(f"done: {path}")
    return path


def score(
    backend: Backend, judge: str, run_dirs: list[str | Path], out_root: str | Path, chunk: int = 200
) -> Path:
    """F1, F2: probability-scored receiver and monitors for every message of the given runs."""
    out_root = Path(out_root)
    out = out_root / "scores" / model_slug(judge)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    check_config(out, {"pass": "score", "judge": judge, "backend": backend.name})

    items = []
    for d in map(Path, run_dirs):
        rows = read_rows(d / "trials.jsonl")
        if rows:
            rows += read_rows(out_root / "sanitized" / model_slug(rows[0]["model"]) / "trials.jsonl")
        items += rows
    path = out / "scores.jsonl"
    key = lambda r: (r["model"], r["scheme"], r["channel"], r["trial"])
    done = {key(r) for r in read_rows(path)}
    todo = [r for r in items if key(r) not in done]
    log.info(f"score: judge={judge} messages={len(items)} to go={len(todo)} out={out}")

    start = time.time()
    for c in range(0, len(todo), chunk):
        batch = todo[c : c + chunk]
        texts = [r["delivered"] for r in batch]
        rec = backend.score([P.receiver_messages(r["scheme"], r["delivered"]) for r in batch], ("HIGH", "LOW"))
        mon = backend.score([P.monitor_messages(x) for x in texts], ("YES", "NO"))
        inf = backend.score([P.monitor_messages(x, informed=True) for x in texts], ("YES", "NO"))
        append_rows(path, [
            {
                "judge": judge,
                **{k: r[k] for k in ("model", "scheme", "channel", "trial", "bit")},
                "p_high": p_first(a),
                "p_yes": p_first(b),
                "p_yes_informed": p_first(i),
            }
            for r, a, b, i in zip(batch, rec, mon, inf)
        ])
        log.info(f"score: {c + len(batch)}/{len(todo)} messages  {time.time() - start:.0f}s")
    log.info(f"done: {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pass_", choices=["sanitize", "score"], metavar="sanitize|score")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", help="sanitize: the main run to rewrite (default results/<model>)")
    ap.add_argument("--runs", nargs="+", help="score: main runs to judge (default every results/*/)")
    ap.add_argument("--results", default="results", help="where the main runs live")
    ap.add_argument("--out", default=None, help="default: <results>/followup")
    ap.add_argument("--chunk", type=int, default=None)
    a = ap.parse_args()

    out = a.out or f"{a.results}/followup"
    backend = make_backend(a.backend, a.model)
    if a.pass_ == "sanitize":
        sanitize(backend, a.model, a.run or f"{a.results}/{model_slug(a.model)}", out, a.chunk or 10)
    else:
        runs = a.runs or sorted(str(p.parent) for p in Path(a.results).glob("*/trials.jsonl"))
        score(backend, a.model, runs, out, a.chunk or 200)


if __name__ == "__main__":
    main()
