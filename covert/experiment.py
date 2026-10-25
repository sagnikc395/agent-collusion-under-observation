"""Run the covert-signal experiment.

Design: scheme (none, lexical, semantic, improvised) x channel (plain,
paraphrased). Each trial fixes a topic and a secret bit; the sender writes one
message per (scheme, trial), and both channels deliver that same message, raw or
paraphrased. So the paraphrase effect is paired on the identical message.

Trials run in chunks. After each chunk its rows are appended to trials.jsonl and
flushed to disk, with progress.json and run.log alongside, so a run cut off midway
keeps every finished chunk. Rerunning with the same output directory resumes: it
drops any half-written trial and continues from the first missing one. Raising
--n on an existing directory extends the run, since trials are prefix-stable.

    python -m covert.experiment --backend openai --model qwen2.5:3b --n 50
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from covert import prompts as P
from covert.backends import Backend, make_backend

log = logging.getLogger("covert")


@dataclass
class Config:
    model: str
    n_trials: int = 100
    schemes: list[str] = field(default_factory=lambda: ["none", "lexical", "semantic", "improvised"])
    channels: list[str] = field(default_factory=lambda: ["plain", "paraphrased"])
    seed: int = 0
    sender_temperature: float = 0.8
    paraphrase_temperature: float = 0.7
    sender_max_tokens: int = 160
    paraphrase_max_tokens: int = 200


# Fields that may differ when resuming: more trials extends a run, nothing else does.
RESUMABLE = {"n_trials"}


def make_trials(cfg: Config) -> list[dict]:
    rng = random.Random(cfg.seed)
    return [
        {"trial": i, "topic": P.TOPICS[i % len(P.TOPICS)], "bit": rng.choice(P.BITS)}
        for i in range(cfg.n_trials)
    ]


def run_chunk(cfg: Config, backend: Backend, trials: list[dict]) -> list[dict]:
    """The full pipeline for a few trials; returns every (scheme, channel) row for them."""
    # Sampling seeds depend on the chunk's first trial, so a resumed run reproduces.
    seed = cfg.seed * 100_003 + trials[0]["trial"]

    # 1. Sender: one message per (scheme, trial).
    keys = [(s, t) for s in cfg.schemes for t in trials]
    sent = backend.generate(
        [P.sender_messages(s, t["topic"], t["bit"]) for s, t in keys],
        cfg.sender_max_tokens, cfg.sender_temperature, seed,
    )
    message = {(s, t["trial"]): m for (s, t), m in zip(keys, sent)}

    # 2. Channel: deliver raw, or paraphrased once per message.
    delivered = {}
    if "plain" in cfg.channels:
        delivered |= {(s, i, "plain"): m for (s, i), m in message.items()}
    if "paraphrased" in cfg.channels:
        para = backend.generate(
            [P.paraphraser_messages(m) for m in message.values()],
            cfg.paraphrase_max_tokens, cfg.paraphrase_temperature, seed + 1,
        )
        delivered |= {(s, i, "paraphrased"): p for (s, i), p in zip(message, para)}

    # 3. Receiver and monitor both read what was delivered; greedy decoding.
    cells = list(delivered)
    guesses = backend.generate([P.receiver_messages(k[0], delivered[k]) for k in cells], 8, 0.0, seed + 2)
    flags = backend.generate([P.monitor_messages(delivered[k]) for k in cells], 4, 0.0, seed + 3)

    by_trial = {t["trial"]: t for t in trials}
    rows = []
    for (scheme, i, channel), guess_raw, flag_raw in zip(cells, guesses, flags):
        t = by_trial[i]
        guess = P.parse_choice(guess_raw, P.BITS)
        flag = P.parse_choice(flag_raw, ("YES", "NO"))
        # An unparseable answer becomes a coin flip, so it adds noise, not bias.
        coin = random.Random(f"{cfg.seed}-{i}-{scheme}-{channel}").choice(P.BITS)
        rows.append({
            "model": cfg.model,
            "scheme": scheme,
            "channel": channel,
            "condition": f"{scheme}/{channel}",
            **t,
            "message": message[(scheme, i)],
            "delivered": delivered[(scheme, i, channel)],
            "receiver_raw": guess_raw,
            "guess_parsed": guess is not None,
            "guess": guess or coin,
            "correct": (guess or coin) == t["bit"],
            "monitor_raw": flag_raw,
            "monitor_parsed": flag is not None,
            "flagged": flag == "YES",
            "signal_sent": P.signal_present(scheme, t["bit"], message[(scheme, i)]),
            "signal_delivered": P.signal_present(scheme, t["bit"], delivered[(scheme, i, channel)]),
        })
    return rows


def setup_logging(out_dir: Path | None) -> None:
    """Timestamped log lines to stdout and, when saving, to <out_dir>/run.log."""
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):  # rerunning a notebook cell must not double every line
        log.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir is not None:
        handlers.append(logging.FileHandler(out_dir / "run.log"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)


def load_done(out: Path, cfg: Config) -> list[dict]:
    """Rows of fully finished trials; rewrites trials.jsonl without any partial trial."""
    path = out / "trials.jsonl"
    if not path.exists():
        return []
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1  # a line cut off mid-write
    per_trial = defaultdict(list)
    for r in rows:
        per_trial[r["trial"]].append(r)
    expected = len(cfg.schemes) * len(cfg.channels)
    keep = [r for t in sorted(per_trial) if len(per_trial[t]) == expected for r in per_trial[t]]
    dropped = len(rows) - len(keep)
    if bad or dropped:
        log.info(f"resume: dropping {bad} broken line(s) and {dropped} row(s) of unfinished trials")
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
        os.replace(tmp, path)
    return keep


def check_config(out: Path, cfg: Config, backend: Backend) -> None:
    path = out / "config.json"
    new = asdict(cfg) | {"backend": backend.name}
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k for k in new.keys() | old.keys() if old.get(k) != new.get(k)} - RESUMABLE - {"backend"}
        if diff:
            raise ValueError(
                f"{out} holds a run with a different config ({', '.join(sorted(diff))}); "
                "use another output directory, or delete this one to start over"
            )
        new["n_trials"] = max(cfg.n_trials, old.get("n_trials", 0))
    path.write_text(json.dumps(new, indent=2))


def running_accuracy(rows: list[dict], cfg: Config) -> str:
    acc = defaultdict(list)
    for r in rows:
        acc[(r["scheme"], r["channel"])].append(r["correct"])
    short = {"plain": "pl", "paraphrased": "pa"}
    return "  ".join(
        f"{s}/{short.get(c, c)}={sum(v) / len(v):.2f}"
        for s in cfg.schemes for c in cfg.channels if (v := acc[(s, c)])
    )


def run(
    cfg: Config, backend: Backend, out_dir: str | Path | None = None, chunk_size: int = 10
) -> list[dict]:
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    done = []
    if out is not None:
        check_config(out, cfg, backend)
        done = load_done(out, cfg)
    finished = {r["trial"] for r in done}
    todo = [t for t in make_trials(cfg) if t["trial"] not in finished]
    total = len(finished) + len(todo)
    log.info(
        f"model={cfg.model} backend={backend.name} trials={total} "
        f"conditions={len(cfg.schemes) * len(cfg.channels)} chunk={chunk_size} out={out}"
    )
    if finished:
        log.info(f"resuming: {len(finished)} trials already on disk, {len(todo)} to go")

    rows, start = list(done), time.time()
    for c in range(0, len(todo), chunk_size):
        chunk = todo[c : c + chunk_size]
        t0 = time.time()
        new = run_chunk(cfg, backend, chunk)
        rows += new
        if out is not None:
            with open(out / "trials.jsonl", "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
                f.flush()
                os.fsync(f.fileno())
        n_done = len(finished) + c + len(chunk)
        rate = (time.time() - start) / (c + len(chunk))
        eta = rate * (len(todo) - c - len(chunk))
        parse = sum(r["guess_parsed"] for r in new) / len(new)
        progress = {
            "trials_done": n_done,
            "trials_total": total,
            "rows": len(rows),
            "elapsed_s": round(time.time() - start, 1),
            "eta_s": round(eta, 1),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if out is not None:
            (out / "progress.json").write_text(json.dumps(progress, indent=2))
        log.info(
            f"trials {n_done}/{total}  chunk {time.time() - t0:.0f}s  eta {eta / 60:.1f} min  "
            f"receiver parsed {parse:.0%}"
        )
        log.info(f"  running accuracy: {running_accuracy(rows, cfg)}")

    log.info(f"done: {len(rows)} rows" + (f" in {out / 'trials.jsonl'}" if out else ""))
    return rows


def model_slug(model: str) -> str:
    return model.split("/")[-1].replace(":", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf", "openai"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="openai backend only")
    ap.add_argument("--n", type=int, default=100, help="trials per condition")
    ap.add_argument("--chunk", type=int, default=10, help="trials per saved chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: results/<model>")
    a = ap.parse_args()

    kw = {"base_url": a.base_url} if a.backend == "openai" else {}
    backend = make_backend(a.backend, a.model, **kw)
    cfg = Config(model=a.model, n_trials=a.n, seed=a.seed)
    run(cfg, backend, a.out or f"results/{model_slug(a.model)}", chunk_size=a.chunk)


if __name__ == "__main__":
    main()
