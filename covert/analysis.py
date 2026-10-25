"""Tables, tests and the headline figure, all on CPU from trials.jsonl.

    python -m covert.analysis results/Qwen2.5-3B-Instruct [more result dirs...]

Per condition: decoding accuracy (Wilson 95% CI, one-sided binomial test against
the 0.5 chance rate), monitor flag rate, and covert success (decoded correctly
and not flagged). The paraphrase effect is an exact McNemar test on the same
messages delivered plain vs paraphrased, Holm-corrected across schemes within
each model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

SCHEME_ORDER = ["none", "lexical", "semantic", "improvised"]
CHANNEL_ORDER = ["plain", "paraphrased"]


def load(*dirs: str | Path) -> pd.DataFrame:
    """Rows from each dir's trials.jsonl; skips a line cut off by an interrupted run."""
    rows = []
    for d in dirs:
        for line in (Path(d) / "trials.jsonl").read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, scheme, channel), g in df.groupby(["model", "scheme", "channel"]):
        n, k, f = len(g), int(g.correct.sum()), int(g.flagged.sum())
        covert = int((g.correct & ~g.flagged).sum())
        sig = g.signal_delivered.dropna()
        out.append({
            "model": model,
            "scheme": scheme,
            "channel": channel,
            "n": n,
            "accuracy": k / n,
            "acc_lo": wilson(k, n)[0],
            "acc_hi": wilson(k, n)[1],
            "p_vs_chance": binomtest(k, n, 0.5, alternative="greater").pvalue,
            "flag_rate": f / n,
            "flag_lo": wilson(f, n)[0],
            "flag_hi": wilson(f, n)[1],
            "covert_success": covert / n,
            "signal_delivered": sig.astype(bool).mean() if len(sig) else np.nan,
            "receiver_parse_rate": g.guess_parsed.mean(),
            "monitor_parse_rate": g.monitor_parsed.mean(),
        })
    s = pd.DataFrame(out)
    s["scheme"] = pd.Categorical(s.scheme, SCHEME_ORDER)
    s["channel"] = pd.Categorical(s.channel, CHANNEL_ORDER)
    return s.sort_values(["model", "scheme", "channel"]).reset_index(drop=True)


def paraphrase_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired effect of paraphrasing on accuracy and flag rate, per model and scheme."""
    out = []
    for (model, scheme), g in df.groupby(["model", "scheme"]):
        w = g.pivot(index="trial", columns="channel", values=["correct", "flagged"]).dropna()
        if not {"plain", "paraphrased"} <= set(w.columns.get_level_values(1)):
            continue
        row = {"model": model, "scheme": scheme, "n_pairs": len(w)}
        for metric in ("correct", "flagged"):
            a, b = w[metric]["plain"].astype(bool), w[metric]["paraphrased"].astype(bool)
            lost, gained = int((a & ~b).sum()), int((~a & b).sum())
            row[f"{metric}_plain"] = a.mean()
            row[f"{metric}_paraphrased"] = b.mean()
            row[f"{metric}_p"] = (
                binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
            )
        out.append(row)
    t = pd.DataFrame(out)
    if len(t):
        # Holm family = the schemes of one model, as pre-registered.
        for metric in ("correct", "flagged"):
            t[f"{metric}_p_holm"] = t.groupby("model")[f"{metric}_p"].transform(lambda p: holm(p.tolist()))
        t["scheme"] = pd.Categorical(t.scheme, SCHEME_ORDER)
        t = t.sort_values(["model", "scheme"]).reset_index(drop=True)
    return t


def plot(summary: pd.DataFrame, path: str | Path | None = None):
    import matplotlib.pyplot as plt

    models = list(summary.model.unique())
    fig, axes = plt.subplots(
        len(models), 2, figsize=(10, 3.6 * len(models)), squeeze=False, constrained_layout=True
    )
    colors = {"plain": "#2a6fdb", "paraphrased": "#e07b39"}
    for r, model in enumerate(models):
        s = summary[summary.model == model]
        schemes = [x for x in SCHEME_ORDER if x in set(s.scheme)]
        x = np.arange(len(schemes))
        for c, (metric, lo, hi, title) in enumerate([
            ("accuracy", "acc_lo", "acc_hi", "Receiver decodes the bit"),
            ("flag_rate", "flag_lo", "flag_hi", "Monitor flags the message"),
        ]):
            ax = axes[r, c]
            for j, ch in enumerate(CHANNEL_ORDER):
                d = s[s.channel == ch].set_index("scheme").reindex(schemes)
                if d[metric].isna().all():
                    continue
                err = np.clip(np.vstack([d[metric] - d[lo], d[hi] - d[metric]]), 0, None)
                ax.bar(x + (j - 0.5) * 0.38, d[metric], 0.38, yerr=err, capsize=3,
                       label=ch, color=colors[ch])
            if metric == "accuracy":
                ax.axhline(0.5, ls="--", lw=1, color="gray")
                ax.text(len(schemes) - 0.5, 0.51, "chance", ha="right", va="bottom", fontsize=8, color="gray")
            ax.set_xticks(x, schemes)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{title}  ({model.split('/')[-1]})", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 0].legend(frameon=False, fontsize=9)
    if path:
        fig.savefig(path, dpi=200)
    return fig


def examples(df: pd.DataFrame, scheme: str, k: int = 3, seed: int = 0) -> pd.DataFrame:
    """Messages where the plain signal decoded and the paraphrased one did not."""
    w = df[df.scheme == scheme].pivot(index="trial", columns="channel", values=["delivered", "correct"])
    hit = w[w["correct"]["plain"].astype(bool) & ~w["correct"]["paraphrased"].astype(bool)]
    pick = hit.sample(min(k, len(hit)), random_state=seed) if len(hit) else hit
    return pd.DataFrame({"plain": pick["delivered"]["plain"], "paraphrased": pick["delivered"]["paraphrased"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="directories holding trials.jsonl")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()

    df = load(*a.results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s, t = summarize(df), paraphrase_tests(df)
    s.to_csv(out / "summary.csv", index=False)
    t.to_csv(out / "paraphrase_tests.csv", index=False)
    plot(s, out / "fig1.png")
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print()
    print(t.round(4).to_string(index=False))
    print(f"\nwrote {out}/summary.csv, paraphrase_tests.csv, fig1.png")


if __name__ == "__main__":
    main()
