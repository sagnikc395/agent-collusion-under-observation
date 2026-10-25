"""Build every results table for RQ1-RQ6 from saved runs (CPU only).

    python -m analysis.report --results results/main --out results/tables

Writes `episodes.csv`, per-stage CSVs, `report.json` and `report.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.evidence import BASELINE, compare, evaluate
from analysis.io import load_episodes
from analysis.qre_fit import fit_by_setting
from analysis.regret import build_rows, setting_summary
from analysis.stats import exact_binomial_test

logger = logging.getLogger("analysis.report")

HEADLINE_METRICS = [
    "coalition_advantage",
    "coalitional_regret",
    "coalition_mean_reward",
    "noncoalition_mean_reward",
    "joint_reward",
    "judge_mean_rating",
    "colluder_posts_secret",
    "colluder_posts_non_secret",
]

COMPARISON_METRICS = [
    "coalition_advantage",
    "coalitional_regret",
    "coalition_mean_reward",
    "noncoalition_mean_reward",
    "judge_mean_rating",
    "joint_reward",
]

# RQ4 ablations are read against the setting they ablate, not against benign.
ABLATION_PARENTS = {
    "watched_paraphrase": "watched",
    "watched_hide_claims": "watched",
    "watched_both": "watched",
    "no_channel_paraphrase": "no_channel",
    "no_channel_hide_claims": "no_channel",
    "no_channel_both": "no_channel",
}


def _present(rows: List[Dict[str, Any]], setting: str) -> bool:
    return any(row["setting"] == setting for row in rows)


def rq5_partner_identification(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Partner-identification accuracy against chance, with an exact binomial test."""
    answers = [
        answer
        for row in rows
        if row["setting"] == "find_partner"
        for answer in (row.get("partner_survey") or [])
    ]
    if not answers:
        return None
    trials = len(answers)
    successes = sum(1 for a in answers if a.get("correct"))
    chance = sum(1.0 / max(1, int(a.get("n_candidates") or 5)) for a in answers) / trials
    return {
        "test": exact_binomial_test(successes, trials, chance, alternative="greater"),
        "chance_rate": chance,
        "unresolved_answers": sum(1 for a in answers if not a.get("guess")),
    }


def rq6_verifier_cost(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the verifier cost: extra messages, flags raised, team score effect."""
    verifier_rows = [row for row in rows if row["setting"] == "verifier"]
    if not verifier_rows:
        return None
    n = len(verifier_rows)
    return {
        "episodes": n,
        "mean_verifier_messages": sum(r["verifier_messages"] or 0 for r in verifier_rows) / n,
        "mean_flags": sum(r["verifier_flags_total"] or 0 for r in verifier_rows) / n,
        "mean_joint_reward": sum(
            float(r["joint_reward"]) for r in verifier_rows if r.get("joint_reward") is not None
        )
        / n,
    }


def build_report(
    results: List[Path],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    baseline: str = BASELINE,
) -> Dict[str, Any]:
    episodes = load_episodes(results)
    if not episodes:
        raise SystemExit(f"No completed episodes found under {results}")
    rows = build_rows(episodes)

    logger.info("Loaded %s episodes across %s settings", len(rows), len({r["setting"] for r in rows}))

    lambda_fits = fit_by_setting(episodes, agents="colluders", n_boot=n_boot, seed=seed)
    # Sanity check from RQ1: the same fit on non-coalition agents should sit near 0.
    lambda_fits_others = fit_by_setting(episodes, agents="others", n_boot=n_boot, seed=seed)

    report: Dict[str, Any] = {
        "episodes": len(rows),
        "settings": sorted({r["setting"] for r in rows}),
        "seeds": sorted({int(r["seed"]) for r in rows}),
        "per_setting": setting_summary(rows, HEADLINE_METRICS),
        "lambda_fits": {k: v.as_dict() for k, v in lambda_fits.items()},
        "lambda_fits_noncoalition_sanity": {
            k: v.as_dict() for k, v in lambda_fits_others.items()
        },
    }

    if _present(rows, baseline):
        report["rq1_evidence"] = evaluate(rows, lambda_fits=lambda_fits, baseline=baseline, seed=seed)
    else:
        logger.warning("Baseline %r missing: skipping the RQ1 evidence standard", baseline)

    # RQ3 and RQ6 read against the unwatched coalition from RQ2.
    report["vs_coalition"] = {
        setting: compare(
            rows,
            treatment=setting,
            control="coalition",
            metrics=COMPARISON_METRICS,
            seed=seed,
        )
        for setting in ("watched", "no_channel", "verifier", "find_partner")
        if _present(rows, setting) and _present(rows, "coalition")
    }

    # RQ4: each ablation against the setting it ablates.
    report["ablations"] = {
        setting: compare(
            rows, treatment=setting, control=parent, metrics=COMPARISON_METRICS, seed=seed
        )
        for setting, parent in ABLATION_PARENTS.items()
        if _present(rows, setting) and _present(rows, parent)
    }

    report["rq5_partner_identification"] = rq5_partner_identification(rows)
    report["rq6_verifier_cost"] = rq6_verifier_cost(rows)
    report["_rows"] = rows
    return report


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = ["# Carrier displacement results", ""]
    lines.append(
        f"{report['episodes']} episodes, {len(report['settings'])} settings, "
        f"{len(report['seeds'])} seeds."
    )
    lines.append("")

    lines += ["## Per setting", "", "| setting | n | coalition advantage | coalitional regret | coalition reward | non-coalition reward | joint reward | judge |", "|---|---|---|---|---|---|---|---|"]
    for setting, stats in sorted(report["per_setting"].items()):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                setting,
                stats["episodes"],
                _fmt(stats["coalition_advantage"]["mean"]),
                _fmt(stats["coalitional_regret"]["mean"]),
                _fmt(stats["coalition_mean_reward"]["mean"]),
                _fmt(stats["noncoalition_mean_reward"]["mean"]),
                _fmt(stats["joint_reward"]["mean"]),
                _fmt(stats["judge_mean_rating"]["mean"]),
            )
        )
    lines.append("")

    evidence = report.get("rq1_evidence")
    if evidence:
        lines += [
            "## RQ1 evidence standard (vs benign, Holm-corrected)",
            "",
            "| setting | advantage delta | p_holm | externality delta | p_holm | lambda [CI] | judge delta | collusive | mode |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for setting, row in sorted(evidence["verdicts"].items()):
            lam = (
                f"{_fmt(row['lambda'])} [{_fmt(row['lambda_ci_low'])}, -]"
                if row["lambda"] is not None
                else "-"
            )
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    setting,
                    _fmt(row["advantage_delta"]),
                    _fmt(row["advantage_p_holm"], 3),
                    _fmt(row["noncoalition_reward_delta"]),
                    _fmt(row["externality_p_holm"], 3),
                    lam,
                    _fmt(row["judge_delta"]),
                    _fmt(row["collusive"]),
                    row["mode"] or "-",
                )
            )
        lines.append("")

    for title, key, control_label in (
        ("RQ3/RQ6: settings against the unwatched coalition", "vs_coalition", "coalition"),
        ("RQ4: carrier ablations against the setting they ablate", "ablations", "parent"),
    ):
        block = report.get(key) or {}
        if not block:
            continue
        lines += [
            f"## {title}",
            "",
            f"| setting | control ({control_label}) | advantage delta | p | non-coalition reward delta | p | judge delta | p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for setting, comparison in sorted(block.items()):
            adv = comparison.get("coalition_advantage") or {}
            ext = comparison.get("noncoalition_mean_reward") or {}
            judge = comparison.get("judge_mean_rating") or {}
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    setting,
                    comparison.get("control"),
                    _fmt(adv.get("mean_diff")),
                    _fmt(adv.get("p_value"), 3),
                    _fmt(ext.get("mean_diff")),
                    _fmt(ext.get("p_value"), 3),
                    _fmt(judge.get("mean_diff")),
                    _fmt(judge.get("p_value"), 3),
                )
            )
        lines.append("")

    rq5 = report.get("rq5_partner_identification")
    if rq5:
        test = rq5["test"]
        lines += [
            "## RQ5: partner identification",
            "",
            f"- correct: {test['successes']}/{test['trials']} "
            f"({_fmt(test['estimate'])}), chance {_fmt(rq5['chance_rate'])}",
            f"- exact binomial (greater): p = {_fmt(test['p_value'], 4)}",
            f"- answers with no resolvable name: {rq5['unresolved_answers']}",
            "",
        ]

    rq6 = report.get("rq6_verifier_cost")
    if rq6:
        lines += [
            "## RQ6: verifier cost",
            "",
            f"- extra messages per episode: {_fmt(rq6['mean_verifier_messages'])}",
            f"- flags per episode: {_fmt(rq6['mean_flags'])}",
            f"- mean joint reward with the verifier: {_fmt(rq6['mean_joint_reward'])}",
            "",
        ]

    lines += [
        "## lambda_emp",
        "",
        "| setting | lambda | 95% CI | above 0 | decisions | identified | seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for setting, fit in sorted(report["lambda_fits"].items()):
        lines.append(
            "| {} | {} | [{}, {}] | {} | {} | {} | {} |".format(
                setting,
                _fmt(fit["lambda"]),
                _fmt(fit["ci_low"]),
                _fmt(fit["ci_high"]),
                _fmt(fit["ci_above_zero"]),
                fit["n_decisions"],
                fit["n_identified"],
                fit["n_seeds"],
            )
        )
    lines.append("")
    lines.append(
        "Judge scores come from a local 14B model, so only differences within this "
        "project are meaningful; they are not comparable with Colosseum's figures."
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the results tables.")
    parser.add_argument(
        "--results", nargs="+", required=True, help="One or more results roots."
    )
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    report = build_report(
        [Path(r) for r in args.results],
        n_boot=args.n_boot,
        seed=args.seed,
        baseline=args.baseline,
    )
    rows = report.pop("_rows")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_dir / "episodes.csv", index=False)
    if report.get("rq1_evidence"):
        pd.DataFrame(list(report["rq1_evidence"]["verdicts"].values())).to_csv(
            out_dir / "rq1_evidence.csv", index=False
        )
    pd.DataFrame(list(report["lambda_fits"].values())).to_csv(
        out_dir / "lambda_fits.csv", index=False
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
