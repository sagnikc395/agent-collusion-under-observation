#!/usr/bin/env python3
"""Write a run's results back into the notebook that produced them.

Colab keeps the notebook and the results in different places: the tables land
under ``RESULTS_ROOT`` on Drive, while the notebook is just a tab in the
browser. That splits the record of a run in two. This script folds the tables
back into the ``.ipynb`` itself, so a single file carries the plan, the cells
that ran and what came out -- which is what gets attached to the write-up.

    python -m scripts.embed_results --notebook notebooks/colab_runner.ipynb \
        --tables results/sprint/tables --results results/sprint

Re-running replaces the embedded section instead of appending a second copy:
every cell it writes carries ``metadata.carriers_embedded_results``.

CPU-only, stdlib-only, and it never touches the cells you wrote.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MARKER = "carriers_embedded_results"
HEADING = "## Results (written back by `scripts/embed_results.py`)"
# episodes.csv is one row per episode; the others are one row per setting.
CSV_TABLES = ("rq1_evidence.csv", "lambda_fits.csv")
MAX_CSV_ROWS = 40


def _git_describe(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--pretty=%h %s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _csv_to_markdown(path: Path, max_rows: int = MAX_CSV_ROWS) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [f"_{path.name} is empty._"]
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body[:max_rows]:
        cells = [c.replace("|", "\\|") for c in row]
        cells += [""] * (len(header) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    if len(body) > max_rows:
        lines.append(f"| _... {len(body) - max_rows} more rows in {path.name}_ |")
    return lines


def _episode_counts(results_roots: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for root in results_roots:
        for metrics in sorted(root.rglob("metrics.json")):
            # <root>/<setting>/<run>/metrics.json, as the runner lays it out.
            setting = metrics.parent.parent.name
            counts[setting] = counts.get(setting, 0) + 1
    return counts


def build_cells(
    tables_dirs: List[Path],
    results_roots: List[Path],
    repo: Path,
    note: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """The markdown cells to splice into the notebook, in order."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        HEADING,
        "",
        f"- embedded: {stamp}",
        f"- commit: `{_git_describe(repo)}`",
    ]
    if tables_dirs:
        lines.append("- tables: " + ", ".join(f"`{d}`" for d in tables_dirs))
    counts = _episode_counts(results_roots)
    if counts:
        total = sum(counts.values())
        lines += [
            f"- episodes with `metrics.json`: {total}",
            "",
            "| setting | episodes |",
            "| --- | --- |",
        ]
        lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    if note:
        lines += ["", note]
    lines += [
        "",
        "Judge scores are from a local 4-bit 7B model: only differences within "
        "this project are meaningful.",
    ]
    cells = [_markdown_cell(lines)]

    for tables_dir in tables_dirs:
        block = [f"### `{tables_dir}`", ""]
        report_md = tables_dir / "report.md"
        if report_md.is_file():
            block += [report_md.read_text(encoding="utf-8").rstrip(), ""]
        else:
            block += [f"_No `report.md` under {tables_dir}._", ""]
        for name in CSV_TABLES:
            path = tables_dir / name
            if not path.is_file():
                continue
            block += [f"#### `{name}`", ""] + _csv_to_markdown(path) + [""]
        cells.append(_markdown_cell(block))

        report_json = tables_dir / "report.json"
        if report_json.is_file():
            payload = report_json.read_text(encoding="utf-8").rstrip()
            cells.append(
                _markdown_cell(
                    [
                        f"<details><summary><code>{tables_dir}/report.json</code>"
                        "</summary>",
                        "",
                        "```json",
                        payload,
                        "```",
                        "",
                        "</details>",
                    ]
                )
            )
    return cells


def _markdown_cell(lines: List[str]) -> Dict[str, Any]:
    text = "\n".join(lines).rstrip() + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {MARKER: True},
        "source": text.splitlines(keepends=True),
    }


def embed(
    notebook: Path,
    tables_dirs: List[Path],
    results_roots: List[Path],
    out: Path,
    repo: Path,
    note: Optional[str] = None,
) -> Path:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    kept = [c for c in nb["cells"] if not c.get("metadata", {}).get(MARKER)]
    nb["cells"] = kept + build_cells(tables_dirs, results_roots, repo, note=note)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, help="The .ipynb to write into.")
    parser.add_argument(
        "--tables", nargs="*", default=[], help="Tables dirs from analysis.report."
    )
    parser.add_argument(
        "--results", nargs="*", default=[], help="Results roots, for episode counts."
    )
    parser.add_argument(
        "--out", default=None, help="Output notebook (default: in place)."
    )
    parser.add_argument("--repo", default=".", help="Repo to read the commit from.")
    parser.add_argument("--note", default=None, help="One extra markdown line.")
    args = parser.parse_args(argv)

    notebook = Path(args.notebook)
    out = Path(args.out) if args.out else notebook
    written = embed(
        notebook,
        [Path(t) for t in args.tables],
        [Path(r) for r in args.results],
        out,
        Path(args.repo),
        note=args.note,
    )
    print(f"Wrote results into {written}")


if __name__ == "__main__":
    main()
