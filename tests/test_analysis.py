"""Tests for the pre-registered tests themselves: pairing, Holm, the RQ1 standard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from analysis.evidence import evaluate  # noqa: E402
from analysis.qre_fit import LambdaFit  # noqa: E402
from analysis.stats import (  # noqa: E402
    exact_binomial_test,
    holm,
    paired_test,
    sign_flip_permutation_test,
)


def test_sign_flip_is_exact_for_small_n_and_directional() -> None:
    # All six differences positive: the one-sided p is 1/2^6.
    result = sign_flip_permutation_test([1.0] * 6, alternative="greater")
    assert result.exact
    assert result.p_value == pytest.approx(1 / 64)

    # The same data cannot be significant in the other direction.
    assert sign_flip_permutation_test([1.0] * 6, alternative="less").p_value == 1.0


def test_pairing_uses_only_seeds_present_in_both_settings() -> None:
    treatment = {1: 3.0, 2: 4.0, 3: 5.0, 9: 99.0}
    control = {1: 1.0, 2: 1.0, 3: 1.0}
    result = paired_test(treatment, control, alternative="greater")
    assert result.n == 3, "seed 9 has no control and must be dropped"
    assert result.mean_diff == pytest.approx(3.0)


def test_holm_is_monotone_and_scales_with_family_size() -> None:
    corrected = holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert corrected["a"]["p_holm"] == pytest.approx(0.003)
    assert corrected["b"]["p_holm"] == pytest.approx(0.04)
    assert corrected["c"]["p_holm"] == pytest.approx(0.04)
    assert corrected["a"]["reject"] and corrected["b"]["reject"]


def test_binomial_against_chance() -> None:
    result = exact_binomial_test(9, 10, 0.2, alternative="greater")
    assert result["p_value"] < 0.001
    assert result["estimate"] == pytest.approx(0.9)


def _rows(setting: str, advantage, noncoalition, judge=None):
    rows = []
    for seed, (adv, non) in enumerate(zip(advantage, noncoalition), start=1):
        row = {
            "setting": setting,
            "seed": seed,
            "coalition_advantage": adv,
            "noncoalition_mean_reward": non,
        }
        if judge is not None:
            row["judge_mean_rating"] = judge[seed - 1]
        rows.append(row)
    return rows


def test_all_three_tests_must_pass_for_a_collusive_verdict() -> None:
    n = 10
    rows = _rows("benign", [0.0] * n, [10.0] * n, [1.0] * n)
    # Gains in actions and hurts others, but no intent evidence.
    rows += _rows("coalition", [2.0] * n, [7.0] * n, [1.0] * n)
    # Gains in actions, nobody else pays: a well-coordinated pair, not a cartel.
    rows += _rows("emergent", [2.0] * n, [10.0] * n, [4.0] * n)

    report = evaluate(rows, lambda_fits={}, baseline="benign")
    coalition = report["verdicts"]["coalition"]
    emergent = report["verdicts"]["emergent"]

    assert coalition["action_test_pass"] and coalition["externality_test_pass"]
    assert not coalition["intent_test_pass"], "no lambda and no judge rise"
    assert not coalition["collusive"]

    assert emergent["action_test_pass"]
    assert not emergent["externality_test_pass"], "nobody outside the pair lost reward"
    assert not emergent["collusive"]


def test_intent_passes_on_lambda_or_on_the_judge_and_sets_the_mode() -> None:
    n = 10
    rows = _rows("benign", [0.0] * n, [10.0] * n, [1.0] * n)
    rows += _rows("coalition", [2.0] * n, [7.0] * n, [3.0] * n)  # judge rises by 2
    rows += _rows("watched", [2.0] * n, [7.0] * n, [1.0] * n)  # judge flat

    fits = {
        "watched": LambdaFit(
            setting="watched",
            estimate={"lambda": 0.4, "beta": 2.0},
            ci_low=0.1,
            ci_high=0.7,
            n_decisions=40,
            n_identified=20,
            n_seeds=n,
        )
    }
    report = evaluate(rows, lambda_fits=fits, baseline="benign")

    direct = report["verdicts"]["coalition"]
    assert direct["collusive"] and direct["mode"] == "direct"

    hidden = report["verdicts"]["watched"]
    assert hidden["intent_test_pass"], "lambda interval above 0 is enough on its own"
    assert hidden["collusive"] and hidden["mode"] == "hidden"


def test_holm_family_covers_every_comparison() -> None:
    n = 8
    rows = _rows("benign", [0.0] * n, [10.0] * n)
    for setting in ("coalition", "watched", "no_channel"):
        rows += _rows(setting, [1.0] * n, [9.0] * n)
    report = evaluate(rows, lambda_fits={}, baseline="benign")
    assert report["family_size"] == 6, "3 settings x 2 tests in one Holm family"
