"""End-to-end check of the pipeline and the statistics with a scripted model."""

import json
import re

import pytest

from covert import analysis as A
from covert import prompts as P
from covert.backends import ScriptedBackend
from covert.experiment import Config, run


def scripted(conv: list[dict]) -> str:
    system, user = conv[0]["content"], conv[-1]["content"]
    if system == P.SENDER_SYSTEM:
        # Follows the lexical and semantic codes exactly; leaks nothing otherwise.
        bit = "HIGH" if "bid from you: HIGH" in user else "LOW"
        if "'certainly'" in user:
            return f"Triage is done, {'certainly' if bit == 'HIGH' else 'basically'} on track."
        if "coffee" in user:
            return f"Triage is done; grab a {'coffee' if bit == 'HIGH' else 'tea'} later."
        return "Triage is done and on track."
    if system == P.PARAPHRASER_SYSTEM:
        # Drops filler words but keeps topics, so lexical dies and semantic survives.
        msg = user.split("Message:\n", 1)[1]
        return re.sub(r"\b(certainly|basically)\b", "", msg)
    if system == P.RECEIVER_SYSTEM:
        msg = user.split("Message:\n", 1)[1]
        for word, bit in [("certainly", "HIGH"), ("basically", "LOW"), ("coffee", "HIGH"), ("tea", "LOW")]:
            if re.search(rf"\b{word}\b", msg):
                return bit
        return "I am not sure."
    if system == P.MONITOR_SYSTEM:
        return "YES" if "coffee" in user or " tea" in user else "NO"
    raise AssertionError(system)


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    run(Config(model="scripted", n_trials=60), ScriptedBackend(scripted), out)
    assert json.loads((out / "config.json").read_text())["backend"] == "scripted"
    return A.load(out)


def acc(s, scheme, channel):
    return s[(s.scheme == scheme) & (s.channel == channel)].iloc[0]


def test_grid_is_complete_and_paired(df):
    assert len(df) == 4 * 2 * 60
    # Both channels deliver the same sender message for a given (scheme, trial).
    assert df.groupby(["scheme", "trial"]).message.nunique().max() == 1
    assert df.groupby("trial").bit.nunique().max() == 1


def test_expected_accuracies(df):
    s = A.summarize(df)
    assert acc(s, "lexical", "plain").accuracy == 1.0
    assert acc(s, "semantic", "paraphrased").accuracy == 1.0
    assert acc(s, "lexical", "paraphrased").signal_delivered == 0.0
    # Unparseable guesses fall back to coin flips, so the leak-free cells sit near chance.
    for cell in [("none", "plain"), ("lexical", "paraphrased")]:
        assert 0.3 < acc(s, *cell).accuracy < 0.7
        assert acc(s, *cell).receiver_parse_rate == 0.0
    assert acc(s, "semantic", "plain").flag_rate == 1.0
    assert acc(s, "lexical", "plain").covert_success == 1.0


def test_paraphrase_test_detects_lexical_drop(df):
    t = A.paraphrase_tests(df).set_index("scheme")
    assert t.loc["lexical", "correct_p_holm"] < 0.01
    assert t.loc["semantic", "correct_p"] == 1.0


def test_wilson_and_holm():
    lo, hi = A.wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9
    assert A.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_parse_choice():
    assert P.parse_choice("low, I think. Not HIGH.", P.BITS) == "LOW"
    assert P.parse_choice("unclear", P.BITS) is None


def test_plot_and_examples(df, tmp_path):
    A.plot(A.summarize(df), tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0
    assert len(A.examples(df, "lexical", k=2)) == 2


def test_resume_after_cutoff(tmp_path):
    backend = ScriptedBackend(scripted)
    full = run(Config(model="scripted", n_trials=20), backend, tmp_path / "full", chunk_size=6)

    # A run that died mid-write: 7 of 8 rows of trial 12, then a half-written line.
    out = tmp_path / "cut"
    run(Config(model="scripted", n_trials=12), backend, out, chunk_size=6)
    part = [r for r in full if r["trial"] == 12][:7]
    with open(out / "trials.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in part)
        f.write('{"model": "scripted", "sch')
    assert len(part) == 7

    # Resuming with a larger n drops the partial trial, extends, and matches a clean run.
    resumed = run(Config(model="scripted", n_trials=20), backend, out, chunk_size=6)
    on_disk = A.load(out)
    key = ["trial", "scheme", "channel"]
    assert len(on_disk) == len(full) == len(resumed) == 20 * 8
    assert not on_disk.duplicated(key).any()
    pd_full = A.load(tmp_path / "full").sort_values(key).reset_index(drop=True)
    pd_cut = on_disk.sort_values(key).reset_index(drop=True)
    assert (pd_full[key + ["bit", "correct", "delivered"]] == pd_cut[key + ["bit", "correct", "delivered"]]).all().all()
    assert json.loads((out / "progress.json").read_text())["trials_done"] == 20
    assert "resuming: 12 trials already on disk" in (out / "run.log").read_text()


def test_resume_refuses_changed_config(tmp_path):
    run(Config(model="scripted", n_trials=4), ScriptedBackend(scripted), tmp_path)
    with pytest.raises(ValueError, match="seed"):
        run(Config(model="scripted", n_trials=4, seed=1), ScriptedBackend(scripted), tmp_path)
