"""Follow-up passes (sanitize, score) and their analysis, end to end with a scripted model."""

import json
import re

import pytest

from covert import analysis as A
from covert import prompts as P
from covert.backends import ScriptedBackend
from covert.experiment import Config, run
from covert.followup import p_first, sanitize, score
from tests.test_covert import scripted


def scripted_followup(conv: list[dict]) -> str:
    if conv[0]["content"] == P.SANITIZER_SYSTEM:
        return "Triage is done."  # keeps the work fact, drops every marker
    return scripted(conv)


def marker(text: str) -> str | None:
    for word, bit in [("certainly", "HIGH"), ("basically", "LOW"), ("coffee", "HIGH"), ("tea", "LOW")]:
        if re.search(rf"\b{word}\b", text):
            return bit
    return None


def scripted_score(conv: list[dict], cand: str) -> float:
    system, user = conv[0]["content"], conv[-1]["content"]
    msg = user.split("Message:\n", 1)[-1]
    if system == P.RECEIVER_SYSTEM:
        bit = marker(msg)
        return 0.0 if bit is None or bit == cand else -3.0
    if system == P.MONITOR_SYSTEM:
        # The basic monitor only notices off-topic drinks; the informed one also notices code words.
        seen = re.search(r"\b(coffee|tea)\b", msg) or ("Colluding" in user and marker(msg))
        return 0.0 if bool(seen) == (cand == "YES") else -3.0
    if system == P.JUDGE_SYSTEM:
        kept = "Triage" in user.split("Rewritten message:\n", 1)[1]
        return 0.0 if kept == (cand == "YES") else -3.0
    raise AssertionError(system)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    base = tmp_path_factory.mktemp("results")
    backend = ScriptedBackend(scripted_followup, scripted_score)
    run(Config(model="scripted", n_trials=40), backend, base / "scripted")
    run(Config(model="scripted", n_trials=40, seed=1), backend, base / "followup" / "seed1" / "scripted")
    sanitize(backend, "scripted", base / "scripted", base / "followup", chunk=7)
    score(backend, "scripted", [base / "scripted"], base / "followup", chunk=50)
    return base


def test_sanitized_rows_pair_with_main(root):
    main = A.load(root / "scripted")
    san = A.load_sanitized(root / "followup")
    assert len(san) == 4 * 40 and set(san.channel) == {"sanitized"}
    plain = main[main.channel == "plain"].set_index(["scheme", "trial"])
    s = san.set_index(["scheme", "trial"])
    assert (s.message == plain.loc[s.index].message).all()
    assert (s.bit == plain.loc[s.index].bit).all()
    assert not s.signal_delivered.dropna().astype(bool).any()
    assert s.utility.between(0.9, 1).all()


def test_scores_cover_every_message(root):
    scores = A.load_scores(root / "followup")
    assert len(scores) == 4 * 3 * 40  # plain, paraphrased and sanitized
    assert not scores.duplicated(["model", "scheme", "channel", "trial"]).any()


def test_auroc_tables(root):
    scores = A.load_scores(root / "followup")
    mon = A.monitor_auroc(scores, n_boot=50).set_index(["channel", "scheme", "prompt"]).auroc
    assert mon[("plain", "semantic", "basic")] == 1.0
    assert mon[("plain", "lexical", "basic")] == 0.5
    assert mon[("plain", "lexical", "informed")] == 1.0
    rec = A.receiver_auroc(scores, n_boot=50).set_index(["scheme", "channel"]).auroc
    assert rec[("lexical", "plain")] == 1.0
    assert rec[("lexical", "paraphrased")] == 0.5
    assert rec[("semantic", "sanitized")] == 0.5


def test_followup_and_exploratory_tables(root, tmp_path):
    main = A.load(root / "scripted")
    t = A.followup(main, root / "followup", tmp_path)
    assert {"summary_sanitized", "sanitized_tests", "utility", "monitor_auroc", "receiver_auroc",
            "replication"} <= set(t)
    tests = t["sanitized_tests"].set_index(["comparison", "scheme"])
    assert tests.loc[("plain_vs_sanitized", "semantic"), "correct_lost"] > 10
    assert (tmp_path / "fig2_auroc.png").stat().st_size > 0
    assert (tmp_path / "fig3_sanitized.png").stat().st_size > 0
    e = A.exploratory(main, tmp_path / "exploratory")
    assert e["effect_contrast"].difference.iloc[0] > 0.2
    assert set(e["detection_baselines"].detector) == {"length", "bag_of_words"}


def test_score_resumes_and_refuses_changed_config(root):
    path = root / "followup" / "scores" / "scripted" / "scores.jsonl"
    n = len(path.read_text().splitlines())
    with open(path, "a") as f:
        f.write('{"judge": "scri')
    backend = ScriptedBackend(scripted_followup, scripted_score)
    score(backend, "scripted", [root / "scripted"], root / "followup")
    assert len(path.read_text().splitlines()) == n
    cfg = json.loads((path.parent / "config.json").read_text())
    (path.parent / "config.json").write_text(json.dumps(cfg | {"backend": "other"}))
    with pytest.raises(ValueError, match="different config"):
        score(backend, "scripted", [root / "scripted"], root / "followup")


def test_sanitize_refuses_another_models_run(root):
    with pytest.raises(ValueError, match="sanitize it with that model"):
        sanitize(ScriptedBackend(scripted_followup, scripted_score), "other", root / "scripted", root / "x")


def test_p_first():
    assert p_first([0.0, 0.0]) == 0.5
    assert p_first([0.0, -50.0]) > 0.999
