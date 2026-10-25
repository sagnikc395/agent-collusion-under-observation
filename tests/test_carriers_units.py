"""Unit tests for the carrier ablations, the verifier and the survey parser."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carriers.hide_claims import strip_action_events, strip_joint_assignment  # noqa: E402
from carriers.llm import SimpleChat  # noqa: E402
from carriers.paraphrase import Paraphraser  # noqa: E402
from carriers.protocol import CarrierProtocol  # noqa: E402
from carriers.runner import _parse_partner_answer  # noqa: E402
from carriers.verifier import TrustedVerifier, find_duplicate_ticket_claims  # noqa: E402
from terrarium.core.blackboard import Megaboard  # noqa: E402


class _ScriptedClient:
    """Returns a fixed rewrite; counts calls so caching can be checked."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def generate_response(self, input, params):
        self.calls += 1
        return {"usage": {}}, self.reply

    @staticmethod
    def get_usage(response, usage):
        return usage


def _chat(reply: str) -> SimpleChat:
    return SimpleChat(client=_ScriptedClient(reply), model="test")


def _megaboard_with_public_and_secret():
    megaboard = Megaboard()
    public = megaboard.add_blackboard(["A", "B", "C"])
    secret = megaboard.add_blackboard(["A", "B"], {"secret_channel": True})
    megaboard.post(public, "A", "communication", {"content": "I take ISSUE-0001::test (3 hours)."})
    megaboard.post(public, "B", "communication", {"content": "I take ISSUE-0001::test too."})
    megaboard.post(secret, "A", "communication", {"content": "Secret plan for ISSUE-0001::test."})
    return megaboard, public, secret


def test_paraphrase_rewrites_other_agents_only_and_caches() -> None:
    megaboard, public, secret = _megaboard_with_public_and_secret()
    client = _ScriptedClient("Rewritten text about ISSUE-0001::test.")
    paraphraser = Paraphraser(chat=SimpleChat(client=client, model="test"))
    protocol = CarrierProtocol(config={}, paraphraser=paraphraser, megaboard=megaboard)
    protocol.secret_blackboard_id = secret

    contexts = asyncio.run(
        protocol._prefetch_blackboard_events("A", phase="planning", iteration=1)
    )
    public_text = contexts[str(public)]
    assert "Rewritten text" in public_text, "B's message should be paraphrased for A"
    assert "I take ISSUE-0001::test (3 hours)." in public_text, "A keeps its own words"
    assert "Secret plan" in contexts[str(secret)], "the private channel is untouched"

    # A second reader must see the same rewrite without a second model call.
    asyncio.run(protocol._prefetch_blackboard_events("C", phase="planning", iteration=1))
    assert client.calls == 2, f"expected one call per other-authored message, got {client.calls}"


def test_paraphrase_falls_back_when_a_ticket_id_is_lost() -> None:
    paraphraser = Paraphraser(chat=_chat("I will handle the first ticket."))
    kept = paraphraser.paraphrase("e1", "I take ISSUE-0002::docs.")
    assert kept == "I take ISSUE-0002::docs."
    assert paraphraser.stats()["fallbacks_to_original"] == 1


def test_hide_claims_removes_action_traces_and_other_agents_assignments() -> None:
    events = [
        {"kind": "communication", "agent": "A", "payload": {"content": "hi"}},
        {"kind": "action_executed", "agent": "B", "payload": {"action_type": "assign_task"}},
    ]
    assert len(strip_action_events(events)) == 1

    context = {
        "agent_name": "A",
        "joint_assignment": {"A": "ISSUE-0001::test", "B": "ISSUE-0002::docs"},
    }
    hidden = strip_joint_assignment(context)
    assert hidden["joint_assignment"] == {"A": "ISSUE-0001::test"}
    assert context["joint_assignment"] == {
        "A": "ISSUE-0001::test",
        "B": "ISSUE-0002::docs",
    }, "the original context must not be mutated"


def test_hide_claims_hides_action_events_from_the_prompt() -> None:
    megaboard, public, _ = _megaboard_with_public_and_secret()
    megaboard.post(
        public,
        "B",
        "action_executed",
        {"action_type": "assign_task", "action_params": {"task_id": "ISSUE-0001::test"}},
    )
    protocol = CarrierProtocol(config={}, hide_claims=True, megaboard=megaboard)
    contexts = asyncio.run(
        protocol._prefetch_blackboard_events("A", phase="execution", iteration=1)
    )
    assert "action_executed" not in contexts[str(public)]


def test_verifier_flags_duplicate_claims_once() -> None:
    megaboard, public, _ = _megaboard_with_public_and_secret()
    duplicates = find_duplicate_ticket_claims(megaboard.blackboards[public].logs)
    assert duplicates == [{"ticket": "ISSUE-0001::TEST", "agents": ["A", "B"]}]

    verifier = TrustedVerifier.attach(
        chat=_chat('{"flags": []}'), megaboard=megaboard, blackboard_id=public
    )
    first = verifier.run_round(planning_round=1)
    assert len(first) == 1 and first[0]["type"] == "duplicate_claim"
    assert verifier.run_round(planning_round=2) == [], "the same clash is not re-flagged"
    assert verifier.summary()["verifier_messages"] == 1
    posted = megaboard.blackboards[public].logs[-1]
    assert posted["agent"] == "Verifier" and "ISSUE-0001::TEST" in posted["payload"]["content"]


def test_verifier_ignores_mentions_that_are_not_claims() -> None:
    megaboard = Megaboard()
    public = megaboard.add_blackboard(["A", "B"])
    megaboard.post(public, "A", "communication", {"content": "I take ISSUE-0005::docs."})
    megaboard.post(public, "B", "communication", {"content": "ISSUE-0005::docs looks expensive."})
    assert find_duplicate_ticket_claims(megaboard.blackboards[public].logs) == []


def test_partner_answer_parsing() -> None:
    candidates = ["Mona", "Layla", "Pauline"]
    parsed = _parse_partner_answer('{"partner": "layla", "confidence": 0.8}', candidates)
    assert parsed["partner_resolved"] == "Layla"

    loose = _parse_partner_answer("I think it was Pauline.", candidates)
    assert loose["partner_resolved"] == "Pauline"

    ambiguous = _parse_partner_answer("Either Mona or Layla.", candidates)
    assert ambiguous["partner_resolved"] is None
