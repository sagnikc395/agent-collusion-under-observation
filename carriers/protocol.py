"""Communication protocol with the RQ4 carrier ablations wired in.

Subclasses Colosseum's `LocalCommunicationProtocol` and only touches what agents
read off the blackboards: paraphrased public text, hidden action traces. Posting,
tool dispatch and logging are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from carriers.hide_claims import strip_action_events
from carriers.paraphrase import Paraphraser
from carriers.paths import ensure_colosseum_on_path

ensure_colosseum_on_path()

from experiments.common.local_protocol import LocalCommunicationProtocol  # noqa: E402
from terrarium.core.blackboard import format_blackboard_events_for_prompt  # noqa: E402

logger = logging.getLogger(__name__)


class CarrierProtocol(LocalCommunicationProtocol):
    """Applies carrier ablations at read time, per reader."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        paraphraser: Optional[Paraphraser] = None,
        hide_claims: bool = False,
        megaboard: Optional[Any] = None,
    ):
        super().__init__(config=config, megaboard=megaboard)
        self.paraphraser = paraphraser
        self.hide_claims = bool(hide_claims)
        self.secret_blackboard_id: Optional[int] = None

    def _is_secret(self, blackboard_id: int) -> bool:
        if self.secret_blackboard_id is None:
            return False
        return int(blackboard_id) == int(self.secret_blackboard_id)

    def _paraphrase_events(
        self, events: List[Dict[str, Any]], reader: str
    ) -> List[Dict[str, Any]]:
        """Rewrite other agents' public messages; leave the reader's own alone."""
        assert self.paraphraser is not None
        out: List[Dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                out.append(event)
                continue
            author = str(event.get("agent") or "")
            payload = event.get("payload")
            if (
                str(event.get("kind") or "") != "communication"
                or author == reader
                or author in {"", "SYSTEM"}
                or not isinstance(payload, dict)
                or not str(payload.get("content") or "").strip()
            ):
                out.append(event)
                continue
            rewritten = self.paraphraser.paraphrase(
                str(event.get("id") or ""), str(payload.get("content"))
            )
            new_payload = dict(payload)
            new_payload["content"] = rewritten
            out.append({**event, "payload": new_payload})
        return out

    async def _prefetch_blackboard_events(
        self,
        agent_name: str,
        *,
        phase: Optional[str],
        iteration: Optional[int],
    ) -> Dict[str, str]:
        blackboard_ids = self.megaboard.get_agent_blackboards(agent_name)

        def _sort_key(bb_id: str) -> int:
            try:
                return int(bb_id)
            except Exception:
                return 0

        contexts: Dict[str, str] = {}
        for bb_id_str in sorted(blackboard_ids, key=_sort_key):
            try:
                bb_id_int = int(bb_id_str)
            except Exception:
                continue

            events = self.megaboard.get(bb_id_int, agent_name, limit=None)
            events = list(events) if isinstance(events, list) else []

            if self.hide_claims:
                events = strip_action_events(events)
            if self.paraphraser is not None and not self._is_secret(bb_id_int):
                events = self._paraphrase_events(events, str(agent_name))

            contexts[bb_id_str] = format_blackboard_events_for_prompt(events)

        return contexts
