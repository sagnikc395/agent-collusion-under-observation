"""A scripted LLM client, so the runner can be exercised without a GPU.

It posts one public message per planning turn and claims a ticket at execution,
which is enough to drive blackboards, tool events, metrics and the carrier hooks.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Tuple

from carriers.paths import ensure_colosseum_on_path

ensure_colosseum_on_path()

from terrarium.llm.clients.abstract_client import AbstractClient  # noqa: E402

_TICKET_RE = re.compile(r"ISSUE-\d+::[a-z]+")
_BB_RE = re.compile(r"Blackboard (\d+):")


def _chat_response(content: str, tool_calls: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "fake",
    }


def _tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeClient(AbstractClient):
    """Deterministic agent: talk during planning, claim a ticket at execution."""

    def __init__(self, agent_name: str, seed: int = 0):
        self.agent_name = agent_name
        self.seed = int(seed)
        self.turns = 0
        self.last_prompt = ""

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def get_usage(response: Dict[str, Any], current_usage: Dict[str, int]) -> Dict[str, int]:
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            current_usage[key] = current_usage.get(key, 0) + usage.get(key, 0)
        return current_usage

    def _pick_ticket(self, prompt: str) -> str:
        tickets = sorted(set(_TICKET_RE.findall(prompt)))
        if not tickets:
            return "skip"
        index = (abs(hash((self.agent_name, self.seed))) + self.turns) % len(tickets)
        return tickets[index]

    def _pick_blackboard(self, prompt: str) -> int:
        ids = [int(x) for x in _BB_RE.findall(prompt)]
        return max(ids) if ids else 0

    def generate_response(
        self, input: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        prompt = "\n".join(str(m.get("content") or "") for m in input if isinstance(m, dict))
        self.last_prompt = prompt
        self.turns += 1

        tool_names = {
            (t.get("function") or {}).get("name") for t in (params.get("tools") or [])
        }
        if self.agent_name.startswith("aux_"):
            # Paraphraser / verifier / survey calls: no tools, plain text back.
            if "flags" in prompt:
                return _chat_response('{"flags": []}'), '{"flags": []}'
            if "partner" in prompt:
                text = '{"partner": "unknown", "confidence": 0.1, "reason": "no signal"}'
                return _chat_response(text), text
            return _chat_response(f"[paraphrased] {prompt[-120:]}"), "[paraphrased]"

        if "assign_task" in tool_names:
            ticket = self._pick_ticket(prompt)
            call = _tool_call("assign_task", {"task_id": ticket})
            return _chat_response(f"Claiming {ticket}.", [call]), f"Claiming {ticket}."

        ticket = self._pick_ticket(prompt)
        call = _tool_call(
            "post_message",
            {
                "blackboard_id": self._pick_blackboard(prompt),
                "message": f"I plan to take {ticket}; my availability is tight.",
            },
        )
        text = f"Posting a plan about {ticket}."
        return _chat_response(text, [call]), text

    async def process_tool_calls(
        self,
        response: Dict[str, Any],
        context: List[Dict[str, Any]],
        execute_tool_callback: Any,
    ) -> Tuple[int, List[Dict[str, Any]], List[str]]:
        message = (response.get("choices") or [{}])[0].get("message") or {}
        context.append(message)
        executed = 0
        names: List[str] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name", "unknown")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            names.append(f"{name} -- {json.dumps(arguments)}")
            result = await execute_tool_callback(name, arguments)
            context.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(result),
                }
            )
            executed += 1
        return executed, context, names


def fake_client_factory(seed: int = 0):
    def factory(llm_cfg: Dict[str, Any], agent_name: str) -> FakeClient:
        return FakeClient(agent_name, seed=seed)

    return factory
