"""A minimal chat wrapper for the auxiliary models (paraphraser, verifier, survey).

These are plain completions with no tools, so they bypass Terrarium's agent loop
and talk to whatever client the runner was given.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SimpleChat:
    """Wrap a Terrarium LLM client as `messages -> text`, counting usage."""

    client: Any
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    calls: int = 0
    failures: int = 0
    usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )

    def __call__(self, messages: List[Dict[str, str]], **overrides: Any) -> Optional[str]:
        params: Dict[str, Any] = {
            "model": self.model,
            "temperature": float(overrides.pop("temperature", self.temperature)),
            "max_tokens": int(overrides.pop("max_tokens", self.max_tokens)),
        }
        params["max_completion_tokens"] = params["max_tokens"]
        params["max_output_tokens"] = params["max_tokens"]
        params.update(overrides)
        self.calls += 1
        try:
            response, text = self.client.generate_response(input=messages, params=params)
        except Exception:
            self.failures += 1
            logger.exception("Auxiliary LLM call failed")
            return None
        get_usage = getattr(self.client, "get_usage", None)
        if callable(get_usage):
            try:
                self.usage = get_usage(response, self.usage)
            except Exception:
                pass
        return text

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": int(self.calls),
            "failures": int(self.failures),
            "usage": dict(self.usage),
        }
