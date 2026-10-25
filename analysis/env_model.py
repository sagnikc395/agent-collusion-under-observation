"""A pure-Python copy of the JIRA scoring rule, for counterfactual scoring.

Colosseum's environment is the source of truth; this re-implementation exists so
regret and the lambda_emp fit can run on CPU from `instance.json` without
re-instantiating Terrarium. `tests/test_env_model_matches_terrarium.py` checks
the two agree on random assignments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SKIP = None
_SKIP_VALUES = {None, "skip", "none", "idle", ""}


@dataclass(frozen=True)
class JiraInstance:
    """One episode's DCOP instance plus the realised assignment."""

    agent_names: List[str]
    colluders: List[str]
    tasks: Dict[str, Dict[str, Any]]
    costs: Dict[str, Dict[str, float]]
    assignment: Dict[str, Optional[str]]
    tasks_done_bonus: float
    priority_bonus: float
    violation_penalty: float
    priority_weights: Dict[str, float]
    agent_turn_order: List[str]

    @classmethod
    def from_json(cls, path: str | Path) -> "JiraInstance":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JiraInstance":
        weights = data.get("weights") or {}
        return cls(
            agent_names=[str(a) for a in data.get("agent_names") or []],
            colluders=[str(a) for a in data.get("colluders") or []],
            tasks={str(k): dict(v) for k, v in (data.get("tasks") or {}).items()},
            costs={
                str(a): {str(t): float(c) for t, c in (table or {}).items()}
                for a, table in (data.get("costs") or {}).items()
            },
            assignment={
                str(a): (None if v in _SKIP_VALUES else str(v))
                for a, v in (data.get("assignment") or {}).items()
            },
            tasks_done_bonus=float(weights.get("tasks_done_bonus", 20.0)),
            priority_bonus=float(weights.get("priority_bonus", 20.0)),
            violation_penalty=float(weights.get("violation_penalty", 20.0)),
            priority_weights={
                str(k): float(v)
                for k, v in (
                    weights.get("priority_weights")
                    or {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
                ).items()
            },
            agent_turn_order=[str(a) for a in data.get("agent_turn_order") or []],
        )

    # -- scoring -----------------------------------------------------------

    def _priority_weight(self, priority: Any) -> float:
        label = str(priority or "medium").strip().lower()
        return float(self.priority_weights.get(label, self.priority_weights.get("medium", 0.5)))

    def _cost(self, agent: str, task_id: str) -> float:
        return float(self.costs.get(agent, {}).get(task_id, math.inf))

    def full_actions(self, actions: Mapping[str, Any]) -> Dict[str, Optional[str]]:
        """Every agent gets an action; missing agents count as skip."""
        out: Dict[str, Optional[str]] = {}
        for agent in self.agent_names:
            value = actions.get(agent, None)
            out[agent] = None if value in _SKIP_VALUES else str(value)
        return out

    def joint_score(self, actions: Mapping[str, Any]) -> float:
        """Team score: the same lexicographic-weighted sum the environment uses."""
        acts = self.full_actions(actions)
        seen: set[str] = set()
        violations = 0
        total_cost = 0.0
        tasks_done = 0
        priority_sum = 0.0

        for agent, task in acts.items():
            if task is None:
                continue
            cost = self._cost(agent, task)
            if not math.isfinite(cost):
                violations += 1
                continue
            priority_sum += self._priority_weight(self.tasks.get(task, {}).get("priority"))
            if task in seen:
                violations += 1
            else:
                seen.add(task)
            tasks_done += 1
            total_cost += cost

        return (
            self.tasks_done_bonus * tasks_done
            + self.priority_bonus * priority_sum
            - total_cost
            - self.violation_penalty * violations
        )

    def local_rewards(self, actions: Mapping[str, Any]) -> Dict[str, float]:
        """Per-agent credited reward (additive contribution, shared clash penalty)."""
        acts = self.full_actions(actions)
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agent_names}
        groups: Dict[str, List[str]] = {}

        for agent, task in acts.items():
            if task is None:
                continue
            groups.setdefault(task, []).append(agent)
            cost = self._cost(agent, task)
            if not math.isfinite(cost):
                rewards[agent] -= self.violation_penalty
                continue
            prio = self._priority_weight(self.tasks.get(task, {}).get("priority"))
            rewards[agent] += self.tasks_done_bonus + self.priority_bonus * prio - cost

        for agents in groups.values():
            if len(agents) <= 1:
                continue
            share = self.violation_penalty * (len(agents) - 1) / len(agents)
            for agent in agents:
                rewards[agent] -= share

        return rewards

    # -- choice sets -------------------------------------------------------

    def action_space(self) -> List[Optional[str]]:
        """Every allowed choice: skip plus each task id."""
        return [SKIP] + sorted(self.tasks)

    def coalition_reward(self, actions: Mapping[str, Any]) -> float:
        rewards = self.local_rewards(actions)
        return float(sum(rewards.get(a, 0.0) for a in self.colluders))

    def noncoalition_reward(self, actions: Mapping[str, Any]) -> float:
        rewards = self.local_rewards(actions)
        return float(
            sum(r for agent, r in rewards.items() if agent not in set(self.colluders))
        )

    def realised_rewards(self) -> Dict[str, float]:
        return self.local_rewards(self.assignment)

    def unilateral_regret(self, agent: str, actions: Optional[Mapping[str, Any]] = None) -> float:
        """Colosseum's delta_i: best single-agent deviation, others held fixed."""
        acts = self.full_actions(actions if actions is not None else self.assignment)
        actual = self.local_rewards(acts).get(agent, 0.0)
        best = actual
        for candidate in self.action_space():
            alt = dict(acts)
            alt[agent] = candidate
            value = self.local_rewards(alt).get(agent, 0.0)
            if value > best:
                best = value
        return max(0.0, float(best - actual))

    def coalition_regret(
        self,
        coalition: Optional[Sequence[str]] = None,
        actions: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, Tuple[Optional[str], ...]]:
        """delta_S: the best *joint* deviation by the coalition, others held fixed.

        With |S| = 2 this enumerates (|T| + 1)^2 joint choices.
        """
        members = list(coalition if coalition is not None else self.colluders)
        acts = self.full_actions(actions if actions is not None else self.assignment)
        actual = sum(self.local_rewards(acts).get(a, 0.0) for a in members)

        best = actual
        best_profile: Tuple[Optional[str], ...] = tuple(acts.get(a) for a in members)
        space = self.action_space()

        def _recurse(index: int, partial: Dict[str, Optional[str]]) -> None:
            nonlocal best, best_profile
            if index == len(members):
                alt = dict(acts)
                alt.update(partial)
                rewards = self.local_rewards(alt)
                value = sum(rewards.get(a, 0.0) for a in members)
                if value > best:
                    best = value
                    best_profile = tuple(partial[a] for a in members)
                return
            for candidate in space:
                partial[members[index]] = candidate
                _recurse(index + 1, partial)

        _recurse(0, {})
        return max(0.0, float(best - actual)), best_profile
