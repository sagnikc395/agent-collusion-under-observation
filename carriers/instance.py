"""Dump the DCOP instance so every analysis runs on CPU from JSON.

Colosseum's `metrics.json` keeps only realised outcomes. Coalitional regret and
the lambda_emp fit need counterfactual scores, so the cost table, the task table
and the scoring weights are written out once per run. `analysis/env_model.py`
re-implements the same scoring rule and is checked against the environment in
`tests/test_env_model_matches_terrarium.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def dump_instance(
    *,
    env: Any,
    colluders: Sequence[str],
    agent_turn_order: Sequence[str],
    assignment: Dict[str, Any],
) -> Dict[str, Any]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for task_id, task in (getattr(env, "tasks", {}) or {}).items():
        tasks[str(task_id)] = {
            "id": str(task_id),
            "issue_id": task.get("issue_id"),
            "priority": task.get("priority"),
            "effort": task.get("effort"),
            "work_type": task.get("work_type"),
            "tags": list(task.get("tags") or []),
        }

    costs: Dict[str, Dict[str, float]] = {}
    for agent, table in (getattr(env, "costs", {}) or {}).items():
        costs[str(agent)] = {str(k): float(v) for k, v in (table or {}).items()}

    private: Dict[str, Dict[str, Any]] = {}
    for agent, state in (getattr(env, "agent_private", {}) or {}).items():
        private[str(agent)] = {
            "availability": state.get("availability"),
            "skills": {str(k): float(v) for k, v in (state.get("skills") or {}).items()},
        }

    agent_names: List[str] = [str(a) for a in (getattr(env, "agent_names", []) or [])]

    return {
        "environment": env.__class__.__name__,
        "agent_names": agent_names,
        "agent_turn_order": [str(a) for a in agent_turn_order],
        "colluders": [str(a) for a in colluders],
        "tasks": tasks,
        "costs": costs,
        "agent_private": private,
        "assignment": {str(k): v for k, v in (assignment or {}).items()},
        "weights": {
            "tasks_done_bonus": float(getattr(env, "tasks_done_bonus", 20.0)),
            "priority_bonus": float(getattr(env, "priority_bonus", 20.0)),
            "violation_penalty": float(getattr(env, "violation_penalty", 20.0)),
            "priority_weights": dict(getattr(env, "PRIORITY_WEIGHTS", {})),
        },
        "max_joint_reward": float(getattr(env, "max_joint_reward", 0.0) or 0.0),
        "optimal_k": int(getattr(env, "optimal_k", 0) or 0),
    }
