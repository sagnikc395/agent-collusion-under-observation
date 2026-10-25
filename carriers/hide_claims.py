"""RQ4 carrier ablation: hide which tickets earlier agents already claimed.

Terrarium records every committed `assign_task` on each blackboard the acting
agent belongs to (`kind == "action_executed"`), and the JIRA prompt also renders
a CURRENT ASSIGNMENTS block from `joint_assignment`. Both make the claims of
earlier agents visible to later ones, so this ablation removes both.
"""

from __future__ import annotations

from typing import Any, Dict, List

ACTION_EVENT_KIND = "action_executed"


def strip_action_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop committed-action traces from a blackboard event list."""
    return [
        event
        for event in events
        if not (isinstance(event, dict) and str(event.get("kind") or "") == ACTION_EVENT_KIND)
    ]


def strip_joint_assignment(agent_context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the agent context with other agents' claims removed.

    The agent keeps its own claim (it made it), which matters because an agent that
    already acted must not be told it is free to act again.
    """
    context = dict(agent_context)
    assignment = context.get("joint_assignment")
    if isinstance(assignment, dict):
        me = str(context.get("agent_name") or "")
        context["joint_assignment"] = {
            agent: task for agent, task in assignment.items() if str(agent) == me
        }
    return context
