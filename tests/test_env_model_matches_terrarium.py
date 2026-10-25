"""The CPU scoring copy must agree with Terrarium's environment, exactly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.env_model import JiraInstance  # noqa: E402
from carriers.instance import dump_instance  # noqa: E402
from carriers.paths import ensure_colosseum_on_path  # noqa: E402

ensure_colosseum_on_path()

from experiments.common.local_protocol import LocalCommunicationProtocol  # noqa: E402
from terrarium.environments.dcops import JiraTicketEnvironment  # noqa: E402


def _build_env(seed: int, num_agents: int = 6, max_tasks: int = 8):
    cfg = {
        "simulation": {"seed": seed, "max_iterations": 1, "tags": ["test"]},
        "environment": {
            "name": "JiraTicketEnvironment",
            "assignment_filling": False,
            "max_tasks": max_tasks,
        },
        "communication_network": {"topology": "complete", "num_agents": num_agents},
        "llm": {"provider": "vllm", "vllm": {"models": [{"checkpoint": "test"}]}},
    }
    protocol = LocalCommunicationProtocol(config=cfg)
    env = JiraTicketEnvironment(protocol, cfg, tool_logger=type("TL", (), {"log_dir": "."})())
    return env


def test_scores_and_rewards_match_on_random_assignments() -> None:
    for seed in (1, 2, 7):
        env = _build_env(seed)
        instance = JiraInstance.from_dict(
            dump_instance(
                env=env,
                colluders=env.agent_names[:2],
                agent_turn_order=env.agent_names,
                assignment={},
            )
        )
        rng = random.Random(seed)
        choices = [None, "skip"] + list(env.tasks)
        for _ in range(40):
            actions = {agent: rng.choice(choices) for agent in env.agent_names}
            expected_joint = env.joint_reward(actions)
            _, expected_rewards = env._rewards(actions)
            assert instance.joint_score(actions) == expected_joint
            got = instance.local_rewards(actions)
            for agent in env.agent_names:
                assert abs(got[agent] - expected_rewards[agent]) < 1e-9, (
                    seed,
                    agent,
                    actions,
                )


def test_unilateral_regret_matches_colosseum_definition() -> None:
    env = _build_env(3)
    task_ids = list(env.tasks)
    assignment = {
        agent: (task_ids[i % len(task_ids)] if i % 3 else None)
        for i, agent in enumerate(env.agent_names)
    }
    instance = JiraInstance.from_dict(
        dump_instance(
            env=env,
            colluders=env.agent_names[:2],
            agent_turn_order=env.agent_names,
            assignment=assignment,
        )
    )
    for agent in env.agent_names:
        actual = env._rewards(assignment)[1][agent]
        best = actual
        for candidate in ["skip"] + task_ids:
            alt = dict(assignment)
            alt[agent] = candidate
            best = max(best, env._rewards(alt)[1][agent])
        assert abs(instance.unilateral_regret(agent) - max(0.0, best - actual)) < 1e-9


def test_coalition_regret_is_at_least_the_best_single_deviation() -> None:
    env = _build_env(5)
    task_ids = list(env.tasks)
    assignment = {agent: task_ids[i % len(task_ids)] for i, agent in enumerate(env.agent_names)}
    coalition = env.agent_names[:2]
    instance = JiraInstance.from_dict(
        dump_instance(
            env=env,
            colluders=coalition,
            agent_turn_order=env.agent_names,
            assignment=assignment,
        )
    )
    joint_regret, profile = instance.coalition_regret()
    assert len(profile) == 2
    # A joint deviation can always copy a unilateral one, so it is never worse.
    rewards = instance.local_rewards(assignment)
    for member in coalition:
        alt_best = max(
            sum(
                instance.local_rewards({**assignment, member: candidate}).get(m, 0.0)
                for m in coalition
            )
            for candidate in instance.action_space()
        )
        baseline = sum(rewards.get(m, 0.0) for m in coalition)
        assert joint_regret >= max(0.0, alt_best - baseline) - 1e-9
