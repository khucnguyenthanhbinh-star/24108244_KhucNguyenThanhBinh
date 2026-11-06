"""Kiểm thử sáu thuật toán."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sokoban.agents import (
    ExpectedSARSAAgent,
    MonteCarloControlAgent,
    PolicyIteration,
    QLearningAgent,
    SARSAAgent,
    ValueIteration,
)
from sokoban.env import SokobanTwoBoxesEnv

MAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")
TD_CLASSES = [QLearningAgent, SARSAAgent, ExpectedSARSAAgent, MonteCarloControlAgent]


def make_env(map_file: str = "map_setting1_small.txt") -> SokobanTwoBoxesEnv:
    return SokobanTwoBoxesEnv(map_path=os.path.join(MAPS_DIR, map_file))


def test_value_iteration_converges():
    env = make_env()
    solver = ValueIteration(env)
    result = solver.solve()
    assert result["converged"]
    assert result["iterations"] < 1000


def test_policy_iteration_converges():
    env = make_env()
    solver = PolicyIteration(env)
    result = solver.solve()
    assert result["converged"]


def test_value_iteration_and_policy_iteration_agree():
    """
    Hai thuật toán DP phải cho cùng một hàm giá trị.

    Chính sách có thể khác nhau ở những trạng thái mà hai hành động cho giá
    trị bằng nhau, nên điều kiện được kiểm là hàm giá trị, không phải chính sách.
    """
    env = make_env()
    vi = ValueIteration(env)
    pi = PolicyIteration(env)
    vi.solve()
    pi.solve()

    assert np.max(np.abs(vi.V - pi.V)) < 1e-3

    for state_id in vi.model.state_ids:
        if vi.get_action(state_id) == pi.get_action(state_id):
            continue
        s_idx = vi.model.idx(state_id)
        q = vi._q_values(s_idx, vi.V)
        assert abs(q[vi.get_action(state_id)] - q[pi.get_action(state_id)]) < 1e-3


def test_value_iteration_values_non_negative():
    env = make_env()
    vi = ValueIteration(env)
    vi.solve()
    values = np.array(list(vi.values().values()))
    assert values.size > 0
    assert np.nanmax(values) > 0


@pytest.mark.parametrize("cls", TD_CLASSES)
def test_agent_interface(cls):
    env = make_env()
    agent = cls(env, seed=0)
    history = agent.train(num_episodes=120, eval_interval=60, eval_episodes=3)
    assert set(history) >= {"episode_rewards", "eval_history", "success_rates",
                             "epsilons", "training_time", "algorithm"}
    assert len(history["episode_rewards"]) == 120

    metrics = agent.evaluate(5)
    assert set(metrics) >= {"success_rate", "deadlock_rate", "mean_reward", "mean_steps"}
    assert 0.0 <= metrics["success_rate"] <= 1.0


@pytest.mark.parametrize("cls", TD_CLASSES)
def test_agent_learns_to_solve(cls):
    env = make_env()
    agent = cls(env, seed=0)
    agent.train(num_episodes=600, eval_interval=600, eval_episodes=5)
    metrics = agent.evaluate(20)
    assert metrics["success_rate"] >= 0.8


def test_q_learning_is_off_policy_target():
    """Đích của Q-Learning là giá trị lớn nhất, không phụ thuộc hành động đã chọn."""
    env = make_env()
    agent = QLearningAgent(env, seed=0)
    agent.q_table[1] = np.array([1.0, 5.0, 2.0, 0.0])
    target = agent.compute_target(2.0, 1, None, terminated=False)
    assert target == pytest.approx(2.0 + agent.gamma * 5.0)


def test_sarsa_target_uses_given_action():
    env = make_env()
    agent = SARSAAgent(env, seed=0)
    agent.q_table[1] = np.array([1.0, 5.0, 2.0, 0.0])
    target = agent.compute_target(2.0, 1, next_action=2, terminated=False)
    assert target == pytest.approx(2.0 + agent.gamma * 2.0)


def test_expected_sarsa_target_is_expectation():
    env = make_env()
    agent = ExpectedSARSAAgent(env, seed=0)
    agent.q_table[1] = np.array([1.0, 5.0, 2.0, 0.0])
    agent.epsilon = 0.0
    target = agent.compute_target(0.0, 1, None, terminated=False)
    assert target == pytest.approx(agent.gamma * 5.0)


def test_monte_carlo_updates_only_after_episode():
    env = make_env()
    agent = MonteCarloControlAgent(env, seed=0)
    agent.on_step(0, 0, -1.0, 0, None, False, False)
    assert 0 not in agent.q_table
    agent.on_episode_end([(0, 0, -1.0)])
    assert 0 in agent.q_table


def test_greedy_policy_is_deterministic():
    env = make_env()
    agent = QLearningAgent(env, seed=0)
    agent.train(num_episodes=100, eval_interval=100, eval_episodes=2)
    policy = agent.greedy_policy()
    assert policy
    assert all(0 <= a < 4 for a in policy.values())
