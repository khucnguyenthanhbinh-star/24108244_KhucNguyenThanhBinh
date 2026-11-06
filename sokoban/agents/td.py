"""Các thuật toán điều khiển không cần mô hình môi trường."""

from typing import List, Optional, Tuple

import numpy as np

from sokoban.agents.base import TabularAgent


class QLearningAgent(TabularAgent):
    """
    Q-Learning (Sutton & Barto, chương 6).

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    """

    name = "Q-Learning"

    def compute_target(self, reward: float, next_state: int,
                       next_action: Optional[int], terminated: bool) -> float:
        if terminated:
            return reward
        return reward + self.gamma * float(np.max(self.get_q(next_state)))


class SARSAAgent(TabularAgent):
    """
    SARSA (Sutton & Barto, chương 6).

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

    a' là hành động thật sự được chọn ở s'.
    """

    name = "SARSA"

    def compute_target(self, reward: float, next_state: int,
                       next_action: Optional[int], terminated: bool) -> float:
        if terminated:
            return reward
        if next_action is None:
            return reward
        return reward + self.gamma * float(self.get_q(next_state)[next_action])


class ExpectedSARSAAgent(TabularAgent):
    """
    Expected SARSA (Sutton & Barto, chương 6).

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a) ]
    """

    name = "Expected SARSA"

    def compute_target(self, reward: float, next_state: int,
                       next_action: Optional[int], terminated: bool) -> float:
        if terminated:
            return reward
        probs = self.policy_probs(next_state)
        q_vals = self.get_q(next_state)
        return reward + self.gamma * float(np.dot(probs, q_vals))


class MonteCarloControlAgent(TabularAgent):
    """
    Monte Carlo Control (Sutton & Barto, chương 5).

    Không bootstrap: cập nhật bằng phần thưởng tích lũy của cả episode.
    First-visit: mỗi cặp (s,a) chỉ được cập nhật một lần trong episode.
    """

    name = "Monte Carlo Control"

    def compute_target(self, reward: float, next_state: int,
                       next_action: Optional[int], terminated: bool) -> float:
        raise NotImplementedError

    def on_step(self, state_id: int, action: int, reward: float, next_state: int,
                next_action: Optional[int], terminated: bool, truncated: bool) -> None:
        return None

    def on_episode_end(self, episode: List[Tuple[int, int, float]]) -> None:
        visited = set()
        g = 0.0
        for state_id, action, reward in reversed(episode):
            g = reward + self.gamma * g
            if (state_id, action) in visited:
                continue
            visited.add((state_id, action))
            self.update(state_id, action, g)
