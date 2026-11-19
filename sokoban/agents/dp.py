"""Các thuật toán quy hoạch động: Value Iteration và Policy Iteration."""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from sokoban.base_env import MDPEnv
from sokoban.mdp import TransitionModel


class DPSolver:
    """Lớp cơ sở cho hai thuật toán quy hoạch động."""

    name = "DP"

    def __init__(self, env: MDPEnv, gamma: float = 0.95, theta: float = 1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.model = TransitionModel(env)
        self.n_states = self.model.n_states
        self.n_actions = env.n_actions
        self.V = np.zeros(self.n_states, dtype=np.float64)
        self.policy = np.zeros(self.n_states, dtype=np.int32)

    def _q_values(self, s_idx: int, V: np.ndarray) -> np.ndarray:
        """Q(s,a) cho mọi hành động tại trạng thái s, dùng hàm giá trị V."""
        q = np.zeros(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            nxt, reward, terminated = self.model.transitions[(s_idx, a)]
            q[a] = reward if terminated else reward + self.gamma * V[nxt]
        return q

    def solve(self) -> Dict[str, Any]:
        raise NotImplementedError

    def train(self, num_episodes: int = 0, eval_interval: int = 0,
              eval_episodes: int = 20) -> Dict[str, Any]:
        """Cùng giao diện với các tác tử TD: giải bài toán rồi đánh giá chính sách."""
        start_time = time.time()
        result = self.solve()
        metrics = self.evaluate(eval_episodes)
        result.update(metrics)
        result.update({
            "algorithm": self.name,
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilons": [],
            "success_rates": [metrics["success_rate"]],
            "deadlock_rates": [metrics["deadlock_rate"]],
            "eval_history": [(result.get("iterations", 0), metrics)],
            "q_table_size": self.n_states,
            "values_history": self._values_history(),
            "training_time": time.time() - start_time,
        })
        return result

    def _values_history(self) -> List[float]:
        return list(getattr(self, "values_history", []))

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """Đánh giá chính sách tối ưu bằng cách chơi thử theo chính sách đó."""
        success = 0
        deadlock = 0
        timeout = 0
        rewards: List[float] = []
        discounted: List[float] = []
        lengths: List[int] = []
        activations: List[int] = []
        uses: List[int] = []

        for _ in range(num_episodes):
            state_id = self.env.reset()
            total = 0.0
            disc = 0.0
            steps = 0
            terminated = truncated = False
            info: Dict[str, Any] = {}

            while not (terminated or truncated):
                action = self.get_action(state_id)
                state_id, reward, terminated, truncated, info = self.env.step(action)
                disc += (self.gamma ** steps) * reward
                total += reward
                steps += 1

            outcome = info.get("outcome", "timeout")
            if outcome == "solved":
                success += 1
            elif outcome == "deadlock":
                deadlock += 1
            else:
                timeout += 1
            rewards.append(total)
            discounted.append(disc)
            lengths.append(steps)
            activations.append(info.get("rescue_activations", 0))
            uses.append(info.get("rescue_uses", 0))

        return {
            "success_rate": success / num_episodes,
            "deadlock_rate": deadlock / num_episodes,
            "timeout_rate": timeout / num_episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_discounted_reward": float(np.mean(discounted)),
            "std_discounted_reward": float(np.std(discounted)),
            "mean_steps": float(np.mean(lengths)),
            "std_steps": float(np.std(lengths)),
            "mean_rescue_activations": float(np.mean(activations)),
            "mean_rescue_uses": float(np.mean(uses)),
        }

    def get_action(self, state_id: int) -> int:
        """Hành động tối ưu tại một trạng thái."""
        return int(self.policy[self.model.idx(state_id)])

    def select_action(self, state_id: int, explore: bool = True) -> int:
        """Cùng giao diện với tác tử TD: chính sách tối ưu không cần thăm dò."""
        return self.get_action(state_id)

    def greedy_policy(self) -> Dict[int, int]:
        return {s: int(self.policy[i]) for i, s in enumerate(self.model.state_ids)}

    def values(self) -> Dict[int, float]:
        return {s: float(self.V[i]) for i, s in enumerate(self.model.state_ids)}

    def save(self, file_path: str) -> None:
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        np.savez_compressed(file_path,
                            values=self.V,
                            policy=self.policy,
                            state_ids=np.array(self.model.state_ids, dtype=np.int64))

    def load(self, file_path: str) -> None:
        data = np.load(file_path)
        self.V = data["values"]
        self.policy = data["policy"]


class ValueIteration(DPSolver):
    """
    Value Iteration (Sutton & Barto, chương 4).

        V(s) <- max_a [ r(s,a) + gamma * V(s') ]
    """

    name = "Value Iteration"

    def solve(self, max_iterations: int = 1000) -> Dict[str, Any]:
        start_time = time.time()
        deltas: List[float] = []
        self.values_history: List[float] = []

        for _ in range(max_iterations):
            delta = 0.0
            for s_idx in range(self.n_states):
                if self.model.is_terminal(s_idx):
                    continue
                q = self._q_values(s_idx, self.V)
                best = float(np.max(q))
                delta = max(delta, abs(self.V[s_idx] - best))
                self.V[s_idx] = best
            deltas.append(delta)
            self.values_history.append(float(np.max(self.V)))
            if delta < self.theta:
                break

        for s_idx in range(self.n_states):
            if self.model.is_terminal(s_idx):
                continue
            self.policy[s_idx] = int(np.argmax(self._q_values(s_idx, self.V)))

        return {
            "algorithm": self.name,
            "iterations": len(deltas),
            "deltas": deltas,
            "training_time": time.time() - start_time,
            "converged": deltas[-1] < self.theta,
            "n_states": self.n_states,
            "n_terminal_states": self.model.n_terminal(),
        }


class PolicyIteration(DPSolver):
    """
    Policy Iteration (Sutton & Barto, chương 4).

        V^pi(s) = r(s, pi(s)) + gamma * V^pi(s')
        pi(s)  <- argmax_a [ r(s,a) + gamma * V^pi(s) ]
    """

    name = "Policy Iteration"

    def policy_evaluation(self, max_iter: int = 1000) -> int:
        """Đánh giá chính sách hiện tại cho tới khi hội tụ."""
        for iteration in range(max_iter):
            delta = 0.0
            for s_idx in range(self.n_states):
                if self.model.is_terminal(s_idx):
                    continue
                a = self.policy[s_idx]
                nxt, reward, terminated = self.model.transitions[(s_idx, a)]
                new_v = reward if terminated else reward + self.gamma * self.V[nxt]
                delta = max(delta, abs(self.V[s_idx] - new_v))
                self.V[s_idx] = new_v
            if delta < self.theta:
                return iteration + 1
        return max_iter

    def policy_improvement(self) -> bool:
        """
        Cải thiện chính sách theo hàm giá trị. Trả về True nếu không còn đổi.

        Chỉ đổi hành động khi hành động mới tốt hơn hẳn, tránh hiện tượng
        chattering do hai hành động có giá trị gần bằng nhau.
        """
        stable = True
        for s_idx in range(self.n_states):
            if self.model.is_terminal(s_idx):
                continue
            q = self._q_values(s_idx, self.V)
            best_action = int(np.argmax(q))
            if q[best_action] > q[self.policy[s_idx]] + 1e-9:
                self.policy[s_idx] = best_action
                stable = False
        return stable

    def solve(self, max_iterations: int = 100) -> Dict[str, Any]:
        start_time = time.time()
        eval_steps: List[int] = []
        self.values_history: List[float] = []
        converged = False

        for i in range(max_iterations):
            eval_steps.append(self.policy_evaluation())
            self.values_history.append(float(np.max(self.V)))
            if self.policy_improvement():
                converged = True
                break

        return {
            "algorithm": self.name,
            "iterations": len(eval_steps),
            "eval_steps_history": eval_steps,
            "training_time": time.time() - start_time,
            "converged": converged,
            "n_states": self.n_states,
            "n_terminal_states": self.model.n_terminal(),
        }
