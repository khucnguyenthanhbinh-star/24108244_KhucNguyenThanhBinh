"""Lớp cơ sở cho các thuật toán điều khiển theo giá trị hành động."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sokoban.base_env import MDPEnv


class TabularAgent:
    """
    Tác tử học theo bảng Q.

    Tham số:
        alpha, gamma        hệ số học và hệ số khuyến đại
        epsilon_*           tham số của chính sách epsilon-greedy
    """

    name = "TabularAgent"

    def __init__(
        self,
        env: MDPEnv,
        alpha: float = 0.15,
        alpha_decay: float = 0.9999,
        alpha_min: float = 0.01,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = env.n_actions

        self.rng = np.random.RandomState(seed)
        self.q_table: Dict[int, np.ndarray] = {}

    def get_q(self, state_id: int) -> np.ndarray:
        if state_id not in self.q_table:
            self.q_table[state_id] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[state_id]

    def policy_probs(self, state_id: int) -> np.ndarray:
        """Xác suất chọn mỗi hành động theo chính sách epsilon-greedy."""
        probs = np.full(self.n_actions, self.epsilon / self.n_actions, dtype=np.float64)
        best = int(np.argmax(self.get_q(state_id)))
        probs[best] += 1.0 - self.epsilon
        return probs

    def select_action(self, state_id: int, explore: bool = True) -> int:
        if explore and self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.n_actions))
        q_vals = self.get_q(state_id)
        best = np.where(q_vals == np.max(q_vals))[0]
        return int(self.rng.choice(best))

    def compute_target(self, reward: float, next_state: int,
                       next_action: Optional[int], terminated: bool) -> float:
        raise NotImplementedError

    def update(self, state_id: int, action: int, target: float) -> None:
        q_vals = self.get_q(state_id)
        q_vals[action] += self.alpha * (target - q_vals[action])

    def on_episode_start(self) -> None:
        return None

    def on_step(self, state_id: int, action: int, reward: float, next_state: int,
                next_action: Optional[int], terminated: bool, truncated: bool) -> None:
        target = self.compute_target(reward, next_state, next_action, terminated)
        self.update(state_id, action, target)

    def on_episode_end(self, episode: List[Tuple[int, int, float]]) -> None:
        return None

    def train(self, num_episodes: int = 3000, eval_interval: int = 500,
              eval_episodes: int = 10) -> Dict[str, Any]:
        start_time = time.time()
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        success_rates: List[float] = []
        deadlock_rates: List[float] = []
        eval_history: List[Tuple[int, Dict[str, float]]] = []
        epsilons: List[float] = []

        for ep in range(num_episodes):
            state_id = self.env.reset()
            self.on_episode_start()
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            episode: List[Tuple[int, int, float]] = []

            while not (terminated or truncated):
                action = self.select_action(state_id, explore=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = None
                if not (terminated or truncated):
                    next_action = self.select_action(next_state, explore=True)

                self.on_step(state_id, action, reward, next_state,
                             next_action, terminated, truncated)
                episode.append((state_id, action, reward))

                state_id = next_state
                total_reward += reward
                steps += 1

            self.on_episode_end(episode)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            epsilons.append(self.epsilon)

            if (ep + 1) % eval_interval == 0:
                metrics = self.evaluate(eval_episodes)
                eval_history.append((ep + 1, metrics))
                success_rates.append(metrics["success_rate"])
                deadlock_rates.append(metrics["deadlock_rate"])

        return {
            "algorithm": self.name,
            "num_episodes": num_episodes,
            "training_time": time.time() - start_time,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "epsilons": epsilons,
            "success_rates": success_rates,
            "deadlock_rates": deadlock_rates,
            "eval_history": eval_history,
            "q_table_size": len(self.q_table),
        }

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """
        Đánh giá chính sách tham lam trên một số episode cố định.

        Vì môi trường tất định và chính sách đánh giá là tham lam, các episode
        cho cùng kết quả, nên tỉ lệ thành công ở đây thực chất là tỉ lệ các
        lần chạy cho kết quả đúng.
        """
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
                action = self.select_action(state_id, explore=False)
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

    def greedy_policy(self) -> Dict[int, int]:
        """Chính sách tham lam đã học được."""
        return {s: int(np.argmax(self.get_q(s))) for s in self.q_table}

    def save(self, file_path: str) -> None:
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        keys = np.array(list(self.q_table.keys()), dtype=np.int64)
        values = np.array(list(self.q_table.values()), dtype=np.float64)
        np.savez_compressed(file_path, keys=keys, values=values)

    def load(self, file_path: str) -> None:
        data = np.load(file_path)
        self.q_table = {int(k): v for k, v in zip(data["keys"], data["values"])}
