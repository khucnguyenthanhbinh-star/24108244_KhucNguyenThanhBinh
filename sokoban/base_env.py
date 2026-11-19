"""Giao diện chung cho các môi trường và các thuật toán tabular."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Discrete:
    """Tập hành động rời rạc gồm n hành động, đánh số từ 0 đến n-1."""

    def __init__(self, n: int, seed: Optional[int] = None):
        if n <= 0:
            raise ValueError("Số hành động phải lớn hơn 0.")
        self.n = int(n)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self._rng.integers(self.n))

    def contains(self, x: int) -> bool:
        return 0 <= int(x) < self.n

    def __contains__(self, x: object) -> bool:
        return isinstance(x, (int, np.integer)) and self.contains(int(x))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class MDPEnv:
    """
    Môi trường học tăng cường rời rạc.

    Mỗi môi trường cài đặt 6 hàm: reset, step, transition, state_ids,
    is_terminal, to_char_grid.
    """

    n_actions: int = 4

    def reset(self) -> int:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError

    def transition(self, state_id: int, action: int) -> Tuple[int, float, bool]:
        raise NotImplementedError

    def state_ids(self) -> List[int]:
        raise NotImplementedError

    def is_terminal(self, state_id: int) -> bool:
        raise NotImplementedError

    def agent_cell(self, state_id: int) -> Tuple[int, int]:
        """Ô đang đứng của tác tử, dùng để vẽ bản đồ nhiệt đồ giá trị."""
        raise NotImplementedError

    def to_char_grid(self) -> List[List[str]]:
        raise NotImplementedError

    def n_states_hint(self) -> int:
        return len(self.state_ids())

    def describe(self) -> str:
        return f"{type(self).__name__}: {self.n_actions} hành động"

    def close(self) -> None:
        return None
