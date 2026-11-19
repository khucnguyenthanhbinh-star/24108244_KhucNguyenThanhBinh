"""Mô hình MDP dùng chung cho các thuật toán quy hoạch động."""

from typing import Dict, List, Tuple

from sokoban.base_env import MDPEnv


class TransitionModel:
    """
    Đồ thị chuyển trạng thái của một MDP rời rạc.

    transitions[(s_idx, a)] = (next_s_idx, reward, terminated)

    Trạng thái kết thúc episode có giá trị bằng 0 và không có hành động đi tiếp.
    """

    def __init__(self, mdp: MDPEnv):
        self.mdp = mdp
        self.state_ids: List[int] = mdp.state_ids()
        self.state_to_idx: Dict[int, int] = {s: i for i, s in enumerate(self.state_ids)}
        self.n_states = len(self.state_ids)
        self.n_actions = mdp.n_actions
        self.transitions: Dict[Tuple[int, int], Tuple[int, float, bool]] = {}
        self.terminal_states: set = set()
        self._build()

    def _build(self) -> None:
        for s_idx, s_id in enumerate(self.state_ids):
            if self.mdp.is_terminal(s_id):
                self.terminal_states.add(s_idx)
                for a in range(self.n_actions):
                    self.transitions[(s_idx, a)] = (s_idx, 0.0, True)
                continue
            for a in range(self.n_actions):
                nxt_id, reward, terminated = self.mdp.transition(s_id, a)
                nxt_idx = self.state_to_idx.get(nxt_id, s_idx)
                self.transitions[(s_idx, a)] = (nxt_idx, reward, terminated)

    def idx(self, state_id: int) -> int:
        return self.state_to_idx.get(state_id, 0)

    def next_idx(self, s_idx: int, action: int) -> int:
        return self.transitions[(s_idx, action)][0]

    def reward(self, s_idx: int, action: int) -> float:
        return self.transitions[(s_idx, action)][1]

    def is_terminal(self, s_idx: int) -> bool:
        """Trạng thái kết thúc episode (V = 0, không có hành động đi tiếp)."""
        return s_idx in self.terminal_states

    def n_terminal(self) -> int:
        return sum(1 for s in range(self.n_states) if self.is_terminal(s))
