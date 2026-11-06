"""Môi trường Sokoban hai thùng có ô cứu hộ (đề tài 16)."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from sokoban.base_env import Discrete, MDPEnv


class StateEncoder:
    """Mã hóa trạng thái thành một số nguyên duy nhất."""

    def __init__(self, height: int, width: int, max_rescue_states: int = 3):
        self.height = height
        self.width = width
        self.max_rescue_states = max_rescue_states
        self.grid_size = height * width

    def canonicalize_boxes(self, boxes: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        """Sắp xếp vị trí hai thùng để (hộp1, hộp2) và (hộp2, hộp1) là một trạng thái."""
        return tuple(sorted(boxes))

    def encode_to_id(self, player: Tuple[int, int],
                     boxes: Sequence[Tuple[int, int]],
                     rescue_state: int) -> int:
        b1, b2 = self.canonicalize_boxes(boxes)
        p = player[0] * self.width + player[1]
        i1 = b1[0] * self.width + b1[1]
        i2 = b2[0] * self.width + b2[1]
        return ((p * self.grid_size + i1) * self.grid_size + i2) * self.max_rescue_states + rescue_state

    def decode_from_id(self, state_id: int) -> Tuple[Tuple[int, int],
                                                   Tuple[Tuple[int, int], Tuple[int, int]],
                                                   int]:
        rescue_state = state_id % self.max_rescue_states
        rem = state_id // self.max_rescue_states
        i2 = rem % self.grid_size
        rem //= self.grid_size
        i1 = rem % self.grid_size
        p = rem // self.grid_size
        player = (p // self.width, p % self.width)
        box1 = (i1 // self.width, i1 % self.width)
        box2 = (i2 // self.width, i2 % self.width)
        return player, (box1, box2), rescue_state


class DeadlockDetector:
    """Phát hiện những cách đặt thùng mà không thể giải được."""

    def __init__(self, walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]],
                 height: int, width: int):
        self.walls = walls
        self.goals = goals
        self.height = height
        self.width = width
        self.corner_deadlocks = self._find_corner_deadlocks()
        self._cache: Dict[Tuple[Tuple[int, int], ...], bool] = {}

    def _is_wall(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return (r, c) in self.walls

    def _find_corner_deadlocks(self) -> Set[Tuple[int, int]]:
        deadlocks: Set[Tuple[int, int]] = set()
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) in self.walls or (r, c) in self.goals:
                    continue
                up, down = self._is_wall(r - 1, c), self._is_wall(r + 1, c)
                left, right = self._is_wall(r, c - 1), self._is_wall(r, c + 1)
                if (up and left) or (up and right) or (down and left) or (down and right):
                    deadlocks.add((r, c))
        return deadlocks

    def _is_wall_deadlock(self, box: Tuple[int, int]) -> bool:
        """
        Kẹt tường: thùng bị chặn bởi tường ở hai phía đối diện, nên chỉ còn
        trượt được theo một trục, mà trên đoạn đó không có ô đích nào.

        Bị chặn trên dưới thì thùng chỉ trượt ngang, nên phải tìm đích trong
        cùng hàng; bị chặn hai bên thì thùng chỉ trượt dọc, nên tìm đích trong
        cùng cột.
        """
        r, c = box
        if box in self.goals:
            return False

        if self._is_wall(r - 1, c) and self._is_wall(r + 1, c):
            if not self._goal_in_row(r, c):
                return True

        if self._is_wall(r, c - 1) and self._is_wall(r, c + 1):
            if not self._goal_in_column(r, c):
                return True

        return False

    def _goal_in_row(self, r: int, c: int) -> bool:
        """Có ô đích trong đoạn hàng chứa ô (r, c) không, tính tới hai bức tường."""
        for cc in range(c - 1, -1, -1):
            if (r, cc) in self.goals:
                return True
            if self._is_wall(r, cc):
                break
        for cc in range(c + 1, self.width):
            if (r, cc) in self.goals:
                return True
            if self._is_wall(r, cc):
                break
        return False

    def _goal_in_column(self, r: int, c: int) -> bool:
        """Có ô đích trong đoạn cột chứa ô (r, c) không, tính tới hai bức tường."""
        for rr in range(r - 1, -1, -1):
            if (rr, c) in self.goals:
                return True
            if self._is_wall(rr, c):
                break
        for rr in range(r + 1, self.height):
            if (rr, c) in self.goals:
                return True
            if self._is_wall(rr, c):
                break
        return False

    def is_2x2_freeze_deadlock(self, boxes: Sequence[Tuple[int, int]]) -> bool:
        """Khối 2x2 gồm toàn tường và thùng, trong đó có thùng chưa nằm trên đích."""
        box_set = set(boxes)
        for r in range(self.height - 1):
            for c in range(self.width - 1):
                tiles = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                n_box = sum(1 for t in tiles if t in box_set)
                n_wall = sum(1 for t in tiles if self._is_wall(*t))
                if n_box + n_wall == 4 and n_box > 0:
                    if any(b not in self.goals for b in box_set if b in tiles):
                        return True
        return False

    def is_deadlock(self, boxes: Sequence[Tuple[int, int]]) -> bool:
        """
        Kiểm tra tình huống kẹt của cấu hình thùng hiện tại.

        Kết quả được ghi lại theo cấu hình vì các cấu hình lặp lại rất nhiều
        trong một episode, việc cache giúp bước mô phỏng nhanh hơn nhiều lần.
        """
        key = tuple(sorted(boxes))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute_deadlock(key)
            self._cache[key] = cached
        return cached

    def _compute_deadlock(self, boxes: Tuple[Tuple[int, int], ...]) -> bool:
        for b in boxes:
            if b in self.corner_deadlocks or self._is_wall_deadlock(b):
                return True
        return self.is_2x2_freeze_deadlock(boxes)


class SokobanTwoBoxesEnv(MDPEnv):
    """
    Hành động: 0=LÊN, 1=XUỐNG, 2=TRÁI, 3=PHẢI
    Trạng thái: (người chơi, hai thùng, trạng thái ô cứu hộ) -> mã số nguyên
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = ["LÊN", "XUỐNG", "TRÁI", "PHẢI"]
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self,
        map_path: Optional[str] = None,
        map_string: Optional[str] = None,
        max_steps: int = 120,
        step_penalty: float = -0.1,
        goal_reward: float = 100.0,
        single_box_reward: float = 15.0,
        deadlock_penalty: float = -50.0,
        rescue_bonus: float = 10.0,
        render_mode: Optional[str] = "ansi",
        seed: Optional[int] = None,
    ):
        if (map_path is None) == (map_string is None):
            raise ValueError("Chỉ cần truyền map_path hoặc map_string.")

        if map_path is not None:
            with open(map_path, "r", encoding="utf-8") as f:
                raw_lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
        else:
            raw_lines = [ln.rstrip("\r\n") for ln in map_string.strip().split("\n") if ln.strip()]

        self.height = len(raw_lines)
        self.width = max(len(ln) for ln in raw_lines)

        self.walls: Set[Tuple[int, int]] = set()
        self.goals: Set[Tuple[int, int]] = set()
        self.initial_boxes: List[Tuple[int, int]] = []
        self.initial_player: Optional[Tuple[int, int]] = None
        self.rescue_tile: Optional[Tuple[int, int]] = None

        for r, line in enumerate(raw_lines):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "G":
                    self.goals.add((r, c))
                elif ch == "B":
                    self.initial_boxes.append((r, c))
                elif ch == "P":
                    self.initial_player = (r, c)
                elif ch == "R":
                    self.rescue_tile = (r, c)
                elif ch == "+":
                    self.goals.add((r, c))
                    self.initial_player = (r, c)

        if len(self.initial_boxes) != 2:
            raise ValueError(f"Cần đúng 2 thùng, bản đồ có {len(self.initial_boxes)}.")
        if len(self.goals) != 2:
            raise ValueError(f"Cần đúng 2 ô đích, bản đồ có {len(self.goals)}.")
        if self.initial_player is None:
            raise ValueError("Bản đồ không có vị trí bắt đầu 'P'.")

        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.single_box_reward = single_box_reward
        self.deadlock_penalty = deadlock_penalty
        self.rescue_bonus = rescue_bonus
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.state_encoder = StateEncoder(self.height, self.width)
        self.deadlock_detector = DeadlockDetector(self.walls, self.goals, self.height, self.width)

        self.action_space = Discrete(4)
        self.observation_space = Discrete((self.height * self.width) ** 3 * 3)
        self.n_actions = 4

        self._rng = np.random.default_rng(seed)
        self.player: Tuple[int, int] = self.initial_player
        self.boxes: List[Tuple[int, int]] = list(self.initial_boxes)
        self.rescue_state = 0
        self.steps = 0
        self.rescue_activations = 0
        self.rescue_uses = 0

    def _apply(self, player: Tuple[int, int], boxes: List[Tuple[int, int]],
               rescue_state: int, action: int) -> Dict[str, Any]:
        """
        Áp dụng một hành động lên một trạng thái, không thay đổi biến thành viên.

        Cả step() và transition() đều gọi hàm này.
        """
        dr, dc = self.MOVES[action]
        boxes = list(boxes)
        player = tuple(player)
        prev_player = player
        next_player = (player[0] + dr, player[1] + dc)

        reward = self.step_penalty
        terminated = False
        is_push = False
        rescue_activated = False
        rescue_used = False
        prev_box_pos: Optional[Tuple[int, int]] = None
        moved_box_idx = -1
        n_goal_before = sum(1 for b in boxes if b in self.goals)

        if next_player in self.walls:
            pass
        elif next_player in boxes:
            idx = boxes.index(next_player)
            prev_box_pos = boxes[idx]
            next_box = (next_player[0] + dr, next_player[1] + dc)
            if next_box not in self.walls and next_box not in boxes:
                boxes[idx] = next_box
                player = next_player
                is_push = True
                moved_box_idx = idx
        else:
            player = next_player

        if self.rescue_tile is not None and player == self.rescue_tile and rescue_state == 0:
            rescue_state = 1
            reward += self.rescue_bonus
            rescue_activated = True

        n_goal_after = sum(1 for b in boxes if b in self.goals)
        if n_goal_after > n_goal_before:
            reward += self.single_box_reward * (n_goal_after - n_goal_before)

        if set(boxes) == self.goals:
            reward += self.goal_reward
            terminated = True
        elif self.deadlock_detector.is_deadlock(boxes):
            if rescue_state == 1 and is_push and prev_box_pos is not None:
                boxes[moved_box_idx] = prev_box_pos
                player = prev_player
                rescue_state = 2
                if not self.deadlock_detector.is_deadlock(boxes):
                    reward += self.rescue_bonus
                    rescue_used = True
                else:
                    reward += self.deadlock_penalty
                    terminated = True
            else:
                reward += self.deadlock_penalty
                terminated = True

        return {
            "player": player,
            "boxes": boxes,
            "rescue_state": rescue_state,
            "reward": reward,
            "terminated": terminated,
            "is_push": is_push,
            "boxes_on_goals": n_goal_after,
            "rescue_activated": rescue_activated,
            "rescue_used": rescue_used,
        }

    def reset(self) -> int:
        self.player = self.initial_player
        self.boxes = list(self.initial_boxes)
        self.rescue_state = 0
        self.steps = 0
        self.rescue_activations = 0
        self.rescue_uses = 0
        return self.state_encoder.encode_to_id(self.player, self.boxes, self.rescue_state)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        out = self._apply(self.player, self.boxes, self.rescue_state, action)
        if out["rescue_activated"]:
            self.rescue_activations += 1
        if out["rescue_used"]:
            self.rescue_uses += 1
        self.player = out["player"]
        self.boxes = out["boxes"]
        self.rescue_state = out["rescue_state"]
        terminated = out["terminated"]
        truncated = self.steps >= self.max_steps
        state_id = self.state_encoder.encode_to_id(self.player, self.boxes, self.rescue_state)
        is_solved = set(self.boxes) == self.goals
        is_deadlock = self.deadlock_detector.is_deadlock(self.boxes)
        if is_solved:
            outcome = "solved"
        elif is_deadlock:
            outcome = "deadlock"
        else:
            outcome = "timeout"
        info = {
            "player": self.player,
            "boxes": list(self.boxes),
            "rescue_state": self.rescue_state,
            "is_solved": is_solved,
            "is_deadlock": is_deadlock,
            "is_push": out["is_push"],
            "boxes_on_goals": out["boxes_on_goals"],
            "steps": self.steps,
            "outcome": outcome,
            "rescue_activations": self.rescue_activations,
            "rescue_uses": self.rescue_uses,
        }
        return state_id, out["reward"], terminated, truncated, info

    def transition(self, state_id: int, action: int) -> Tuple[int, float, bool]:
        """Mô hình MDP: (s, a) -> (s', r, terminated)."""
        player, boxes, rescue_state = self.state_encoder.decode_from_id(state_id)
        out = self._apply(player, list(boxes), rescue_state, action)
        next_id = self.state_encoder.encode_to_id(out["player"], out["boxes"], out["rescue_state"])
        return next_id, out["reward"], out["terminated"]

    def is_terminal(self, state_id: int) -> bool:
        """Episode kết thúc khi cả hai thùng vào đích, hoặc thùng bị kẹt."""
        _, boxes, _ = self.state_encoder.decode_from_id(state_id)
        if set(boxes) == self.goals:
            return True
        return self.deadlock_detector.is_deadlock(boxes)

    def agent_cell(self, state_id: int) -> Tuple[int, int]:
        """Ô đang đứng của người chơi tại trạng thái này."""
        player, _, _ = self.state_encoder.decode_from_id(state_id)
        return player

    def state_ids(self) -> List[int]:
        """Duyệt BFS lấy tất cả trạng thái tới được từ trạng thái đầu."""
        saved = (self.player, list(self.boxes), self.rescue_state, self.steps)
        try:
            start = self.reset()
            visited = {start}
            queue = [start]
            cache: Dict[Tuple[int, int], Tuple[int, float, bool]] = {}

            while queue:
                cur = queue.pop(0)
                if self.is_terminal(cur):
                    continue
                for a in range(self.n_actions):
                    if (cur, a) not in cache:
                        cache[(cur, a)] = self.transition(cur, a)
                    nxt = cache[(cur, a)][0]
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            return sorted(visited)
        finally:
            self.player, self.boxes, self.rescue_state, self.steps = saved

    def to_char_grid(self) -> List[List[str]]:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for (r, c) in self.walls:
            grid[r][c] = "#"
        for (r, c) in self.goals:
            grid[r][c] = "G"
        if self.rescue_tile is not None:
            r, c = self.rescue_tile
            grid[r][c] = "R"
        for b in self.boxes:
            grid[b[0]][b[1]] = "*" if b in self.goals else "B"
        pr, pc = self.player
        grid[pr][pc] = "+" if (pr, pc) in self.goals else "P"
        return grid

    def render(self) -> Optional[str]:
        grid = self.to_char_grid()
        out = "\n".join("".join(row) for row in grid)
        labels = {0: "chưa dùng", 1: "đã kích hoạt", 2: "đã dùng"}
        out += f"\n[Bước {self.steps}] Ô cứu hộ: {labels[self.rescue_state]}"
        out += f" | Thùng trên đích: {sum(1 for b in self.boxes if b in self.goals)}/2"
        return out

    def describe(self) -> str:
        return (f"Sokoban hai thùng: lưới {self.height}x{self.width}, "
                f"{self.n_actions} hành động, 2 thùng / 2 ô đích")
