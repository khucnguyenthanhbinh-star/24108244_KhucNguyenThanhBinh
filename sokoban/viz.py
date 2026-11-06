"""Vẽ bàn cờ, chính sách, hàm giá trị và xuất ảnh/GIF cho báo cáo."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# Màu và kiểu của từng loại ô
CELL_STYLE = {
    "#": ("#39414f", "#242a35", "wall"),
    ".": ("#f6f1e4", "#ddd3bd", "floor"),
    "G": ("#f6f1e4", "#ddd3bd", "goal"),
    "B": ("#c98a4b", "#96622f", "box"),
    "*": ("#2fa898", "#1d7a70", "box"),
    "P": ("#ef6f51", "#c04c33", "player"),
    "+": ("#ef6f51", "#c04c33", "player"),
    "R": ("#e9c46a", "#b8952f", "rescue"),
}

ACTION_ARROW = {
    0: (0, -1),    # LÊN
    1: (0, 1),     # XUỐNG
    2: (-1, 0),    # TRÁI
    3: (1, 0),     # PHẢI
}

ACTION_LABEL = {0: "Lên", 1: "Xuống", 2: "Trái", 3: "Phải"}

LEGEND = [
    ("#", "Tường"),
    (".", "Nền"),
    ("G", "Ô đích"),
    ("B", "Thùng"),
    ("*", "Thùng trên đích"),
    ("P", "Người chơi"),
    ("R", "Ô cứu hộ"),
]

LABELS = {
    "vi": {
        "value": "Giá trị",
        "step": "bước",
        "reward": "thưởng",
        "mean_reward": "Phần thưởng trung bình",
        "failed": "(thất bại)",
        "legend": [("Tường", "Nền", "Ô đích", "Thùng", "Thùng trên đích",
                    "Người chơi", "Ô cứu hộ")],
    },
    "en": {
        "value": "Value",
        "step": "step",
        "reward": "reward",
        "mean_reward": "Mean reward",
        "failed": "(failed)",
        "legend": [("Wall", "Floor", "Goal", "Box", "Box on goal",
                    "Player", "Rescue tile")],
    },
}

_LANG = "vi"


def set_language(lang: str) -> None:
    """Chọn ngôn ngữ cho các nhãn vẽ trong hình (vi hoặc en)."""
    global _LANG
    if lang not in LABELS:
        raise ValueError(f"Ngôn ngữ không hỗ trợ: {lang}")
    _LANG = lang


def text(key: str) -> str:
    """Lấy một nhãn theo ngôn ngữ hiện tại."""
    return LABELS[_LANG][key]


def _rounded(ax, x, y, w, h, face, edge, lw=1.2, radius=0.12, z=1, inset=0.04):
    patch = FancyBboxPatch((x + inset, y + inset), w - 2 * inset, h - 2 * inset,
                            boxstyle=f"round,pad=0,rounding_size={radius}",
                            facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def _crate(ax, x, y, face, edge):
    """Vẽ một thùng gỗ có gân chéo."""
    _rounded(ax, x, y, 1, 1, face, edge, lw=1.6, radius=0.11, z=3)
    ax.add_patch(Rectangle((x + 0.30, y + 0.30), 0.40, 0.40, facecolor=edge,
                           edgecolor="none", alpha=0.30, zorder=4))


def _target(ax, cx, cy):
    """Vẽ ô đích: hai vòng tròn đồng tâm."""
    ax.add_patch(Circle((cx, cy), 0.26, facecolor="none", edgecolor="#2fa898",
                        linewidth=3.0, zorder=2))
    ax.add_patch(Circle((cx, cy), 0.13, facecolor="none", edgecolor="#2fa898",
                        linewidth=2.0, zorder=2))


def _player(ax, cx, cy):
    """Vẽ người chơi: hình tròn có điểm sáng."""
    ax.add_patch(Circle((cx + 0.02, cy + 0.03), 0.25, facecolor="#00000022",
                        edgecolor="none", zorder=4))
    ax.add_patch(Circle((cx, cy), 0.25, facecolor="#ef6f51", edgecolor="#ffffff",
                        linewidth=1.8, zorder=5))
    ax.add_patch(Circle((cx - 0.07, cy - 0.08), 0.07, facecolor="#ffd9cd",
                        edgecolor="none", zorder=6))


def draw_grid(ax, grid: Sequence[Sequence[str]], title: str = "",
              arrows: Optional[Dict[Tuple[int, int], int]] = None) -> None:
    """Vẽ một bàn cờ từ ma trận ký tự."""
    height = len(grid)
    width = max(len(row) for row in grid)

    ax.set_xlim(-0.4, width + 0.4)
    ax.set_ylim(height + 0.4, -0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Khung bàn cờ
    _rounded(ax, -0.32, -0.32, width + 0.64, height + 0.64, "#e7dfcd",
             "#cdc2aa", lw=1.6, radius=0.22, z=0)

    for r in range(height):
        for c in range(len(grid[r])):
            ch = grid[r][c]
            face, edge, kind = CELL_STYLE.get(ch, CELL_STYLE["."])
            if kind == "wall":
                _rounded(ax, c, r, 1, 1, face, edge, lw=1.4, radius=0.1, z=2)
                ax.add_patch(Rectangle((c + 0.16, r + 0.14), 0.68, 0.14,
                                       facecolor="#ffffff", edgecolor="none",
                                       alpha=0.10, zorder=3))
            else:
                _rounded(ax, c, r, 1, 1, face, edge, lw=1.1, radius=0.08, z=1)
                if kind == "goal":
                    _target(ax, c + 0.5, r + 0.5)
                elif kind == "box":
                    _crate(ax, c, r, face, edge)
                elif kind == "player":
                    _player(ax, c + 0.5, r + 0.5)
                elif kind == "rescue":
                    ax.add_patch(Circle((c + 0.5, r + 0.5), 0.27,
                                         facecolor="none", edgecolor="#ffffff",
                                         linewidth=2.2, zorder=4))
                    ax.add_patch(Rectangle((c + 0.40, r + 0.29), 0.20, 0.42,
                                           facecolor="#8a5a1d", edgecolor="none",
                                           zorder=5))
                    ax.add_patch(Rectangle((c + 0.29, r + 0.40), 0.42, 0.20,
                                           facecolor="#8a5a1d", edgecolor="none",
                                           zorder=5))

    if arrows:
        for (r, c), action in arrows.items():
            dr, dc = ACTION_ARROW[action]
            ax.arrow(c + 0.5, r + 0.5, dc * 0.36, dr * 0.36, width=0.045,
                     head_width=0.22, head_length=0.22, color="#264653",
                     length_includes_head=True, zorder=7)

    if title:
        ax.set_title(title, fontsize=12, weight="bold", color="#264653", pad=10)


def add_legend(fig, ax):
    """Chú thích các loại ô, đặt dưới bàn cờ."""
    handles = []
    names = LABELS[_LANG]["legend"]
    for (ch, _), label in zip(LEGEND, names):
        face, edge, kind = CELL_STYLE[ch]
        if kind == "player":
            handle = Line2D([], [], marker="o", color=edge, markerfacecolor=face,
                            markeredgecolor="#ffffff", markersize=11, label=label)
        elif kind == "box":
            handle = Line2D([], [], marker="s", color=edge, markerfacecolor=face,
                            markersize=11, label=label)
        elif kind == "goal":
            handle = Line2D([], [], marker="o", color=face, markerfacecolor="none",
                            markeredgecolor="#2fa898", markersize=11, label=label)
        else:
            handle = Line2D([], [], marker="s", color=edge, markerfacecolor=face,
                            markersize=11, label=label)
        handles.append(handle)
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))


def draw_board(ax, env, title: str = "",
               arrows: Optional[Dict[Tuple[int, int], int]] = None) -> None:
    draw_grid(ax, env.to_char_grid(), title=title, arrows=arrows)


def board_figure(env, title: str = "", figsize: Tuple[float, float] = (5.0, 4.6)):
    """Tạo figure vẽ bàn cờ, cùng bố cục với khung hình của file GIF."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fbf8f2")
    draw_board(ax, env, title=title)
    fig.tight_layout()
    return fig, ax


def snapshot(env) -> List[List[str]]:
    """Chụp lại trạng thái hiện tại của môi trường thành ma trận ký tự."""
    return [list(row) for row in env.to_char_grid()]


def _draw_heatmap(ax, env, state_values: Dict[int, float], vmin: float, vmax: float,
                  title: str = ""):
    """Vẽ một nhiệt đồ giá trị với khoảng màu cho trước."""
    height, width = env.height, env.width
    board = np.full((height, width), np.nan)

    for state_id, value in state_values.items():
        r, c = env.agent_cell(state_id)
        if np.isnan(board[r][c]) or value > board[r][c]:
            board[r][c] = value

    for (r, c) in env.walls:
        board[r][c] = np.nan

    im = ax.imshow(board, cmap="YlGnBu", origin="upper", interpolation="nearest",
                   extent=(-0.5, width - 0.5, height - 0.5, -0.5),
                   vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for (r, c) in env.walls:
        _rounded(ax, c - 0.5, r - 0.5, 1, 1, "#39414f", "#242a35", lw=1.2,
                 radius=0.1, z=3, inset=0.0)
    for (r, c) in env.goals:
        ax.add_patch(Circle((c, r), 0.24, facecolor="none", edgecolor="#ef6f51",
                            linewidth=2.4, zorder=4))
    boxes = getattr(env, "initial_boxes", getattr(env, "boxes", []))
    for (r, c) in boxes:
        _crate(ax, c - 0.5, r - 0.5, "#d4a373", "#8a5a1d")
    if env.rescue_tile is not None:
        r, c = env.rescue_tile
        _rounded(ax, c - 0.5, r - 0.5, 1, 1, "#e9c46a", "#b8952f", lw=1.2,
                 radius=0.1, z=4, inset=0.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, weight="bold", color="#264653")
    return im


def plot_value_heatmap(ax, env, state_values: Dict[int, float],
                       title: str = "Hàm giá trị V(s)") -> None:
    """
    Vẽ nhiệt đồ giá trị trên lưới.

    Với mỗi ô, lấy giá trị lớn nhất trong số các trạng thái mà người chơi
    đứng ở ô đó, cho biết tác tử muốn đến ô nào nhất.
    """
    values = list(state_values.values())
    vmin, vmax = (min(values), max(values)) if values else (0.0, 1.0)
    _draw_heatmap(ax, env, state_values, vmin, vmax, title)


def record_episode(env, agent, max_steps: int = 200) -> List[List[str]]:
    """Chạy một episode theo chính sách tham lam và ghi lại từng khung hình."""
    return run_episode(env, agent, max_steps)["frames"]


def run_episode(env, agent, max_steps: int = 200) -> Dict[str, Any]:
    """
    Chạy một episode theo chính sách tham lam.

    Trả về các khung hình, đường đi của người chơi và thông tin kết quả.
    """
    state_id = env.reset()
    frames = [snapshot(env)]
    path_cells = [env.player]
    total = 0.0
    terminated = truncated = False
    info: Dict[str, Any] = {}

    while not (terminated or truncated) and len(path_cells) <= max_steps:
        action = agent.select_action(state_id, explore=False)
        state_id, reward, terminated, truncated, info = env.step(action)
        frames.append(snapshot(env))
        path_cells.append(env.player)
        total += reward

    return {
        "frames": frames,
        "path": path_cells,
        "steps": len(path_cells) - 1,
        "reward": total,
        "solved": bool(info.get("is_solved")),
        "deadlock": bool(info.get("is_deadlock")),
    }


def draw_path(ax, path_cells: Sequence[Tuple[int, int]]) -> None:
    """Vẽ đường đi của người chơi lên bàn cờ với gradient màu và điểm mốc bắt đầu/kết thúc."""
    n_steps = len(path_cells) - 1
    for i in range(n_steps):
        r0, c0 = path_cells[i]
        r1, c1 = path_cells[i + 1]
        if (r0, c0) == (r1, c1):
            continue
        # Color gradient: lighter blue for early steps, deeper navy for late steps
        prog = (i + 1) / max(1, n_steps)
        color = plt.cm.Blues(0.45 + 0.55 * prog)
        ax.arrow(c0 + 0.5, r0 + 0.5, c1 - c0, r1 - r0, width=0.038,
                 head_width=0.20, head_length=0.20, color=color,
                 alpha=0.90, length_includes_head=True, zorder=6)
    if path_cells:
        r_start, c_start = path_cells[0]
        ax.add_patch(Circle((c_start + 0.5, r_start + 0.5), 0.18, facecolor="#e76f51",
                            edgecolor="#ffffff", linewidth=1.2, zorder=7))
        r_end, c_end = path_cells[-1]
        ax.add_patch(Circle((c_end + 0.5, r_end + 0.5), 0.14, facecolor="#1d3557",
                            edgecolor="#ffffff", linewidth=1.0, zorder=7))


def plot_trajectories(env, results: Dict[str, Dict[str, Any]], path: str,
                      title: str = "Đường đi của chính sách tham lam") -> str:
    """
    Vẽ đường đi mà mỗi thuật toán đi trên cùng một bản đồ, đặt cạnh nhau.
    """
    names = list(results.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.5))
    fig.patch.set_facecolor("#fbf8f2")
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        res = results[name]
        label = (f"{name}\n{res['steps']} {text('step')}, "
                 f"{text('reward')} {res['reward']:.1f}")
        if not res["solved"]:
            label += "\n" + text("failed")
        draw_grid(ax, res["frames"][-1], title=label)
        # Draw dashed outlines for initial box locations
        for (br, bc) in getattr(env, "initial_boxes", []):
            ax.add_patch(Rectangle((bc + 0.12, br + 0.12), 0.76, 0.76,
                                   facecolor="none", edgecolor="#8a5a1d",
                                   linestyle="--", linewidth=1.6, zorder=5))
        draw_path(ax, res["path"])

    fig.suptitle(title, fontsize=12, weight="bold", color="#264653", y=1.05)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fbf8f2")
    plt.close(fig)
    return path


def save_filmstrip(frames: Sequence[Sequence[Sequence[str]]], env, path: str,
                   n: int = 5, title: str = "") -> str:
    """
    Chọn n khung hình đều nhau và xếp thành một dải, dùng để in trong báo cáo.
    """
    if len(frames) < n:
        n = len(frames)
    idx = [round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(n)]
    idx = sorted(set(idx))

    fig, axes = plt.subplots(1, len(idx), figsize=(2.5 * len(idx), 3.0))
    fig.patch.set_facecolor("#fbf8f2")
    if len(idx) == 1:
        axes = [axes]

    for ax, i in zip(axes, idx):
        draw_grid(ax, frames[i], title=f"{text('step')} {i}")

    if title:
        fig.suptitle(title, fontsize=12, weight="bold", color="#264653", y=1.04)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fbf8f2")
    plt.close(fig)
    return path


def save_episode_gif(frames: Sequence[Sequence[Sequence[str]]], env, path: str,
                     title: str = "", fps: int = 4) -> str:
    """Xuất một episode thành file GIF."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    fig.patch.set_facecolor("#fbf8f2")

    def update(i: int):
        ax.clear()
        draw_grid(ax, frames[i], title=f"{title}  |  bước {i}/{len(frames) - 1}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def save_episode_png(frames: Sequence[Sequence[Sequence[str]]], env, folder: str,
                     prefix: str = "frame") -> List[str]:
    """Xuất từng khung hình thành file PNG riêng."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, grid in enumerate(frames):
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        fig.patch.set_facecolor("white")
        draw_grid(ax, grid, title=f"bước {i}/{len(frames) - 1}")
        p = os.path.join(folder, f"{prefix}_{i:03d}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    return paths


def plot_learning_curves(histories: Dict[str, Dict[str, Any]], path: str,
                         title: str = "Đường học theo episode", window: int = 50) -> str:
    """Vẽ đường học (phần thưởng trung bình động) của nhiều thuật toán."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for name, history in histories.items():
        rewards = np.asarray(history["episode_rewards"], dtype=float)
        if rewards.size < window:
            continue
        kernel = np.ones(window) / window
        smooth = np.convolve(rewards, kernel, mode="valid")
        x = np.arange(window - 1, rewards.size)
        ax.plot(x, smooth, label=name, linewidth=1.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel(text("mean_reward"))
    ax.set_title(title, fontsize=12, weight="bold", color="#264653")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_maps_grid(envs: Sequence[Any], titles: Sequence[str], path: str,
                   title: str = "Bốn bản đồ của bốn setting") -> str:
    """Vẽ các bản đồ cạnh nhau thành một hình."""
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.9))
    fig.patch.set_facecolor("#fbf8f2")
    if n == 1:
        axes = [axes]
    for ax, env, t in zip(axes, envs, titles):
        draw_board(ax, env, title=t)
    fig.suptitle(title, fontsize=13, weight="bold", color="#264653", y=1.02)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_value_heatmaps(env, panels: Sequence[Tuple[str, Dict[int, float]]],
                        path: str, title: str = "") -> str:
    """
    Vẽ nhiều nhiệt đồ giá trị cạnh nhau, dùng chung một thang màu.

    Dùng chung thang màu là điều kiện để so sánh được giữa các thuật toán.
    """
    values = [v for _, vals in panels for v in vals.values()]
    if not values:
        raise ValueError("Không có giá trị để vẽ.")
    vmin, vmax = min(values), max(values)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.0))
    fig.patch.set_facecolor("#fbf8f2")
    if len(panels) == 1:
        axes = [axes]

    im = None
    for ax, (name, vals) in zip(axes, panels):
        im = _draw_heatmap(ax, env, vals, vmin, vmax, name)

    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label(text("value"), fontsize=9)
    cb.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=12, weight="bold", color="#264653", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fbf8f2")
    plt.close(fig)
    return path


def plot_policy_arrows(env, panels: Sequence[Tuple[str, Dict[Tuple[int, int], int]]],
                       path: str, title: str = "") -> str:
    """
    Vẽ hành động mà chính sách chọn tại mỗi ô, với hai thùng đặt ở vị trí ban đầu.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.6))
    fig.patch.set_facecolor("#fbf8f2")
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, actions) in zip(axes, panels):
        draw_grid(ax, env.to_char_grid(), title=name)
        for (r, c), action in actions.items():
            dr, dc = ACTION_ARROW[action]
            ax.arrow(c + 0.5, r + 0.5, dc * 0.32, dr * 0.32, width=0.04,
                     head_width=0.2, head_length=0.2, color="#264653",
                     length_includes_head=True, zorder=7)

    if title:
        fig.suptitle(title, fontsize=12, weight="bold", color="#264653", y=1.03)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fbf8f2")
    plt.close(fig)
    return path


def plot_learning_curves_multi(histories: Dict[str, Sequence[Dict[str, Any]]],
                               path: str, title: str = "", window: int = 50,
                               ylabel: str = "") -> str:
    """
    Đường học của nhiều thuật toán, mỗi thuật toán là trung bình của nhiều
    seed kèm dải sai số.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for name, runs in histories.items():
        curves = []
        win = max(2, window // 5) if rewards.size <= 200 else window
        step = 5 if rewards.size <= 200 else 1
        for run in runs:
            rewards = np.asarray(run.get("rewards", run.get("episode_rewards", [])),
                                dtype=float)
            if rewards.size < win:
                win = max(2, rewards.size // 4)
            if rewards.size < win:
                continue
            kernel = np.ones(win) / win
            curves.append(np.convolve(rewards, kernel, mode="valid"))
        if not curves:
            continue
        n = min(len(c) for c in curves)
        stack = np.vstack([c[:n] for c in curves])
        x = (np.arange(win - 1, win - 1 + n) + 1) * step
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(x, mean, label=name, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel or text("mean_reward"))
    ax.set_title(title, fontsize=12, weight="bold", color="#264653")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_grouped_bars(summary: Dict[str, Dict[str, float]], path: str,
                      metric: str = "success_rate", title: str = "",
                      order: Optional[Sequence[str]] = None) -> str:
    """
    Biểu đồ cột nhóm: mỗi nhóm là một setting, mỗi cột là một thuật toán,
    có thanh sai số.
    """
    import numpy as np

    settings = sorted({k.split("|")[0] for k in summary})
    algos = list(order) if order else []
    for k in summary:
        a = k.split("|")[1]
        if a not in algos:
            algos.append(a)

    n_algo = len(algos)
    x = np.arange(len(settings))
    width = 0.8 / max(n_algo, 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#457b9d", "#8d99ae"]
    for i, algo in enumerate(algos):
        vals, errs = [], []
        for s in settings:
            m = summary.get(f"{s}|{algo}")
            vals.append(m.get(metric, 0.0) if m else 0.0)
            errs.append(m.get(metric + "_std", 0.0) if m else 0.0)
        if metric == "success_rate":
            lower_err = [min(v, e) for v, e in zip(vals, errs)]
            upper_err = [min(max(0.0, 1.0 - v), e) for v, e in zip(vals, errs)]
            yerr = [lower_err, upper_err]
        else:
            yerr = errs
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
               yerr=yerr, capsize=3, label=algo, color=colors[i % len(colors)],
               edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("setting", "Setting ") for s in settings])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, fontsize=12, weight="bold", color="#264653")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, ncol=2, frameon=True)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_ablation(with_rescue: Dict[str, Dict[str, float]],
                  without_rescue: Dict[str, Dict[str, float]],
                  path: str, title: str = "") -> str:
    """
    So sánh kết quả khi có và không có ô cứu hộ, trên cùng một bản đồ.
    """
    import numpy as np

    algos = [a for a in without_rescue if a in with_rescue]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    x = np.arange(len(algos))
    width = 0.36

    for ax, metric, label in [(axes[0], "success_rate", "Success rate"),
                              (axes[1], "mean_steps", "Steps")]:
        a = [with_rescue[k].get(metric, 0.0) for k in algos]
        ae = [with_rescue[k].get(metric + "_std", 0.0) for k in algos]
        b = [without_rescue[k].get(metric, 0.0) for k in algos]
        be = [without_rescue[k].get(metric + "_std", 0.0) for k in algos]
        if metric == "success_rate":
            ae_lower = [min(v, e) for v, e in zip(a, ae)]
            ae_upper = [min(max(0.0, 1.0 - v), e) for v, e in zip(a, ae)]
            yerr_a = [ae_lower, ae_upper]
            be_lower = [min(v, e) for v, e in zip(b, be)]
            be_upper = [min(max(0.0, 1.0 - v), e) for v, e in zip(b, be)]
            yerr_b = [be_lower, be_upper]
            ax.set_ylim(0, 1.05)
        else:
            yerr_a, yerr_b = ae, be
        ax.bar(x - width / 2, a, width, yerr=yerr_a, capsize=3, label="with rescue tile",
               color="#2a9d8f", edgecolor="white", linewidth=0.8)
        ax.bar(x + width / 2, b, width, yerr=yerr_b, capsize=3, label="without rescue tile",
               color="#e9c46a", edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(" Control", "\nControl") for a in algos],
                           fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, frameon=True)
        if metric == "success_rate":
            ax.set_ylim(0, 1.15)

    if title:
        fig.suptitle(title, fontsize=12, weight="bold", color="#264653", y=1.02)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison_bar(metrics: Dict[str, Dict[str, float]], path: str,
                         metric: str = "success_rate",
                         title: str = "So sánh các thuật toán",
                         order: Optional[Sequence[str]] = None) -> str:
    """Vẽ biểu đồ cột so sánh một chỉ số giữa các thuật toán."""
    if order:
        names = [n for n in order if n in metrics]
        names += [n for n in metrics if n not in names]
    else:
        names = list(metrics.keys())
    values = [metrics[n].get(metric, 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(names, values, color="#2a9d8f", edgecolor="#1b736a", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)

    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, fontsize=12, weight="bold", color="#264653")
    ax.set_ylim(0, max(values + [1.0]) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
