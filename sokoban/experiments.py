"""Chạy thực nghiệm: huấn luyện, so sánh, phân tích độ nhạy tham số và sinh hình."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sokoban.agents import (
    ExpectedSARSAAgent,
    MonteCarloControlAgent,
    PolicyIteration,
    QLearningAgent,
    SARSAAgent,
    ValueIteration,
)
from sokoban.env import SokobanTwoBoxesEnv
from sokoban import viz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(ROOT, "maps")
FIGURES_DIR = os.path.join(ROOT, "report", "figures")
MODELS_DIR = os.path.join(ROOT, "saved_models")

SETTINGS: List[Dict[str, str]] = [
    {"key": "setting1", "name": "Setting 1: bản đồ nhỏ", "map": "map_setting1_small.txt"},
    {"key": "setting2", "name": "Setting 2: hành lang hẹp", "map": "map_setting2_corridor.txt"},
    {"key": "setting3", "name": "Setting 3: có ô cứu hộ", "map": "map_setting3_rescue.txt"},
    {"key": "setting4", "name": "Setting 4: bản đồ nâng cao", "map": "map_setting4_advanced.txt"},
]

TEXT = {
    "vi": {
        "maps": "Bốn bản đồ của bốn setting",
        "learning": "Đường học trên {label}",
        "success": "Tỉ lệ thành công trên {label}",
        "value": "Hàm giá trị trên {label} (cùng thang màu)",
        "policy": "Hành động của chính sách trên {label}",
        "traj": "Đường đi của chính sách tham lam trên {label}",
        "film": "Một episode của Q-Learning trên {label}",
        "ablation": "Ảnh hưởng của ô cứu hộ trên Setting 3",
        "optimal": "(tối ưu)",
        "settings": ["Setting 1: bản đồ nhỏ", "Setting 2: hành lang hẹp",
                     "Setting 3: có ô cứu hộ", "Setting 4: bản đồ nâng cao"],
    },
    "en": {
        "maps": "The four maps of the four settings",
        "learning": "Learning curves on {label}",
        "success": "Success rate on {label}",
        "value": "Value function on {label} (shared colour scale)",
        "policy": "Policy actions on {label}",
        "traj": "Greedy policy trajectories on {label}",
        "film": "One episode of Q-Learning on {label}",
        "ablation": "Effect of the rescue tile on Setting 3",
        "optimal": "(optimal)",
        "settings": ["Setting 1: small map", "Setting 2: narrow corridor",
                     "Setting 3: with rescue tile", "Setting 4: advanced map"],
    },
}

# Hai thuật toán quy hoạch động cần dựng mô hình nên chậm hơn, đề yêu cầu
# so sánh Policy Iteration trên phiên bản thu nhỏ
DP_SETTINGS = ["setting1", "setting2", "setting3", "setting4"]

TD_ALGOS = ["Q-Learning", "SARSA", "Expected SARSA", "Monte Carlo Control"]
DP_ALGOS = ["Value Iteration", "Policy Iteration"]
ALL_ALGOS = DP_ALGOS + TD_ALGOS

DEFAULT_EPISODES = 1000
DEFAULT_SEEDS = 3


def json_default(obj: Any) -> Any:
    """Chuyển các kiểu số của numpy về kiểu chuẩn để ghi JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Không ghi được kiểu {type(obj).__name__} ra JSON.")


def figures_dir(lang: str = "vi") -> str:
    """Thu muc chua hinh, tach rieng cho ban tieng Viet va tieng Anh."""
    name = "figures" if lang == "vi" else "figures_en"
    return os.path.join(ROOT, "report", name)


def setting_label(setting_key: str) -> str:
    return setting_key.replace("setting", "Setting ")


def map_path(setting_key: str) -> str:
    for s in SETTINGS:
        if s["key"] == setting_key:
            return os.path.join(MAPS_DIR, s["map"])
    raise KeyError(setting_key)


def make_env(setting_key: str, **overrides: Any) -> SokobanTwoBoxesEnv:
    """Tạo môi trường cho một setting, cho phép ghi đè tham số phần thưởng."""
    return SokobanTwoBoxesEnv(map_path=map_path(setting_key), **overrides)


def make_env_from_string(map_string: str, **overrides: Any) -> SokobanTwoBoxesEnv:
    """Tạo môi trường từ nội dung bản đồ (dùng cho thí nghiệm ablation)."""
    return SokobanTwoBoxesEnv(map_string=map_string, **overrides)


# --------------------------------------------------------------------------
# Chạy nhiều seed song song
# --------------------------------------------------------------------------
def _seed_task(task):
    """
    Một lượt chạy độc lập: huấn luyện một thuật toán trên một setting với
    một seed, rồi đánh giá chính sách tham lam.
    """
    (algo, setting, seed, episodes, map_string,
     env_kwargs, agent_kwargs) = task
    if map_string is None:
        env = make_env(setting, **env_kwargs)
    else:
        env = make_env_from_string(map_string, **env_kwargs)
    agent = build_agent(algo, env, seed=seed, **agent_kwargs)
    t0 = time.time()
    history = agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
    metrics = agent.evaluate(num_episodes=50)
    metrics.update({
        "algorithm": algo,
        "setting": setting,
        "seed": seed,
        "episodes": episodes,
        "wall_clock": time.time() - t0,
        "q_table_size": history["q_table_size"],
        "rewards": history["episode_rewards"][::5],
    })
    return f"{setting}|{algo}|{seed}", metrics


def run_seeds(algorithms: Sequence[str], settings: Sequence[str], seeds: int,
              episodes: int = DEFAULT_EPISODES, workers: int = 0,
              map_string: Optional[str] = None,
              env_overrides: Optional[Dict[str, Any]] = None,
              agent_overrides: Optional[Dict[str, Any]] = None,
              progress: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Chạy tất cả tổ hợp (thuật toán, setting, seed) theo nhiều tiến trình.

    Kết quả trả về dạng {f"{setting}|{algo}": [dòng của từng seed]}.
    """
    import multiprocessing as mp

    env_kwargs = env_overrides or {}
    agent_kwargs = agent_overrides or {}
    tasks = [(algo, setting, seed, episodes, map_string, env_kwargs, agent_kwargs)
             for setting in settings for algo in algorithms for seed in range(seeds)]
    total = len(tasks)
    if progress:
        print(f"Chạy {total} lượt trên {workers or mp.cpu_count()} tiến trình...")

    results: Dict[str, List[Dict[str, Any]]] = {}
    if workers == 1:
        it = map(_seed_task, tasks)
        for i, (key, metrics) in enumerate(it, 1):
            results.setdefault(key.rsplit("|", 1)[0], []).append(metrics)
            if progress and i % 10 == 0:
                print(f"  {i}/{total}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers or mp.cpu_count()) as pool:
            for i, (key, metrics) in enumerate(
                    pool.imap_unordered(_seed_task, tasks, chunksize=1), 1):
                results.setdefault(key.rsplit("|", 1)[0], []).append(metrics)
                if progress and i % 10 == 0:
                    print(f"  {i}/{total}", flush=True)

    for runs in results.values():
        runs.sort(key=lambda r: r["seed"])
    return results


def summarize_seeds(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Gộp nhiều seed thành trung bình và độ lệch chuẩn cho từng cặp."""
    keys = ["success_rate", "deadlock_rate", "timeout_rate", "mean_reward",
            "mean_discounted_reward", "mean_steps", "mean_rescue_activations",
            "mean_rescue_uses", "wall_clock"]
    out: Dict[str, Dict[str, float]] = {}
    for key, runs in results.items():
        row: Dict[str, float] = {"n_seeds": len(runs)}
        for k in keys:
            vals = [r[k] for r in runs]
            row[k] = float(np.mean(vals))
            row[k + "_std"] = float(np.std(vals))
        out[key] = row
    return out


def build_agent(algo: str, env: SokobanTwoBoxesEnv, seed: int = 0, **kwargs: Any):
    if algo == "Value Iteration":
        return ValueIteration(env)
    if algo == "Policy Iteration":
        return PolicyIteration(env)
    if algo == "Q-Learning":
        return QLearningAgent(env, seed=seed, **kwargs)
    if algo == "SARSA":
        return SARSAAgent(env, seed=seed, **kwargs)
    if algo == "Expected SARSA":
        return ExpectedSARSAAgent(env, seed=seed, **kwargs)
    if algo == "Monte Carlo Control":
        return MonteCarloControlAgent(env, seed=seed, **kwargs)
    raise KeyError(algo)


def algos_for(setting_key: str) -> List[str]:
    if setting_key in DP_SETTINGS:
        return list(ALL_ALGOS)
    return list(TD_ALGOS)


STATE_COUNT: Dict[str, int] = {}


def state_count(setting_key: str) -> int:
    """Số trạng thái tới được của một setting, tính một lần rồi lưu lại."""
    if setting_key not in STATE_COUNT:
        STATE_COUNT[setting_key] = len(make_env(setting_key).state_ids())
    return STATE_COUNT[setting_key]


def train_one(algo: str, setting_key: str, episodes: int = DEFAULT_EPISODES,
              seed: int = 0, eval_episodes: int = 20) -> Tuple[Dict[str, Any], Any]:
    """Huấn luyện một thuật toán trên một setting và đánh giá chính sách."""
    env = make_env(setting_key)
    agent = build_agent(algo, env, seed=seed)
    t0 = time.time()
    history = agent.train(num_episodes=episodes, eval_interval=max(1, episodes // 4),
                          eval_episodes=eval_episodes)
    metrics = agent.evaluate(num_episodes=50)
    history.update(metrics)
    history["setting"] = setting_key
    history["seed"] = seed
    history["wall_clock"] = time.time() - t0
    history["n_states"] = state_count(setting_key)
    return history, agent


def run_all_training(episodes: int = DEFAULT_EPISODES, seeds: int = DEFAULT_SEEDS,
                     out_file: str = "training_results.json") -> Dict[str, Any]:
    """Huấn luyện tất cả thuật toán trên tất cả setting, lưu kết quả ra JSON."""
    all_results: Dict[str, Any] = {}
    print(f"{'Thuật toán':<22}{'Setting':<12}{'Seed':<6}{'Thành công':<12}"
          f"{'Kẹt':<8}{'Thời gian'}")

    for setting in SETTINGS:
        key = setting["key"]
        for algo in algos_for(key):
            for seed in range(seeds):
                history, _ = train_one(algo, key, episodes=episodes, seed=seed)
                all_results.setdefault(f"{key}|{algo}", []).append(history)
                print(f"{algo:<22}{key:<12}{seed:<6}"
                      f"{history['success_rate']:<12.2f}{history['deadlock_rate']:<8.2f}"
                      f"{history['wall_clock']:.1f}s")

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, out_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=json_default)
    print(f"\nĐã lưu kết quả: {path}")
    return all_results


def aggregate(all_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Gộp nhiều seed thành số trung bình cho từng cặp (setting, thuật toán)."""
    out: Dict[str, Dict[str, float]] = {}
    for key, runs in all_results.items():
        setting, algo = key.split("|")
        out[key] = {
            "success_rate": float(np.mean([r["success_rate"] for r in runs])),
            "success_std": float(np.std([r["success_rate"] for r in runs])),
            "deadlock_rate": float(np.mean([r["deadlock_rate"] for r in runs])),
            "mean_reward": float(np.mean([r["mean_reward"] for r in runs])),
            "mean_steps": float(np.mean([r["mean_steps"] for r in runs])),
            "training_time": float(np.mean([r["training_time"] for r in runs])),
            "n_states": int(runs[0]["n_states"]),
        }
    return out


def evaluate_comparison(episodes: int = DEFAULT_EPISODES, seeds: int = DEFAULT_SEEDS,
                         out_file: str = "comparison.json") -> Dict[str, Any]:
    """Chạy lại toàn bộ và lưu bảng so sánh."""
    all_results = run_all_training(episodes=episodes, seeds=seeds,
                                   out_file="training_results.json")
    summary = aggregate(all_results)
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, out_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print("\n" + "=" * 82)
    print(f"{'Setting':<12}{'Thuật toán':<22}{'Thành công':<12}{'Kẹt':<8}"
          f"{'Bước TB':<9}{'Số state'}")
    print("=" * 82)
    for key, m in summary.items():
        setting, algo = key.split("|")
        print(f"{setting:<12}{algo:<22}{m['success_rate']:<12.2f}"
              f"{m['deadlock_rate']:<8.2f}{m['mean_steps']:<9.1f}{m['n_states']}")
    return summary


def sensitivity_analysis(episodes: int = 600, seeds: int = 3,
                         setting_key: str = "setting3",
                         out_file: str = "sensitivity.json") -> Dict[str, Any]:
    """Phân tích độ nhạy theo alpha, gamma và cấu hình phần thưởng."""
    results: Dict[str, Any] = {"setting": setting_key, "seeds": seeds,
                              "alpha": {}, "gamma": {}, "reward": {}}
    print(f"Phân tích độ nhạy trên {setting_key}, {seeds} seed\n")

    def run(env_kwargs: Dict[str, Any], **agent_kwargs: Any) -> Dict[str, float]:
        rewards, successes, steps, deadlocks = [], [], [], []
        for seed in range(seeds):
            env = make_env(setting_key, **env_kwargs)
            agent = QLearningAgent(env, seed=seed, **agent_kwargs)
            agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
            m = agent.evaluate(num_episodes=30)
            rewards.append(m["mean_reward"])
            successes.append(m["success_rate"])
            steps.append(m["mean_steps"])
            deadlocks.append(m["deadlock_rate"])
        return {
            "success_rate": float(np.mean(successes)),
            "success_std": float(np.std(successes)),
            "deadlock_rate": float(np.mean(deadlocks)),
            "mean_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps)),
            "per_seed_success": successes,
        }

    for alpha in [0.05, 0.1, 0.2, 0.4]:
        m = run({}, alpha=alpha)
        results["alpha"][str(alpha)] = m
        print(f"alpha={alpha:<6} thành công={m['success_rate']:.2f} ± {m['success_std']:.2f} "
              f"bước TB={m['mean_steps']:.1f} thưởng TB={m['mean_reward']:.1f}")

    for gamma in [0.8, 0.9, 0.95, 0.99]:
        m = run({}, gamma=gamma)
        results["gamma"][str(gamma)] = m
        print(f"gamma={gamma:<6} thành công={m['success_rate']:.2f} ± {m['success_std']:.2f} "
              f"bước TB={m['mean_steps']:.1f} thưởng TB={m['mean_reward']:.1f}")

    reward_configs = {
        "mặc định (bước -0.1, thùng +15, hoàn thành +100)": {},
        "ưu tiên số thùng đúng (bước -0.2, thùng +30, hoàn thành +100)":
            {"step_penalty": -0.2, "single_box_reward": 30.0},
        "phạt kẹt nặng hơn (bước -0.1, thùng +15, hoàn thành +100, kẹt -100)":
            {"step_penalty": -0.1, "single_box_reward": 15.0, "deadlock_penalty": -100.0},
    }
    for label, kwargs in reward_configs.items():
        m = run(kwargs)
        results["reward"][label] = m
        print(f"{label:<58} thành công={m['success_rate']:.2f} "
              f"bước TB={m['mean_steps']:.1f} thưởng TB={m['mean_reward']:.1f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, out_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)
    print(f"\nĐã lưu kết quả: {path}")
    return results


def demo_all_algorithms(setting_key: str = "setting1",
                        episodes: int = DEFAULT_EPISODES,
                        seed: int = 0, show_board: bool = True) -> Dict[str, Any]:
    """
    Chạy thử một episode của cả sáu thuật toán trên một setting.

    In kết quả ra terminal, lưu ảnh đường đi để so sánh và lưu dải hình
    của episode dùng cho báo cáo.
    """
    results: Dict[str, Any] = {}
    env = make_env(setting_key)

    print(f"\n{'=' * 74}")
    print(f"DEMO SÁU THUẬT TOÁN TRÊN {setting_key.upper()}")
    print("=" * 74)

    for algo in ALL_ALGOS:
        agent_env = make_env(setting_key)
        agent = build_agent(algo, agent_env, seed=seed)
        agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
        res = viz.run_episode(agent_env, agent)
        res["algorithm"] = algo
        results[algo] = res

        state = "thành công" if res["solved"] else ("bị kẹt" if res["deadlock"] else "hết bước")
        print(f"{algo:<22} {res['steps']:>3} bước  thưởng {res['reward']:>7.1f}  {state}")

        if show_board:
            print()
            print(agent_env.render())

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, f"fig7_trajectories_{setting_key}.png")
    viz.plot_trajectories(env, results, fig_path,
                          title=f"Đường đi của chính sách tham lam trên {setting_key}")
    print(f"\nĐã lưu ảnh đường đi: {fig_path}")

    strip_path = os.path.join(FIGURES_DIR, f"fig5_episode_{setting_key}.png")
    first = results[ALL_ALGOS[0]]
    viz.save_filmstrip(first["frames"], env, strip_path, n=5,
                       title=f"Một episode của {ALL_ALGOS[0]} trên {setting_key}")
    print(f"Đã lưu dải hình: {strip_path}")
    return results


def _policy_actions(env, agent, boxes: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """
    Hành động mà chính sách chọn tại mỗi ô, khi hai thùng đặt ở vị trí đã cho.
    """
    actions: Dict[Tuple[int, int], int] = {}
    for r in range(env.height):
        for c in range(env.width):
            if (r, c) in env.walls:
                continue
            state_id = env.state_encoder.encode_to_id((r, c), boxes, 0)
            if hasattr(agent, "model"):
                if state_id not in agent.model.state_to_idx:
                    continue
                actions[(r, c)] = agent.get_action(state_id)
            else:
                if state_id not in agent.q_table:
                    continue
                actions[(r, c)] = agent.select_action(state_id, explore=False)
    return actions


def value_field(env, agent, boxes: List[Tuple[int, int]]) -> Dict[int, float]:
    """
    Giá trị của mọi trạng thái khi hai thùng được giữ cố định ở vị trí đã cho.

    Với cấu hình thùng cố định, mỗi ô chỉ ứng với đúng một trạng thái, nên
    giá trị vẽ trên lưới là giá trị của chính trạng thái đó.
    """
    out: Dict[int, float] = {}
    enc = env.state_encoder
    for r in range(env.height):
        for c in range(env.width):
            if (r, c) in env.walls:
                continue
            state_id = enc.encode_to_id((r, c), boxes, 0)
            if hasattr(agent, "model"):
                if state_id not in agent.model.state_to_idx:
                    continue
                out[state_id] = float(agent.V[agent.model.idx(state_id)])
            else:
                if state_id not in agent.q_table:
                    continue
                out[state_id] = float(np.max(agent.q_table[state_id]))
    return out


def make_policy_figures(setting_key: str = "setting1",
                        episodes: int = DEFAULT_EPISODES,
                        lang: str = "vi") -> List[str]:
    """
    Sinh hai hình: nhiệt đồ giá trị và hình mũi tên chính sách.

    Cả hai hình dùng chung một thang màu để so sánh được giữa các thuật toán.
    """
    env = make_env(setting_key)
    made: List[str] = []
    out_dir = figures_dir(lang)
    label = setting_label(setting_key)
    viz.set_language(lang)

    pi = PolicyIteration(env)
    pi.solve()
    ql = QLearningAgent(env, seed=0)
    ql.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
    sa = SARSAAgent(env, seed=0)
    sa.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)

    value_panels = [
        (f"Policy Iteration {TEXT[lang]['optimal']}",
         value_field(env, pi, env.initial_boxes)),
        ("Q-Learning (off-policy)",
         value_field(env, ql, env.initial_boxes)),
        ("SARSA (on-policy)",
         value_field(env, sa, env.initial_boxes)),
    ]
    path = os.path.join(out_dir, f"fig4_value_heatmaps_{setting_key}.png")
    viz.plot_value_heatmaps(env, value_panels, path,
                            title=TEXT[lang]["value"].format(label=label))
    made.append(path)

    action_panels = [
        ("Policy Iteration", _policy_actions(env, pi, env.initial_boxes)),
        ("Q-Learning", _policy_actions(env, ql, env.initial_boxes)),
        ("SARSA", _policy_actions(env, sa, env.initial_boxes)),
    ]
    path = os.path.join(out_dir, f"fig5_policy_{setting_key}.png")
    viz.plot_policy_arrows(env, action_panels, path,
                           title=TEXT[lang]["policy"].format(label=label))
    made.append(path)
    return made


def make_figures(all_results: Optional[Dict[str, Any]] = None,
                 episodes: int = DEFAULT_EPISODES,
                 lang: str = "vi") -> List[str]:
    """Sinh toàn bộ hình dùng cho báo cáo."""
    if all_results is None:
        path = os.path.join(MODELS_DIR, "training_results.json")
        with open(path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    out_dir = figures_dir(lang)
    os.makedirs(out_dir, exist_ok=True)
    viz.set_language(lang)
    made: List[str] = []

    # 1. Đường học của 4 thuật toán TD trên setting 1
    histories = {}
    for algo in TD_ALGOS:
        key = f"setting1|{algo}"
        if key in all_results:
            histories[algo] = all_results[key][0]
    if histories:
        made.append(viz.plot_learning_curves(
            histories, os.path.join(out_dir, "fig1_learning_curves.png"),
            title=TEXT[lang]["learning"]))

    # 2. So sánh tỉ lệ thành công giữa các thuật toán
    summary = aggregate(all_results)
    per_setting: Dict[str, Dict[str, float]] = {}
    for key, m in summary.items():
        setting, algo = key.split("|")
        per_setting.setdefault(setting, {})[algo] = m
    for setting, metrics in per_setting.items():
        label = setting_label(setting)
        made.append(viz.plot_comparison_bar(
            metrics, os.path.join(out_dir, f"fig2_success_{setting}.png"),
            metric="success_rate",
            title=TEXT[lang]["success"].format(label=label),
            order=ALL_ALGOS))

    # 3. Nhiệt đồ giá trị và hình mũi tên chính sách
    made += make_policy_figures("setting1", episodes=episodes, lang=lang)
    made += make_policy_figures("setting3", episodes=episodes, lang=lang)

    # 4. GIF một episode của Q-Learning trên 4 setting
    for i, setting in enumerate(SETTINGS):
        env = make_env(setting["key"])
        agent = QLearningAgent(env, seed=0)
        agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
        frames = viz.record_episode(env, agent)
        gif_path = os.path.join(out_dir, f"fig5_episode_{setting['key']}.gif")
        name = TEXT[lang]["settings"][i]
        viz.save_episode_gif(frames, env, gif_path, title=f"Q-Learning - {name}")
        made.append(gif_path)

    # 4b. Dải hình tĩnh của một episode, dùng cho báo cáo in
    env = make_env("setting1")
    agent = QLearningAgent(env, seed=0)
    agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
    res = viz.run_episode(env, agent)
    path = os.path.join(out_dir, "fig5_episode_setting1.png")
    viz.save_filmstrip(res["frames"], env, path, n=4,
                       title=TEXT[lang]["film"].format(label=setting_label("setting1")))
    made.append(path)

    # 4c. Đường đi của sáu thuật toán trên cùng một bản đồ
    trajectories: Dict[str, Any] = {}
    for algo in ALL_ALGOS:
        agent_env = make_env("setting1")
        agent = build_agent(algo, agent_env, seed=0)
        agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
        trajectories[algo] = viz.run_episode(agent_env, agent)
    path = os.path.join(out_dir, "fig7_trajectories_setting1.png")
    viz.plot_trajectories(env, trajectories, path,
                          title=TEXT[lang]["traj"].format(label=setting_label("setting1")))
    made.append(path)

    # 5. Bản đồ của 4 setting
    envs = [make_env(s["key"]) for s in SETTINGS]
    names = TEXT[lang]["settings"]
    made.append(viz.plot_maps_grid(
        envs, names, os.path.join(out_dir, "fig6_maps.png")))
    for setting, env, name in zip(SETTINGS, envs, names):
        path = os.path.join(out_dir, f"fig6_map_{setting['key']}.png")
        fig, ax = viz.board_figure(env, title=name)
        fig.savefig(path, dpi=150, facecolor="#fbf8f2")
        viz.plt.close(fig)
        made.append(path)

    print(f"Đã tạo {len(made)} hình trong {out_dir}")
    return made


def demo_episode(algo: str = "Q-Learning", setting_key: str = "setting1",
                 episodes: int = DEFAULT_EPISODES, show_gif: bool = True) -> None:
    """Chạy một episode bằng chính sách đã học và in ra terminal."""
    env = make_env(setting_key)
    agent = build_agent(algo, env, seed=0)
    agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)

    state_id = env.reset()
    print(f"\n{algo} trên {setting_key}\n")
    print(env.render())
    terminated = truncated = False
    info: Dict[str, Any] = {}
    while not (terminated or truncated):
        action = agent.select_action(state_id, explore=False)
        state_id, reward, terminated, truncated, info = env.step(action)
        print(f"\nHành động {action}  phần thưởng {reward:+.1f}")
        print(env.render())

    print(f"\nKết quả: {'thành công' if info.get('is_solved') else 'thất bại'}"
          f" sau {env.steps} bước")

    if show_gif:
        env2 = make_env(setting_key)
        agent2 = build_agent(algo, env2, seed=0)
        agent2.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
        frames = viz.record_episode(env2, agent2)
        path = os.path.join(
            FIGURES_DIR,
            f"demo_{algo.lower().replace(' ', '_')}_{setting_key}.gif")
        viz.save_episode_gif(frames, env2, path, title=f"{algo} - {setting_key}")
        print(f"Đã lưu GIF: {path}")


# --------------------------------------------------------------------------
# Thực nghiệm đầy đủ: DP + TD nhiều seed + ablation + độ nhạy tham số
# --------------------------------------------------------------------------
def map_without_rescue(setting_key: str) -> str:
    """Nội dung bản đồ của một setting, bỏ ô cứu hộ (dùng cho ablation)."""
    with open(map_path(setting_key), "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    return "\n".join(line.replace("R", ".") for line in lines)


def run_dp_study(settings: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Chạy hai thuật toán quy hoạch động trên tất cả setting (không cần seed)."""
    from sokoban.agents import PolicyIteration, ValueIteration

    settings = settings or [s["key"] for s in SETTINGS]
    out: Dict[str, Any] = {}
    for key in settings:
        env = make_env(key)
        pi = PolicyIteration(env)
        t0 = time.time()
        r_pi = pi.solve()
        t_pi = time.time() - t0

        env2 = make_env(key)
        vi = ValueIteration(env2)
        t0 = time.time()
        r_vi = vi.solve()
        t_vi = time.time() - t0

        m_pi = pi.evaluate(num_episodes=20)
        m_vi = vi.evaluate(num_episodes=20)
        s0 = env.state_encoder.encode_to_id(env.initial_player, env.initial_boxes, 0)
        same = sum(1 for s in vi.model.state_ids
                   if vi.get_action(s) == pi.get_action(s))
        out[key] = {
            "policy_iteration": {**m_pi, "iterations": r_pi["iterations"],
                                 "converged": r_pi["converged"], "time": t_pi,
                                 "V_star_s0": float(pi.V[pi.model.idx(s0)])},
            "value_iteration": {**m_vi, "iterations": r_vi["iterations"],
                                "converged": r_vi["converged"], "time": t_vi,
                                "V_star_s0": float(vi.V[vi.model.idx(s0)])},
            "n_states": pi.model.n_states,
            "n_terminal_states": len(pi.model.terminal_states),
            "max_abs_value_difference": float(np.max(np.abs(vi.V - pi.V))),
            "same_policy_states": same,
        }
        print(f"  DP {key}: PI {r_pi['iterations']} vong ({t_pi:.2f}s), "
              f"VI {r_vi['iterations']} vong ({t_vi:.2f}s), "
              f"V*(s0)={pi.V[pi.model.idx(s0)]:.2f}, "
              f"max|dV|={out[key]['max_abs_value_difference']:.1e}")
    return out


def run_epsilon_sweep(setting_key: str = "setting3", seeds: int = 30,
                     episodes: int = 1000) -> Dict[str, Any]:
    """
    Quet he so giam epsilon de kiem tra khoang cach giua chinh sach on-policy
    va off-policy co co lai khi epsilon nho khong.
    """
    out: Dict[str, Any] = {"setting": setting_key, "seeds": seeds,
                           "episodes": episodes, "decays": {}}
    for decay in [0.9, 0.99, 0.995, 0.9992]:
        res = run_seeds(TD_ALGOS, [setting_key], seeds, episodes=episodes,
                       agent_overrides={"epsilon_decay": decay}, progress=False)
        summary = summarize_seeds(res)
        out["decays"][str(decay)] = {
            algo: summary[f"{setting_key}|{algo}"] for algo in TD_ALGOS
            if f"{setting_key}|{algo}" in summary
        }
        ql = summary.get(f"{setting_key}|Q-Learning", {})
        sa = summary.get(f"{setting_key}|SARSA", {})
        out["decays"][str(decay)]["epsilon_final"] = float(decay ** episodes)
        print(f"  epsilon_decay={decay}: epsilon cuoi = {decay ** episodes:.4f} | "
              f"QL {ql.get('success_rate', 0):.2f}/{ql.get('mean_reward', 0):.1f} "
              f"({ql.get('mean_steps', 0):.1f} buoc) | "
              f"SARSA {sa.get('success_rate', 0):.2f}/{sa.get('mean_reward', 0):.1f} "
              f"({sa.get('mean_steps', 0):.1f} buoc)")
    return out


def run_sensitivity_seeds(episodes: int = 600, seeds: int = 30,
                          setting_key: str = "setting3") -> Dict[str, Any]:
    """Phân tích độ nhạy tham số với nhiều seed."""
    out: Dict[str, Any] = {"setting": setting_key, "seeds": seeds, "episodes": episodes}

    for alpha in [0.05, 0.1, 0.2, 0.4]:
        res = run_seeds(["Q-Learning"], [setting_key], seeds, episodes=episodes,
                        agent_overrides={"alpha": alpha}, progress=False)
        out.setdefault("alpha", {})[str(alpha)] = \
            summarize_seeds(res)[f"{setting_key}|Q-Learning"]

    for gamma in [0.8, 0.9, 0.95, 0.99]:
        res = run_seeds(["Q-Learning"], [setting_key], seeds, episodes=episodes,
                        agent_overrides={"gamma": gamma}, progress=False)
        out.setdefault("gamma", {})[str(gamma)] = \
            summarize_seeds(res)[f"{setting_key}|Q-Learning"]

    reward_configs = {
        "mac dinh (buoc -0.1, thung +15, hoan thanh +100, cuu ho +10)": {},
        "uu tien so thung dung (buoc -0.2, thung +30)":
            {"step_penalty": -0.2, "single_box_reward": 30.0},
        "phat ket nang hon (ket -100)": {"deadlock_penalty": -100.0},
        "khong thuong cuu ho (cuu ho 0)": {"rescue_bonus": 0.0},
        "thuong cuu ho am (cuu ho -10)": {"rescue_bonus": -10.0},
    }
    for label, kwargs in reward_configs.items():
        res = run_seeds(["Q-Learning"], [setting_key], seeds, episodes=episodes,
                        env_overrides=kwargs, progress=False)
        out.setdefault("reward", {})[label] = \
            summarize_seeds(res)[f"{setting_key}|Q-Learning"]
    return out


def run_full_study(episodes: int = 1000, seeds: int = 30, workers: int = 0,
                   out_file: str = "study_results.json",
                   skip_sensitivity: bool = False,
                   skip_epsilon_sweep: bool = False) -> Dict[str, Any]:
    """
    Chạy toàn bộ thực nghiệm và lưu kết quả ra JSON.

    Gồm hai thuật toán quy hoạch động, bốn thuật toán TD/Monte Carlo với
    nhiều seed, thí nghiệm ablation có và không có ô cứu hộ, và phân tích độ
    nhạy tham số.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    study: Dict[str, Any] = {"config": {
        "episodes": episodes, "seeds": seeds,
        "algorithms": ALL_ALGOS, "settings": [s["key"] for s in SETTINGS]}}
    path = os.path.join(MODELS_DIR, out_file)

    def save() -> None:
        """Ghi kết quả từng phần, để một sự cố ở phần sau không làm mất phần trước."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(study, f, ensure_ascii=False, indent=2, default=json_default)

    print("[1/4] Hai thuat toan quy hoach dong")
    study["dp"] = run_dp_study()
    save()

    print(f"[2/4] Bon thuat toan TD/Monte Carlo voi {seeds} seed")
    td = run_seeds(TD_ALGOS, [s["key"] for s in SETTINGS], seeds,
                   episodes=episodes, workers=workers)
    study["td"] = td
    study["td_summary"] = summarize_seeds(td)
    save()

    print("[3/4] Ablation: co va khong co o cuu ho tren ban do setting3")
    with_rescue = run_seeds(TD_ALGOS, ["setting3"], seeds, episodes=episodes,
                            workers=workers, progress=False)
    without = run_seeds(TD_ALGOS, ["setting3"], seeds, episodes=episodes,
                        workers=workers, map_string=map_without_rescue("setting3"),
                        progress=False)
    study["ablation"] = {
        "with_rescue": summarize_seeds(with_rescue),
        "without_rescue": summarize_seeds(without),
    }
    save()

    if not skip_sensitivity:
        print("[4/5] Phan tich do nhay tham so")
        study["sensitivity"] = run_sensitivity_seeds(
            episodes=min(episodes, 600), seeds=min(seeds, 30))
        save()

    if not skip_epsilon_sweep:
        print("[5/5] Quet he so giam epsilon")
        study["epsilon_sweep"] = run_epsilon_sweep(
            setting_key="setting3", seeds=seeds, episodes=episodes)
        save()

    print(f"\nDa luu ket qua: {path}")
    return study

# --------------------------------------------------------------------------
# Sinh toàn bộ hình từ kết quả thực nghiệm
# --------------------------------------------------------------------------
def make_study_figures(study: Dict[str, Any], lang: str = "vi",
                       episodes: int = 1000) -> List[str]:
    """
    Sinh các hình dùng cho báo cáo từ kết quả thực nghiệm đã có.

    Hình nhiệt đồ giá trị và hình mũi tên chính sách cần huấn luyện lại một số
    tác tử, các hình còn lại chỉ dùng dữ liệu đã tính.
    """
    out_dir = figures_dir(lang)
    os.makedirs(out_dir, exist_ok=True)
    viz.set_language(lang)
    made: List[str] = []
    label = lambda k: setting_label(k)

    # 1. Bốn bản đồ
    envs = [make_env(s["key"]) for s in SETTINGS]
    names = TEXT[lang]["settings"]
    made.append(viz.plot_maps_grid(envs, names,
                                   os.path.join(out_dir, "fig1_maps.png"),
                                   title=TEXT[lang]["maps"]))

    # 2. Đường học nhiều seed cho từng setting
    for setting in [s["key"] for s in SETTINGS]:
        histories = {}
        for algo in TD_ALGOS:
            key = f"{setting}|{algo}"
            if key in study.get("td", {}):
                histories[algo] = study["td"][key]
        if not histories:
            continue
        made.append(viz.plot_learning_curves_multi(
            histories, os.path.join(out_dir, f"fig2_learning_{setting}.png"),
            title=TEXT[lang]["learning"].format(label=label(setting))))

    # 3. Biểu đồ cột nhóm cho cả bốn setting, có thanh sai số
    made.append(viz.plot_grouped_bars(
        study["td_summary"], os.path.join(out_dir, "fig3_success_all.png"),
        metric="success_rate",
        title=TEXT[lang]["success"].format(label="all settings"),
        order=TD_ALGOS))

    # 4. Nhiệt đồ giá trị và mũi tên chính sách
    made += make_policy_figures("setting1", episodes=episodes, lang=lang)
    made += make_policy_figures("setting3", episodes=episodes, lang=lang)

    # 5. Dải hình một episode
    env = make_env("setting1")
    agent = QLearningAgent(env, seed=0)
    agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
    res = viz.run_episode(env, agent)
    made.append(viz.save_filmstrip(
        res["frames"], env, os.path.join(out_dir, "fig6_episode_setting1.png"),
        n=4, title=TEXT[lang]["film"].format(label=label("setting1"))))

    # 6. Đường đi của sáu thuật toán
    trajectories: Dict[str, Any] = {}
    for algo in ALL_ALGOS:
        agent_env = make_env("setting1")
        ag = build_agent(algo, agent_env, seed=0)
        ag.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
        trajectories[algo] = viz.run_episode(agent_env, ag)
    made.append(viz.plot_trajectories(
        env, trajectories, os.path.join(out_dir, "fig7_trajectories_setting1.png"),
        title=TEXT[lang]["traj"].format(label=label("setting1"))))

    # 7. Ablation có và không có ô cứu hộ
    if "ablation" in study:
        made.append(viz.plot_ablation(
            study["ablation"]["with_rescue"], study["ablation"]["without_rescue"],
            os.path.join(out_dir, "fig8_ablation.png"),
            title=TEXT[lang].get("ablation", "Rescue tile ablation")))

    print(f"Da tao {len(made)} hinh trong {out_dir}")
    return made


def write_summary(study: Dict[str, Any], path: str) -> str:
    """
    Ghi tóm tắt toàn bộ số liệu ra tệp văn bản để đối chiếu với báo cáo.
    """
    lines: List[str] = []
    cfg = study["config"]
    lines.append("=" * 90)
    lines.append(f"SO LIEU THUC NGHIEM - episodes={cfg['episodes']} seeds={cfg['seeds']}")
    lines.append("=" * 90)

    lines.append("")
    lines.append("### A. HAI THUAT TOAN QUY HOACH DONG")
    for setting, d in study["dp"].items():
        pi, vi = d["policy_iteration"], d["value_iteration"]
        lines.append(f"{setting}: states={d['n_states']} terminal={d['n_terminal_states']}")
        lines.append(f"  PI: iters={pi['iterations']} conv={pi['converged']} "
                     f"time={pi['time']:.2f}s V*(s0)={pi['V_star_s0']:.4f} "
                     f"success={pi['success_rate']:.3f} ret={pi['mean_reward']:.2f} "
                     f"disc={pi['mean_discounted_reward']:.4f} steps={pi['mean_steps']:.2f}")
        lines.append(f"  VI: iters={vi['iterations']} conv={vi['converged']} "
                     f"time={vi['time']:.2f}s V*(s0)={vi['V_star_s0']:.4f} "
                     f"success={vi['success_rate']:.3f} ret={vi['mean_reward']:.2f} "
                     f"disc={vi['mean_discounted_reward']:.4f} steps={vi['mean_steps']:.2f}")
        lines.append(f"  max|V_VI-V_PI|={d['max_abs_value_difference']:.3e} "
                     f"same_policy={d['same_policy_states']}/{d['n_states']}")

    lines.append("")
    lines.append("### B. BON THUAT TOAN TD / MONTE CARLO")
    header = (f"{'setting':<10}{'algo':<22}{'succ':>7}{'std':>7}{'dead':>7}{'tout':>7}"
              f"{'return':>10}{'std':>8}{'disc':>10}{'steps':>8}{'act':>7}{'use':>6}")
    lines.append(header)
    for key, m in study["td_summary"].items():
        setting, algo = key.split("|")
        lines.append(f"{setting:<10}{algo:<22}{m['success_rate']:>7.3f}"
                     f"{m['success_rate_std']:>7.3f}{m['deadlock_rate']:>7.3f}"
                     f"{m['timeout_rate']:>7.3f}{m['mean_reward']:>10.2f}"
                     f"{m['mean_reward_std']:>8.2f}{m['mean_discounted_reward']:>10.3f}"
                     f"{m['mean_steps']:>8.2f}{m['mean_rescue_activations']:>7.2f}"
                     f"{m['mean_rescue_uses']:>6.2f}")

    lines.append("")
    lines.append("### C. ABLATION O CUU HO (setting3)")
    for algo in TD_ALGOS:
        a = study["ablation"]["with_rescue"].get(f"setting3|{algo}")
        b = study["ablation"]["without_rescue"].get(f"setting3|{algo}")
        if not a or not b:
            continue
        lines.append(f"{algo:<22} CO: succ={a['success_rate']:.3f}+-{a['success_rate_std']:.3f} "
                     f"ret={a['mean_reward']:.2f} steps={a['mean_steps']:.2f} "
                     f"act={a['mean_rescue_activations']:.2f} use={a['mean_rescue_uses']:.2f} "
                     f"tout={a['timeout_rate']:.3f}")
        lines.append(f"{'':<22} KHONG: succ={b['success_rate']:.3f}+-{b['success_rate_std']:.3f} "
                     f"ret={b['mean_reward']:.2f} steps={b['mean_steps']:.2f} "
                     f"act={b['mean_rescue_activations']:.2f} use={b['mean_rescue_uses']:.2f} "
                     f"tout={b['timeout_rate']:.3f}")

    sens = study.get("sensitivity")
    if sens:
        lines.append("")
        lines.append("### D. DO NHAY THAM SO")
        lines.append(f"setting={sens['setting']} seeds={sens['seeds']} episodes={sens['episodes']}")
        for group in ["alpha", "gamma", "reward"]:
            if group not in sens:
                continue
            lines.append(f"-- {group}")
            for k, m in sens[group].items():
                lines.append(f"   {k:<58} succ={m['success_rate']:.3f}+-{m['success_rate_std']:.3f} "
                             f"ret={m['mean_reward']:.2f} steps={m['mean_steps']:.2f} "
                             f"act={m['mean_rescue_activations']:.2f} use={m['mean_rescue_uses']:.2f}")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
