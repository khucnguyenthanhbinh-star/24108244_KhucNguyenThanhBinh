"""Probe: deadlock frequency and rescue-tile effect on training.

Modes:
  audit : random roll-outs on all settings, count deadlocks / tile visits
  run   : train TD agents under 3 conditions (no tile, r=0, r=+10)

Usage (from repo root):
  python tools/rescue_probe.py --mode audit
  python tools/rescue_probe.py --mode run --algos Q-Learning --seeds 30
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sokoban.env import SokobanTwoBoxesEnv
from sokoban.experiments import build_agent, make_env, map_path

MAPS = ["setting1", "setting2", "setting3", "setting4"]


def map_without_rescue(setting_key):
    with open(map_path(setting_key), "r", encoding="utf-8") as f:
        return "\n".join(line.replace("R", ".") for line in f.read().split("\n"))


def make_cond_env(cond, setting_key):
    if cond == "no_tile":
        return SokobanTwoBoxesEnv(map_string=map_without_rescue(setting_key))
    if cond == "r0":
        return make_env(setting_key, rescue_bonus=0.0)
    if cond == "r10":
        return make_env(setting_key, rescue_bonus=10.0)
    raise ValueError(cond)


class LoggingEnv:
    """Delegates to the real env and records one record per episode."""

    def __init__(self, env):
        self._env = env
        self.records = []
        self._cur = None

    def reset(self):
        if self._cur is not None:
            self.records.append(self._cur)
        self._cur = dict(solved=0, deadlock=0, timeout=0, steps=0,
                         ret=0.0, resc_act=0, resc_use=0, visited_tile=0)
        return self._env.reset()

    def step(self, action):
        sid, r, term, trunc, info = self._env.step(action)
        c = self._cur
        c["steps"] += 1
        c["ret"] += float(r)
        if info.get("rescue_state") == 1:
            c["visited_tile"] = 1
        if term or trunc:
            c["resc_act"] = info.get("rescue_activations", 0)
            c["resc_use"] = info.get("rescue_uses", 0)
            outcome = info.get("outcome", "timeout")
            c[outcome] = c.get(outcome, 0) + 1
            self.records.append(c)
            self._cur = None
        return sid, r, term, trunc, info

    def __getattr__(self, name):
        return getattr(self._env, name)


def audit(args):
    out = {}
    rng = np.random.default_rng(0)
    for key in MAPS:
        for bonus in (0.0, 10.0):
            env = make_env(key, rescue_bonus=bonus)
            stats = dict(episodes=args.episodes, deadlock=0, solved=0, timeout=0,
                         rescues=0, activations=0, visited_before=0, steps=0)
            for _ in range(args.episodes):
                env.reset()
                done = False
                visited = False
                while not done:
                    _, _, term, trunc, info = env.step(int(rng.integers(4)))
                    if info.get("rescue_state") == 1:
                        visited = True
                    done = term or trunc
                stats[info["outcome"]] = stats.get(info["outcome"], 0) + 1
                stats["rescues"] += info.get("rescue_uses", 0)
                stats["activations"] += info.get("rescue_activations", 0)
                stats["visited_before"] += int(visited)
                stats["steps"] += info.get("steps", 0)
            stats["deadlock_rate"] = stats.get("deadlock", 0) / max(1, stats["episodes"])
            stats["visited_rate"] = stats["visited_before"] / max(1, stats["episodes"])
            out[f"{key}|bonus{bonus:g}"] = stats
            print(key, bonus, json.dumps(stats))
    return out


def _one_run(task):
    cond, algo, seed, episodes, setting = task
    env = LoggingEnv(make_cond_env(cond, setting))
    agent = build_agent(algo, env, seed=seed)
    t0 = time.time()
    hist = agent.train(num_episodes=episodes, eval_interval=episodes, eval_episodes=5)
    rec = env.records[:episodes]  # drop the eval episodes appended by train()
    return {
        "cond": cond, "algo": algo, "seed": seed, "setting": setting,
        "wall": time.time() - t0,
        "solved": [r["solved"] for r in rec],
        "deadlock": [r["deadlock"] for r in rec],
        "timeout": [r["timeout"] for r in rec],
        "steps": [r["steps"] for r in rec],
        "ret": [r["ret"] for r in rec],
        "resc_act": [r["resc_act"] for r in rec],
        "resc_use": [r["resc_use"] for r in rec],
        "visited": [r["visited_tile"] for r in rec],
        "episode_rewards": hist["episode_rewards"],
        "eval_success": hist["success_rates"][-1] if hist["success_rates"] else None,
    }


def run(args):
    conds = ["no_tile", "r0", "r10"]
    tasks = [(c, a, s, args.episodes, args.setting)
             for c in conds for a in args.algos for s in range(args.seeds)]
    print(f"{len(tasks)} runs on {args.workers} processes", flush=True)
    t0 = time.time()
    if args.workers == 1:
        rows = [_one_run(t) for t in tasks]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(_one_run, tasks, chunksize=1)
    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return rows


def summarize(rows, bins=50, threshold=0.9, window=50):
    summary = {}
    groups = {}
    for r in rows:
        groups.setdefault((r["cond"], r["algo"]), []).append(r)
    for key, runs in groups.items():
        n = min(len(r["solved"]) for r in runs)
        solved = np.array([r["solved"][:n] for r in runs], dtype=float)
        dead = np.array([r["deadlock"][:n] for r in runs], dtype=float)
        tout = np.array([r["timeout"][:n] for r in runs], dtype=float)
        ret = np.array([r["ret"][:n] for r in runs], dtype=float)
        act = np.array([r["resc_act"][:n] for r in runs], dtype=float)
        use = np.array([r["resc_use"][:n] for r in runs], dtype=float)
        visited = np.array([r["visited"][:n] for r in runs], dtype=float)
        nb = n // bins
        def binstats(x):
            return (x[:, :nb * bins].reshape(len(runs), nb, bins).mean(axis=2)
                    .mean(axis=0).tolist())
        # episodes to threshold: first episode where forward window mean >= threshold
        e2t = []
        for row in solved:
            idx = None
            for e in range(0, max(1, n - window)):
                if row[e:e + window].mean() >= threshold:
                    idx = e
                    break
            e2t.append(idx if idx is not None else n)
        summary[f"{key[0]}|{key[1]}"] = {
            "seeds": len(runs),
            "success_rate": float(solved.mean()),
            "success_bin": binstats(solved),
            "deadlock_bin": binstats(dead),
            "timeout_bin": binstats(tout),
            "return_bin": binstats(ret),
            "rescue_act_bin": binstats(act),
            "rescue_use_bin": binstats(use),
            "visited_bin": binstats(visited),
            "episodes_to_threshold_mean": float(np.mean(e2t)),
            "episodes_to_threshold_median": float(np.median(e2t)),
            "deadlock_episodes": float(dead.mean()),
            "timeout_episodes": float(tout.mean()),
            "bin_size": bins,
            "raw_episodes": n,
        }
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"no_tile": "#666666", "r0": "#1f77b4", "r10": "#d62728"}
    labels = {"no_tile": "no tile", "r0": "tile, r=0", "r10": "tile, r=+10"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
    for key, s in summary.items():
        cond, algo = key.split("|")
        if algo != "Q-Learning":
            continue
        x = np.arange(len(s["success_bin"])) * s["bin_size"] + s["bin_size"]
        axes[0].plot(x, s["success_bin"], color=colors[cond], label=labels[cond], lw=1.4)
        axes[1].plot(x, s["deadlock_bin"], color=colors[cond], label=labels[cond], lw=1.4)
    axes[0].set_ylabel("training success rate")
    axes[1].set_ylabel("training deadlock rate")
    for ax in axes:
        ax.set_xlabel("episode")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "run"], required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--algos", nargs="+", default=["Q-Learning"])
    ap.add_argument("--setting", default="setting3")
    ap.add_argument("--out", default="saved_models/rescue_probe.json")
    ap.add_argument("--fig", default="report/figures/fig9_rescue_training.png")
    args = ap.parse_args()
    args.workers = args.workers or os.cpu_count() or 1

    if args.mode == "audit":
        res = audit(args)
    else:
        rows = run(args)
        res = {"config": vars(args), "rows": rows,
               "summary": summarize(rows)}
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        plot(res["summary"], args.fig)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
