"""Rerun main + no-tile ablation on all four maps (all now have a rescue tile)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sokoban.experiments import (TD_ALGOS, make_env, map_without_rescue,
                                 run_seeds, summarize_seeds)
from sokoban.agents import PolicyIteration, ValueIteration

SETTINGS = ["setting1", "setting2", "setting3", "setting4"]
W = 4
R0 = {"rescue_bonus": 0.0}


def main():
    out = {"config": {"episodes": 1000, "seeds": 30, "rescue_bonus": 0.0}}
    print("main, all maps with tile", flush=True)
    td = run_seeds(TD_ALGOS, SETTINGS, 30, episodes=1000, workers=W,
                   env_overrides=R0)
    out["td"] = td
    out["td_summary"] = summarize_seeds(td)

    print("ablation: without tile", flush=True)
    abl = {}
    for key in SETTINGS:
        res = run_seeds(TD_ALGOS, [key], 30, episodes=1000, workers=W,
                        env_overrides=R0, map_string=map_without_rescue(key))
        abl[key] = summarize_seeds(res)
    out["ablation_without"] = abl
    out["ablation_with"] = {k: out["td_summary"][k] for k in out["td_summary"]}

    print("DP", flush=True)
    dp = {}
    for key in SETTINGS:
        for gamma in (0.95, 0.99):
            for cls in (ValueIteration, PolicyIteration):
                env = make_env(key, rescue_bonus=0.0)
                ag = cls(env, gamma=gamma)
                res = ag.solve()
                sid = env.reset()
                met = ag.evaluate(50)
                dp[f"{key}|{cls.name}|{gamma}"] = {
                    "algorithm": cls.name, "setting": key, "gamma": gamma,
                    "iterations": res.get("iterations"),
                    "training_time": res.get("training_time"),
                    "v_start": ag.values()[sid],
                    "success_rate": met["success_rate"],
                    "mean_reward": met["mean_reward"],
                    "mean_steps": met["mean_steps"],
                    "mean_rescue_uses": met["mean_rescue_uses"],
                }
    out["dp"] = dp
    path = os.path.join(ROOT, "saved_models", "study_all.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
