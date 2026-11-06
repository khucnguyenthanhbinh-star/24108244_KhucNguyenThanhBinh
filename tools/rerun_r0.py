"""Rerun the whole study with rescue rewards set to 0 (main configuration).

Saves saved_models/study_r0.json with:
  td, td_summary, ablation, sensitivity, dp
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sokoban.experiments import (TD_ALGOS, make_env, map_without_rescue,
                                 run_seeds, summarize_seeds)
from sokoban.agents import PolicyIteration, ValueIteration

SETTINGS = ["setting1", "setting2", "setting3", "setting4"]
WORKERS = 4
R0 = {"rescue_bonus": 0.0}


def main():
    out = {"config": {"episodes": 1000, "seeds": 30, "workers": WORKERS,
                      "rescue_bonus": 0.0}}

    print("== main TD, r=0 ==", flush=True)
    td = run_seeds(TD_ALGOS, SETTINGS, 30, episodes=1000, workers=WORKERS,
                   env_overrides=R0)
    out["td"] = td
    out["td_summary"] = summarize_seeds(td)

    print("== ablation, r=0 ==", flush=True)
    with_r = run_seeds(TD_ALGOS, ["setting3"], 30, episodes=1000,
                       workers=WORKERS, env_overrides=R0)
    without_r = run_seeds(TD_ALGOS, ["setting3"], 30, episodes=1000,
                          workers=WORKERS, env_overrides=R0,
                          map_string=map_without_rescue("setting3"))
    out["ablation"] = {"with_rescue": summarize_seeds(with_r),
                       "without_rescue": summarize_seeds(without_r)}
    out["ablation_raw"] = {"with_rescue": with_r, "without_rescue": without_r}

    print("== sensitivity, r=0 ==", flush=True)
    sens = {"alpha": {}, "gamma": {}, "reward": {}}
    for a in (0.05, 0.1, 0.2, 0.4):
        r = run_seeds(["Q-Learning"], ["setting3"], 3, episodes=600,
                      workers=WORKERS, env_overrides=R0,
                      agent_overrides={"alpha": a})
        sens["alpha"][str(a)] = summarize_seeds(r)["setting3|Q-Learning"]
    for g in (0.8, 0.9, 0.95, 0.99):
        r = run_seeds(["Q-Learning"], ["setting3"], 3, episodes=600,
                      workers=WORKERS, env_overrides=R0,
                      agent_overrides={"gamma": g})
        sens["gamma"][str(g)] = summarize_seeds(r)["setting3|Q-Learning"]
    rewards = {
        "r=0": (R0, {}),
        "r=+10": ({"rescue_bonus": 10.0}, {}),
        "r=-10": ({"rescue_bonus": -10.0}, {}),
        "box-focused": ({"rescue_bonus": 0.0, "step_penalty": -0.2,
                         "single_box_reward": 30.0}, {}),
        "deadlock-heavy": ({"rescue_bonus": 0.0, "deadlock_penalty": -100.0}, {}),
    }
    for name, (env_kw, ag_kw) in rewards.items():
        r = run_seeds(["Q-Learning"], ["setting3"], 3, episodes=600,
                      workers=WORKERS, env_overrides=env_kw,
                      agent_overrides=ag_kw)
        sens["reward"][name] = summarize_seeds(r)["setting3|Q-Learning"]
    out["sensitivity"] = sens

    print("== DP, r=0 ==", flush=True)
    dp = {}
    for key in SETTINGS:
        for gamma in (0.95, 0.99):
            for cls in (ValueIteration, PolicyIteration):
                env = make_env(key, rescue_bonus=0.0)
                ag = cls(env, gamma=gamma)
                res = ag.solve()
                sid = env.reset()
                v0 = ag.values()[sid]
                met = ag.evaluate(50)
                dp[f"{key}|{cls.name}|{gamma}"] = {
                    "algorithm": cls.name, "setting": key, "gamma": gamma,
                    "iterations": res.get("iterations"),
                    "training_time": res.get("training_time"),
                    "v_start": v0, "success_rate": met["success_rate"],
                    "mean_reward": met["mean_reward"],
                    "mean_steps": met["mean_steps"],
                    "mean_rescue_uses": met["mean_rescue_uses"],
                }
                print(key, cls.name, gamma, round(v0, 3),
                      met["mean_steps"], flush=True)
    out["dp"] = dp

    path = os.path.join(ROOT, "saved_models", "study_r0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
