"""Sensitivity analysis (alpha, gamma, reward) at r_rescue = 0, 30 seeds."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sokoban.experiments import run_seeds, summarize_seeds

W = 4
R0 = {"rescue_bonus": 0.0}


def main():
    out = {"alpha": {}, "gamma": {}, "reward": {}}
    for a in (0.05, 0.1, 0.2, 0.4):
        r = run_seeds(["Q-Learning"], ["setting3"], 30, episodes=600,
                      workers=W, env_overrides=R0, agent_overrides={"alpha": a})
        out["alpha"][str(a)] = summarize_seeds(r)["setting3|Q-Learning"]
        print("alpha", a, out["alpha"][str(a)]["mean_reward"], flush=True)
    for g in (0.8, 0.9, 0.95, 0.99):
        r = run_seeds(["Q-Learning"], ["setting3"], 30, episodes=600,
                      workers=W, env_overrides=R0, agent_overrides={"gamma": g})
        out["gamma"][str(g)] = summarize_seeds(r)["setting3|Q-Learning"]
        print("gamma", g, out["gamma"][str(g)]["mean_reward"], flush=True)
    configs = {
        "r=0": (R0, {}),
        "r=+10": ({"rescue_bonus": 10.0}, {}),
        "r=-10": ({"rescue_bonus": -10.0}, {}),
        "box-focused": ({"rescue_bonus": 0.0, "step_penalty": -0.2,
                         "single_box_reward": 30.0}, {}),
        "deadlock-heavy": ({"rescue_bonus": 0.0, "deadlock_penalty": -100.0}, {}),
    }
    for name, (env_kw, ag_kw) in configs.items():
        r = run_seeds(["Q-Learning"], ["setting3"], 30, episodes=600,
                      workers=W, env_overrides=env_kw, agent_overrides=ag_kw)
        out["reward"][name] = summarize_seeds(r)["setting3|Q-Learning"]
        print("reward", name, out["reward"][name]["mean_reward"], flush=True)
    path = os.path.join(ROOT, "saved_models", "sens_r0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
