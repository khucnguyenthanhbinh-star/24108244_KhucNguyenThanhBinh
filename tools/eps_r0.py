"""Epsilon sweep at r_rescue = 0 (Setting 3, 30 seeds, 1000 episodes)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sokoban.experiments import TD_ALGOS, run_seeds, summarize_seeds

W = 4
R0 = {"rescue_bonus": 0.0}


def main():
    out = {"setting": "setting3", "seeds": 30, "episodes": 1000, "decays": {}}
    for decay in (0.9, 0.99, 0.995, 0.9992):
        res = run_seeds(TD_ALGOS, ["setting3"], 30, episodes=1000,
                        workers=W, env_overrides=R0,
                        agent_overrides={"epsilon_decay": decay},
                        progress=False)
        summary = summarize_seeds(res)
        out["decays"][str(decay)] = {a: summary[f"setting3|{a}"]
                                     for a in TD_ALGOS}
        out["decays"][str(decay)]["epsilon_final"] = decay ** 1000
        print(decay, {a: (round(summary[f"setting3|{a}"]["success_rate"], 2),
                          round(summary[f"setting3|{a}"]["mean_reward"], 1))
                      for a in TD_ALGOS}, flush=True)
    path = os.path.join(ROOT, "saved_models", "eps_r0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
