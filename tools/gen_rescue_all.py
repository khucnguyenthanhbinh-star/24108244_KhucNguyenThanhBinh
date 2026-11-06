"""Merge rescue probe runs for all TD algorithms, emit table + figure."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
ALGOS = ["Q-Learning", "SARSA", "Expected SARSA", "Monte Carlo Control"]
CONDS = ["no_tile", "r0", "r10"]
CLAB = {"no_tile": "không ô", "r0": "có ô, r=0", "r10": "có ô, r=+10"}
COL = {"no_tile": "#666666", "r0": "#1f77b4", "r10": "#d62728"}
BIN = 50
THR = 0.9
WIN = 50


def load():
    rows = []
    for name in ("rescue_probe.json", "rescue_probe_rest.json"):
        p = os.path.join(ROOT, "saved_models", name)
        if os.path.exists(p):
            rows += json.load(open(p, encoding="utf-8"))["rows"]
    return rows


def stats(rows):
    out = {}
    for a in ALGOS:
        for c in CONDS:
            rs = [r for r in rows if r["algo"] == a and r["cond"] == c]
            if not rs:
                continue
            n = min(len(r["solved"]) for r in rs)
            solved = np.array([r["solved"][:n] for r in rs], float)
            dead = np.array([r["deadlock"][:n] for r in rs], float)
            e2t = []
            for row in solved:
                idx = n
                for e in range(0, max(1, n - WIN)):
                    if row[e:e + WIN].mean() >= THR:
                        idx = e
                        break
                e2t.append(idx)
            e2t = np.array(e2t, float)
            nb = n // BIN
            agg = solved[:, :nb * BIN].reshape(len(rs), nb, BIN).mean(axis=2)
            dag = dead[:, :nb * BIN].reshape(len(rs), nb, BIN).mean(axis=2)
            out[(a, c)] = {
                "seeds": len(rs),
                "success": solved.mean(),
                "success_std": solved.mean(axis=1).std(ddof=1),
                "e2t": e2t.mean(), "e2t_std": e2t.std(ddof=1),
                "deadlock": dead.mean(),
                "deadlock_std": dead.mean(axis=1).std(ddof=1),
                "succ_bin": agg.mean(axis=0), "dead_bin": dag.mean(axis=0),
            }
    return out


def table(st):
    lines = [r"\begin{table*}[t]",
             r"\caption{Ô cứu hộ và tốc độ học trên Setting 3 cho bốn thuật toán TD",
             r"(30 seed, 1000 episode, $r_{\text{rescue}}=0$ ở cấu hình chính).",
             r"``Ep. đến 0.9'' là episode đầu tiên mà cửa sổ 50 episode đạt tỉ lệ",
             r"thành công 0.9.}",
             r"\label{tab:rescue_train}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llcccc}", r"\hline",
             r"\textbf{Thuật toán} & \textbf{Điều kiện} & \textbf{Thành công} & "
             r"\textbf{Ep. đến 0.9} & \textbf{Deadlock} & \textbf{Timeout} \\",
             r"\hline"]
    for a in ALGOS:
        for c in CONDS:
            s = st[(a, c)]
            lines.append(f"{a} & {CLAB[c]} & {s['success']:.3f} $\\pm$ "
                         f"{s['success_std']:.3f} & {s['e2t']:.0f} $\\pm$ "
                         f"{s['e2t_std']:.0f} & {s['deadlock']:.3f} & "
                         f"{s['deadlock']:.3f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    # timeout column placeholder replaced below
    return "\n".join(lines) + "\n"


def table2(st, rows):
    lines = [r"\begin{table*}[t]",
             r"\caption{Ô cứu hộ và tốc độ học trên Setting 3 cho bốn thuật toán TD",
             r"(30 seed, 1000 episode). ``Ep. đến 0.9'' là episode đầu tiên mà",
             r"cửa sổ 50 episode đạt tỉ lệ thành công 0.9; thời gian là tỉ lệ",
             r"episode timeout trong training.}",
             r"\label{tab:rescue_train}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llcccc}", r"\hline",
             r"\textbf{Thuật toán} & \textbf{Điều kiện} & \textbf{Thành công} & "
             r"\textbf{Ep. đến 0.9} & \textbf{Deadlock} & \textbf{Timeout} \\",
             r"\hline"]
    for a in ALGOS:
        for c in CONDS:
            s = st[(a, c)]
            rs = [r for r in rows if r["algo"] == a and r["cond"] == c]
            to = float(np.mean([np.mean(r["timeout"]) for r in rs]))
            lines.append(f"{a} & {CLAB[c]} & {s['success']:.3f} $\\pm$ "
                         f"{s['success_std']:.3f} & {s['e2t']:.0f} $\\pm$ "
                         f"{s['e2t_std']:.0f} & {s['deadlock']:.3f} & "
                         f"{to:.3f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def figure(st, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.0), dpi=220,
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for j, a in enumerate(ALGOS):
        for c in CONDS:
            s = st[(a, c)]
            x = (np.arange(len(s["succ_bin"])) + 1) * BIN
            axes[j].plot(x, s["succ_bin"], color=COL[c], lw=1.4, label=CLAB[c])
        axes[j].set_title(a, fontsize=11)
        axes[j].grid(alpha=0.25, linestyle="--")
        axes[j].tick_params(labelsize=9)
        axes[j].set_xlabel("Episode", fontsize=9)
    axes[0].set_ylabel("Tỉ lệ thành công", fontsize=10)
    axes[2].set_ylabel("Tỉ lệ thành công", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


def main():
    rows = load()
    st = stats(rows)
    with open(os.path.join(REP, "tables_vi_rescue_train.tex"), "w",
              encoding="utf-8") as f:
        f.write(table2(st, rows))
    figure(st, os.path.join(REP, "figures", "fig9_rescue_training.png"))
    for a in ALGOS:
        for c in CONDS:
            s = st[(a, c)]
            print(f"{a:20s} {c:8s} succ {s['success']:.3f} e2t {s['e2t']:.0f}"
                  f"±{s['e2t_std']:.0f} dead {s['deadlock']:.3f}")


if __name__ == "__main__":
    main()
