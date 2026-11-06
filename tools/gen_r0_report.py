"""Generate Vietnamese tables and learning curves from study_r0.json."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
FIG = os.path.join(REP, "figures")

TD = ["Q-Learning", "SARSA", "Expected SARSA", "Monte Carlo Control"]
SET = ["setting1", "setting2", "setting3", "setting4"]
SLAB = {"setting1": "Setting 1", "setting2": "Setting 2",
        "setting3": "Setting 3", "setting4": "Setting 4"}


def ms(m, k, dec=2):
    return f"{m[k]:.{dec}f} $\\pm$ {m[k + '_std']:.{dec}f}"


def write(name, text):
    with open(os.path.join(REP, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", name)


def td_tables(d):
    s = d["td_summary"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Hiệu năng của bốn thuật toán không cần mô hình trên 30 seed: tỉ lệ",
             r"thành công, tỉ lệ hết giờ, return không chiết khấu, return chiết khấu và số",
             r"bước. Cấu hình chính dùng $r_{\text{rescue}}=0$.}",
             r"\label{tab:td_perf}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Thành công} & \textbf{Hết giờ} & \textbf{Return} & \textbf{Return chiết khấu} & \textbf{Bước} \\",
             r"\hline"]
    for k in SET:
        for a in TD:
            m = s[f"{k}|{a}"]
            lines.append(f"{SLAB[k]} & {a} & {ms(m,'success_rate')} & "
                         f"{m['timeout_rate']:.3f} & {ms(m,'mean_reward',1)} & "
                         f"{ms(m,'mean_discounted_reward')} & {m['mean_steps']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_td_perf.tex", "\n".join(lines) + "\n")

    lines = [r"\begin{table*}[t]",
             r"\caption{Hành vi sử dụng ô cứu hộ trên 30 seed: tỉ lệ episode bị phát hiện",
             r"deadlock, số lần kích hoạt và số lần cứu hộ trung bình mỗi episode.",
             r"Cấu hình chính dùng $r_{\text{rescue}}=0$.}",
             r"\label{tab:td_behavior}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{llccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Kẹt} & \textbf{Kích hoạt} & \textbf{Cứu} \\",
             r"\hline"]
    for k in SET:
        for a in TD:
            m = s[f"{k}|{a}"]
            lines.append(f"{SLAB[k]} & {a} & {m['deadlock_rate']:.3f} & "
                         f"{m['mean_rescue_activations']:.2f} & {m['mean_rescue_uses']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_td_behavior.tex", "\n".join(lines) + "\n")


def ablation_table(d):
    ab = d["ablation"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Ảnh hưởng của ô cứu hộ, so sánh cùng một bản đồ có và không có",
             r"ô cứu hộ, với $r_{\text{rescue}}=0$.}",
             r"\label{tab:ablation}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Thuật toán} & \textbf{Bản đồ} & \textbf{Thành công} & \textbf{Return} & \textbf{Chiết khấu} & \textbf{Bước} & \textbf{Kích hoạt} & \textbf{Cứu} \\",
             r"\hline"]
    for a in TD:
        for side, lab in (("with_rescue", "có ô cứu hộ"),
                          ("without_rescue", "không có")):
            m = ab[side][f"setting3|{a}"]
            lines.append(f"{a} & {lab} & {ms(m,'success_rate')} & "
                         f"{m['mean_reward']:.1f} & {ms(m,'mean_discounted_reward')} & "
                         f"{m['mean_steps']:.1f} & {m['mean_rescue_activations']:.2f} & "
                         f"{m['mean_rescue_uses']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_ablation.tex", "\n".join(lines) + "\n")


def dp_table(d):
    dp = d["dp"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Kết quả của hai thuật toán quy hoạch động trên bốn cấu hình,",
             r"với $r_{\text{rescue}}=0$.}",
             r"\label{tab:dp}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Thành công} & \textbf{Thời gian (s)} & \textbf{Số vòng} & \textbf{$V^*(s_0)$} & \textbf{Return} & \textbf{Bước} \\",
             r"\hline"]
    for k in SET:
        for alg in ("Value Iteration", "Policy Iteration"):
            m = dp[f"{k}|{alg}|0.95"]
            lines.append(f"{SLAB[k]} & {alg} & {m['success_rate']:.2f} & "
                         f"{m['training_time']:.2f} & {m['iterations']} & "
                         f"{m['v_start']:.2f} & {m['mean_reward']:.1f} & "
                         f"{m['mean_steps']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_dp.tex", "\n".join(lines) + "\n")


def sens_table(sens):
    rows = []
    for a in ("0.05", "0.1", "0.2", "0.4"):
        m = sens["alpha"][a]
        rows.append(f"$\\alpha$ & {a} & {ms(m,'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} \\\\")
    for g in ("0.8", "0.9", "0.95", "0.99"):
        m = sens["gamma"][g]
        rows.append(f"$\\gamma$ & {g} & {ms(m,'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} \\\\")
    lines = [r"\begin{table}[t]",
             r"\caption{Độ nhạy của hệ số học $\alpha$ và hệ số chiết khấu $\gamma$ trên",
             r"Setting 3 (30 seed, 600 episode), với $r_{\text{rescue}}=0$.}",
             r"\label{tab:sens}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Tham số} & \textbf{Giá trị} & \textbf{Thành công} & \textbf{Return} & \textbf{Bước} \\",
             r"\hline"] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_sens.tex", "\n".join(lines) + "\n")

    rows = []
    for g in ("0.8", "0.9", "0.95", "0.99"):
        m = sens["gamma"][g]
        rows.append(f"$\\gamma$ & {g} & {ms(m,'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} \\\\")
    lines = [r"\begin{table}[t]",
             r"\caption{Độ nhạy theo hệ số chiết khấu $\gamma$ trên Setting 3",
             r"(30 seed, 600 episode), với $r_{\text{rescue}}=0$.}",
             r"\label{tab:sens-gamma}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Tham số} & \textbf{Giá trị} & \textbf{Thành công} & \textbf{Return} & \textbf{Bước} \\",
             r"\hline"] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_sens-gamma.tex", "\n".join(lines) + "\n")

    labels = {"r=0": "$r=0$ (chính)", "r=+10": "$r=+10$ (biến thể)",
              "r=-10": "$r=-10$", "box-focused": "ưu tiên thùng ($r=0$)",
              "deadlock-heavy": "phạt kẹt $-100$ ($r=0$)"}
    rows = []
    for k in ("r=0", "r=+10", "r=-10", "box-focused", "deadlock-heavy"):
        m = sens["reward"][k]
        rows.append(f"{labels[k]} & {ms(m,'success_rate')} & {m['mean_reward']:.1f} & "
                    f"{m['mean_steps']:.1f} & {m['mean_rescue_uses']:.2f} \\\\")
    lines = [r"\begin{table}[t]",
             r"\caption{Ảnh hưởng của cấu hình phần thưởng đối với Q-Learning trên",
             r"Setting 3 (30 seed, 600 episode). Các dòng không ghi khác dùng",
             r"$r_{\text{rescue}}=0$; dòng $+10$ là biến thể do nhóm thêm.}",
             r"\label{tab:sens-reward}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thành công} & \textbf{Return} & \textbf{Bước} & \textbf{Cứu hộ} \\",
             r"\hline"] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_sens-reward.tex", "\n".join(lines) + "\n")


def learning_curves(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    td = d["td"]
    for k in SET:
        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
        for a in TD:
            runs = td[f"{k}|{a}"]
            arr = np.array([r["rewards"] for r in runs], dtype=float)
            x = (np.arange(arr.shape[1]) + 1) * 5
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(x, mean, lw=1.6, label=a)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18)
        ax.set_title(f"Đường học trên {SLAB[k]}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Return huấn luyện trung bình", fontsize=12)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(frameon=True, fontsize=9, loc="lower right")
        fig.tight_layout()
        p = os.path.join(FIG, f"fig2_learning_{k}.png")
        fig.savefig(p)
        plt.close(fig)
        print("wrote", p)


def main():
    d = json.load(open(os.path.join(ROOT, "saved_models", "study_r0.json"),
                       encoding="utf-8"))
    td_tables(d)
    ablation_table(d)
    dp_table(d)
    sp = os.path.join(ROOT, "saved_models", "sens_r0.json")
    if os.path.exists(sp):
        sens = json.load(open(sp, encoding="utf-8"))
        sens_table(sens)
    learning_curves(d)


if __name__ == "__main__":
    main()
