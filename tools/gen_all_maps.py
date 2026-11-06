"""Tables for the all-maps (tile everywhere) study."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
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


def main():
    d = json.load(open(os.path.join(ROOT, "saved_models", "study_all.json"),
                       encoding="utf-8"))
    s = d["td_summary"]

    lines = [r"\begin{table*}[t]",
             r"\caption{Hiệu năng của bốn thuật toán không cần mô hình trên 30 seed:",
             r"tỉ lệ thành công, tỉ lệ hết giờ, return không chiết khấu, return",
             r"chiết khấu và số bước. Cả bốn bản đồ đều có ô cứu hộ, với",
             r"$r_{\text{rescue}}=0$.}",
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

    lines = [r"\begin{table}[t]",
             r"\caption{Hành vi sử dụng ô cứu hộ trên 30 seed: tỉ lệ episode bị phát",
             r"hiện deadlock, số lần kích hoạt và số lần cứu hộ trung bình mỗi episode.",
             r"Cả bốn bản đồ đều có ô, với $r_{\text{rescue}}=0$.}",
             r"\label{tab:td_behavior}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{llccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Kẹt} & \textbf{Kích hoạt} & \textbf{Cứu hộ} \\",
             r"\hline"]
    for k in SET:
        for a in TD:
            m = s[f"{k}|{a}"]
            lines.append(f"{SLAB[k]} & {a} & {m['deadlock_rate']:.3f} & "
                         f"{m['mean_rescue_activations']:.2f} & "
                         f"{m['mean_rescue_uses']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_td_behavior.tex", "\n".join(lines) + "\n")

    # ablation per map: tile vs no tile
    aw = d["ablation_with"]
    ao = d["ablation_without"]
    lines = [r"\begin{table*}[t]",
             r"\caption{So sánh có ô cứu hộ và bỏ ô trên từng bản đồ, 30 seed,",
             r"$r_{\text{rescue}}=0$. Cột ``có'' lấy từ thí nghiệm chính; cột",
             r"``không'' chạy lại trên cùng bản đồ với ô \texttt{R} thay bằng sàn.}",
             r"\label{tab:ablation}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Bản đồ} & \textbf{Thuật toán} & \textbf{Thành công} & "
             r"\textbf{Return} & \textbf{Bước} & \textbf{Thành công} & "
             r"\textbf{Return} & \textbf{Bước} \\",
             r"\hline",
             r"\multicolumn{2}{c}{} & \multicolumn{3}{c}{\textbf{Có ô}} & "
             r"\multicolumn{3}{c}{\textbf{Không ô}} \\",
             r"\hline"]
    for k in SET:
        for a in TD:
            w = aw[f"{k}|{a}"]
            o = ao[k][f"{k}|{a}"]
            lines.append(f"{SLAB[k]} & {a} & {w['success_rate']:.2f} & "
                         f"{w['mean_reward']:.1f} & {w['mean_steps']:.1f} & "
                         f"{o['success_rate']:.2f} & {o['mean_reward']:.1f} & "
                         f"{o['mean_steps']:.1f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_ablation.tex", "\n".join(lines) + "\n")

    # DP table, gamma 0.95
    dp = d["dp"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Kết quả của hai thuật toán quy hoạch động trên bốn cấu hình,",
             r"cả bốn đều có ô cứu hộ, với $r_{\text{rescue}}=0$.}",
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


if __name__ == "__main__":
    main()
