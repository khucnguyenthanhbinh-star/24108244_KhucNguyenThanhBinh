# -*- coding: utf-8 -*-
"""Sinh toan bo bang so lieu cua bao cao, ca ban Viet va ban Anh.

    python tools/gen_tables.py

Doc cac tep trong saved_models/ va ghi ra report/:

    tables_vi_setting.tex / tables_en_setting.tex       bon cau hinh ban do
    tables_vi_placement.tex / tables_en_placement.tex   danh gia vi tri o cuu ho
    tables_vi_dp.tex / tables_en_dp.tex                 ket qua quy hoach dong
    tables_vi_vipi.tex / tables_en_vipi.tex             so sanh VI va PI
    tables_vi_td_perf.tex / tables_en_td_perf.tex       hieu nang bon thuat toan TD
    tables_vi_td_behavior.tex / ..._en_td_behavior.tex  hanh vi o cuu ho
    tables_vi_ablation.tex / tables_en_ablation.tex     co o va khong o
    tables_vi_rescue_train.tex / ..._rescue_train.tex   toc do hoc khi co o
    tables_vi_sens.tex / tables_en_sens.tex             do nhay alpha, gamma
    tables_vi_sens-reward.tex / ..._sens-reward.tex     do nhay phan thuong
    tables_vi_epsilon.tex / tables_en_epsilon.tex       quet he so giam epsilon

Moi bang chi co mot ham sinh: cung mot nguon du lieu, hai ngon ngu chi khac
caption va ten cot. Nho vay so lieu cua hai ban khong the lech nhau.

Script nay chi sinh bang, khong dung lai hinh: cac hinh trong report/figures
duoc tao boi sokoban/experiments.py va tools/rescue_probe.py.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SET_ORDER = ["setting1", "setting2", "setting3", "setting4"]
SLAB = {"setting1": "Setting 1", "setting2": "Setting 2",
        "setting3": "Setting 3", "setting4": "Setting 4"}
TD = ["Q-Learning", "SARSA", "Expected SARSA", "Monte Carlo Control"]

# O cuu ho khi huan luyen (tools/rescue_probe.py).
RP_ALGOS = TD
RP_CONDS = ["no_tile", "r0", "r10"]
RP_CLAB = {"no_tile": "không ô", "r0": "có ô, r=0", "r10": "có ô, r=+10"}
RP_EN = {"no_tile": "No tile", "r0": r"With tile, $r=0$",
         "r10": r"With tile, $r=+10$"}
RP_BIN = 50
RP_THR = 0.9
RP_WIN = 50

# Mo ta va kich thuoc cua bon ban do.
SETTING_ROWS_VI = [
    ("setting1", "Bản đồ nhỏ cơ sở", r"$5\times6$", r"81\,000"),
    ("setting2", "Hành lang hẹp", r"$6\times7$", r"222\,264"),
    ("setting3", "Tường ngăn phân vùng", r"$6\times7$", r"222\,264"),
    ("setting4", "Mở rộng, hốc cứu hộ riêng", r"$6\times8$", r"331\,776"),
]
SETTING_ROWS_EN = [
    ("setting1", "Small baseline map", r"$5\times6$", r"81\,000"),
    ("setting2", "Narrow corridor", r"$6\times7$", r"222\,264"),
    ("setting3", "Partitioning interior wall", r"$6\times7$", r"222\,264"),
    ("setting4", "Extended layout, dedicated rescue alcove", r"$6\times8$",
     r"331\,776"),
]

PLACEMENT_ROWS_VI = [
    (r"Cách Start 2 bước", "Có", "Thấp", r"Lệch khỏi đường đi tối ưu"),
    (r"Ngay sát Start", "Có", r"$\approx 0$", r"Trở thành lượt thử miễn phí"),
    (r"Ngẫu nhiên", "Có", "Đổi", r"Có thể nằm ngoài vùng học"),
    (r"Kích hoạt sẵn ($u=1$)", "Không", "0", r"Bỏ điều kiện kích hoạt trước"),
]
PLACEMENT_ROWS_EN = [
    (r"Fixed, 2 steps from start", "Yes", "Low",
     r"Detours from geometric optimal path"),
    (r"Directly adjacent to start", "Yes", r"$\approx 0$",
     r"Becomes an essentially free trial"),
    (r"Random across maps", "Yes", "Variable",
     r"May fall outside reachable zone"),
    (r"Pre-activated ($u=1$)", "No", "0", r"Violates pre-activation rule"),
]

REWARD_LABEL_VI = {
    "r=0": r"$r=0$ (chính)",
    "r=+10": r"$r=+10$ (biến thể)",
    "r=-10": r"$r=-10$",
    "box-focused": r"ưu tiên thùng ($r=0$)",
    "deadlock-heavy": r"phạt kẹt $-100$ ($r=0$)",
}
REWARD_LABEL_EN = {
    "r=0": r"$r=0$ (Baseline)",
    "r=+10": r"$r=+10$ (Bonus)",
    "r=-10": r"$r=-10$ (Penalty)",
    "box-focused": r"Box Priority ($r=0$)",
    "deadlock-heavy": r"Deadlock $-100$",
}

EPS_DECAYS = ["0.9", "0.99", "0.995", "0.9992"]
EPS_FLOOR_EN = {"0.9": "38", "0.99": "390", "0.995": "781", "0.9992": "$>$1000"}


def load(name):
    path = os.path.join(ROOT, "saved_models", name)
    if not os.path.exists(path):
        raise SystemExit("Khong tim thay %s. Hay chay tools/run_study.py truoc." % path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write(name, lines, newline=None):
    os.makedirs(REP, exist_ok=True)
    path = os.path.join(REP, name)
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


def thousands(n):
    """Chen khoang trang mong vao so lon: 3004 -> 3\\,004."""
    return f"{n:,}".replace(",", r"\,")


def sci(value):
    """Dang $1.4 \\times 10^{-5}$ cho cot chenh lech gia tri."""
    text = f"{value:.1e}"
    mant, exp = text.split("e")
    return r"$%s \times 10^{%d}$" % (mant, int(exp))


def ms(m, key, dec=2):
    return f"{m[key]:.{dec}f} $\\pm$ {m[key + '_std']:.{dec}f}"


# --------------------------------------------------------------------------
# Bang tieng Viet
# --------------------------------------------------------------------------
def vi_setting(dp):
    lines = [r"\begin{table*}[t]",
             r"\caption{Bốn cấu hình bản đồ và kích thước không gian trạng thái.}",
             r"\label{tab:setting}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrrrr}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Đặc điểm} & \textbf{Lưới} & "
             r"\textbf{Số state tiềm năng} & \textbf{State tới được} & "
             r"\textbf{Trạng thái kết thúc} \\"]
    for key, desc, grid, potential in SETTING_ROWS_VI:
        d = dp[key]
        lines.append(f"{SLAB[key]} & {desc} & {grid} & {potential} & "
                     f"{thousands(d['n_states'])} & "
                     f"{thousands(d['n_terminal_states'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_setting.tex", lines)


def vi_placement():
    lines = [r"\begin{table}[t]",
             r"\caption{Đánh giá định tính các phương án thiết kế vị trí ô cứu hộ.}",
             r"\label{tab:placement}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lccl}", r"\hline",
             r"\textbf{Phương án vị trí} & \textbf{Đúng đề} & \textbf{Chi phí} & "
             r"\textbf{Đặc điểm hành vi / Rủi ro} \\",
             r"\hline"]
    for row in PLACEMENT_ROWS_VI:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_placement.tex", lines, newline="\n")


def vi_vipi(dp):
    lines = [r"\begin{table}[t]",
             r"\caption{So sánh giá trị và chính sách giữa Value Iteration và Policy Iteration.}",
             r"\label{tab:vipi}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{$\max|V_{\text{VI}}-V_{\text{PI}}|$} & "
             r"\textbf{Trùng khớp chính sách} & \textbf{Số state} \\",
             r"\hline"]
    for key in SET_ORDER:
        d = dp[key]
        n = thousands(d["n_states"])
        lines.append(f"{SLAB[key]} & {sci(d['max_abs_value_difference'])} & "
                     f"{d['same_policy_states']} / {n} & {n} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_vipi.tex", lines)


def vi_td_perf(summary):
    lines = [r"\begin{table*}[t]",
             r"\caption{Hiệu năng của bốn thuật toán không cần mô hình trên 30 seed:",
             r"tỉ lệ thành công, tỉ lệ hết giờ, return không chiết khấu, return",
             r"chiết khấu và số bước. Cả bốn bản đồ đều có ô cứu hộ, với",
             r"$r_{\text{rescue}}=0$.}",
             r"\label{tab:td_perf}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Thành công} & "
             r"\textbf{Hết giờ} & \textbf{Return} & \textbf{Return chiết khấu} & "
             r"\textbf{Bước} \\",
             r"\hline"]
    for key in SET_ORDER:
        for algo in TD:
            m = summary[f"{key}|{algo}"]
            lines.append(f"{SLAB[key]} & {algo} & {ms(m, 'success_rate')} & "
                         f"{m['timeout_rate']:.3f} & {ms(m, 'mean_reward', 1)} & "
                         f"{ms(m, 'mean_discounted_reward')} & "
                         f"{m['mean_steps']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_td_perf.tex", lines)


def vi_td_behavior(summary):
    lines = [r"\begin{table*}[t]",
             r"\caption{Hành vi sử dụng ô cứu hộ trên 30 seed: tỉ lệ episode bị phát",
             r"hiện deadlock, số lần kích hoạt và số lần cứu hộ trung bình mỗi episode.",
             r"Cả bốn bản đồ đều có ô, với $r_{\text{rescue}}=0$.}",
             r"\label{tab:td_behavior}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{llccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Kẹt} & "
             r"\textbf{Kích hoạt} & \textbf{Cứu} \\",
             r"\hline"]
    for key in SET_ORDER:
        for algo in TD:
            m = summary[f"{key}|{algo}"]
            lines.append(f"{SLAB[key]} & {algo} & {m['deadlock_rate']:.3f} & "
                         f"{m['mean_rescue_activations']:.2f} & "
                         f"{m['mean_rescue_uses']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_td_behavior.tex", lines)


def vi_ablation(data):
    with_tile = data["ablation_with"]
    without = data["ablation_without"]
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
    for key in SET_ORDER:
        for algo in TD:
            k = f"{key}|{algo}"
            w = with_tile[k]
            o = without[key][k]
            lines.append(f"{SLAB[key]} & {algo} & {w['success_rate']:.2f} & "
                         f"{w['mean_reward']:.1f} & {w['mean_steps']:.1f} & "
                         f"{o['success_rate']:.2f} & {o['mean_reward']:.1f} & "
                         f"{o['mean_steps']:.1f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_ablation.tex", lines)


def vi_dp(dp):
    lines = [r"\begin{table*}[t]",
             r"\caption{Kết quả của hai thuật toán quy hoạch động trên bốn cấu hình,",
             r"cả bốn đều có ô cứu hộ, với $r_{\text{rescue}}=0$.}",
             r"\label{tab:dp}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thuật toán} & \textbf{Thành công} & "
             r"\textbf{Thời gian (s)} & \textbf{Số vòng} & "
             r"\textbf{$V^*(s_0)$} & \textbf{Return} & \textbf{Bước} \\",
             r"\hline"]
    for key in SET_ORDER:
        for alg in ("Value Iteration", "Policy Iteration"):
            m = dp[f"{key}|{alg}|0.95"]
            lines.append(f"{SLAB[key]} & {alg} & {m['success_rate']:.2f} & "
                         f"{m['training_time']:.2f} & {m['iterations']} & "
                         f"{m['v_start']:.2f} & {m['mean_reward']:.1f} & "
                         f"{m['mean_steps']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_dp.tex", lines)


def vi_sens(sens):
    rows = []
    for a in ("0.05", "0.1", "0.2", "0.4"):
        m = sens["alpha"][a]
        rows.append(f"$\\alpha$ & {a} & {ms(m, 'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} \\\\")
    for g in ("0.8", "0.9", "0.95", "0.99"):
        m = sens["gamma"][g]
        rows.append(f"$\\gamma$ & {g} & {ms(m, 'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} \\\\")
    lines = [r"\begin{table}[t]",
             r"\caption{Độ nhạy của hệ số học $\alpha$ và hệ số chiết khấu $\gamma$ trên",
             r"Setting 3 (30 seed, 600 episode), với $r_{\text{rescue}}=0$.}",
             r"\label{tab:sens}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Tham số} & \textbf{Giá trị} & \textbf{Thành công} & "
             r"\textbf{Return} & \textbf{Bước} \\",
             r"\hline"] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_sens.tex", lines)

    rows = []
    for g in ("0.8", "0.9", "0.95", "0.99"):
        m = sens["gamma"][g]
        rows.append(f"$\\gamma$ & {g} & {ms(m, 'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} \\\\")
    lines = [r"\begin{table}[t]",
             r"\caption{Độ nhạy theo hệ số chiết khấu $\gamma$ trên Setting 3",
             r"(30 seed, 600 episode), với $r_{\text{rescue}}=0$.}",
             r"\label{tab:sens-gamma}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Tham số} & \textbf{Giá trị} & \textbf{Thành công} & "
             r"\textbf{Return} & \textbf{Bước} \\",
             r"\hline"] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_sens-gamma.tex", lines)

    rows = []
    for key in ("r=0", "r=+10", "r=-10", "box-focused", "deadlock-heavy"):
        m = sens["reward"][key]
        rows.append(f"{REWARD_LABEL_VI[key]} & {ms(m, 'success_rate')} & "
                    f"{m['mean_reward']:.1f} & {m['mean_steps']:.1f} & "
                    f"{m['mean_rescue_uses']:.2f} \\\\")
    lines = [r"\begin{table}[t]",
             r"\caption{Ảnh hưởng của cấu hình phần thưởng đối với Q-Learning trên",
             r"Setting 3 (30 seed, 600 episode). Các dòng không ghi khác dùng",
             r"$r_{\text{rescue}}=0$; dòng $+10$ là biến thể do nhóm thêm.}",
             r"\label{tab:sens-reward}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Thành công} & \textbf{Return} & "
             r"\textbf{Bước} & \textbf{Cứu hộ} \\",
             r"\hline"] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_sens-reward.tex", lines)


def vi_epsilon(data):
    lines = [r"\begin{table*}[t]",
             r"\caption{Quét hệ số giảm $\varepsilon$ trên Setting 3 (30 seed, 1000 episode,",
             r"$r_{\text{rescue}}=0$). $\varepsilon$ cuối bằng 0.02 ở ba lịch đầu và 0.4492 ở",
             r"lịch 0.9992. Cột cuối là số lần kích hoạt ô cứu hộ; số lần sử dụng bằng 0 ở mọi",
             r"dòng.}",
             r"\label{tab:epsilon}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{llccccc}", r"\hline",
             r"\textbf{Decay} & \textbf{Thuật toán} & \textbf{Thành công} & "
             r"\textbf{ĐLC} & \textbf{Return} & \textbf{Bước} & \textbf{Kích hoạt} \\",
             r"\hline"]
    for decay in EPS_DECAYS:
        block = data["decays"][decay]
        for algo in TD:
            m = block[algo]
            lines.append(f"{float(decay):.4f} & {algo} & {m['success_rate']:.3f} & "
                         f"{m['success_rate_std']:.3f} & {m['mean_reward']:.1f} & "
                         f"{m['mean_steps']:.1f} & "
                         f"{m['mean_rescue_activations']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_epsilon.tex", lines, newline="\n")


# --------------------------------------------------------------------------
# O cuu ho khi huan luyen (dung chung cho hai ngon ngu)
# --------------------------------------------------------------------------
def rescue_rows():
    rows = []
    for name in ("rescue_probe.json", "rescue_probe_rest.json"):
        path = os.path.join(ROOT, "saved_models", name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                rows += json.load(f)["rows"]
    return rows


def rescue_stats(rows):
    out = {}
    for a in RP_ALGOS:
        for c in RP_CONDS:
            rs = [r for r in rows if r["algo"] == a and r["cond"] == c]
            if not rs:
                continue
            n = min(len(r["solved"]) for r in rs)
            solved = np.array([r["solved"][:n] for r in rs], float)
            dead = np.array([r["deadlock"][:n] for r in rs], float)
            e2t = []
            for row in solved:
                idx = n
                for e in range(0, max(1, n - RP_WIN)):
                    if row[e:e + RP_WIN].mean() >= RP_THR:
                        idx = e
                        break
                e2t.append(idx)
            e2t = np.array(e2t, float)
            out[(a, c)] = {
                "seeds": len(rs),
                "success": solved.mean(),
                "success_std": solved.mean(axis=1).std(ddof=1),
                "e2t": e2t.mean(),
                "e2t_std": e2t.std(ddof=1),
                "deadlock": dead.mean(),
            }
    return out


def vi_rescue_train(st, rows):
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
    for algo in RP_ALGOS:
        for cond in RP_CONDS:
            s = st[(algo, cond)]
            rs = [r for r in rows if r["algo"] == algo and r["cond"] == cond]
            to = float(np.mean([np.mean(r["timeout"]) for r in rs]))
            lines.append(f"{algo} & {RP_CLAB[cond]} & {s['success']:.3f} $\\pm$ "
                         f"{s['success_std']:.3f} & {s['e2t']:.0f} $\\pm$ "
                         f"{s['e2t_std']:.0f} & {s['deadlock']:.3f} & "
                         f"{to:.3f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_rescue_train.tex", lines)


# --------------------------------------------------------------------------
# Bang tieng Anh
# --------------------------------------------------------------------------
def en_setting(dp):
    lines = [r"\begin{table*}[t]",
             r"\caption{Four map configurations and state space dimensions.}",
             r"\label{tab:setting}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrrrr}", r"\hline",
             r"\textbf{Configuration} & \textbf{Description} & \textbf{Grid} & "
             r"\textbf{Potential States} & \textbf{Reachable States} & "
             r"\textbf{Terminal States} \\",
             r"\hline"]
    for key, desc, grid, potential in SETTING_ROWS_EN:
        d = dp[key]
        lines.append(f"{SLAB[key]} & {desc} & {grid} & {potential} & "
                     f"{thousands(d['n_states'])} & "
                     f"{thousands(d['n_terminal_states'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_en_setting.tex", lines)


def en_placement():
    lines = [r"\begin{table}[t]",
             r"\caption{Qualitative comparison of rescue tile placement strategies.}",
             r"\label{tab:placement}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lccl}", r"\hline",
             r"\textbf{Placement Strategy} & \textbf{Valid} & \textbf{Cost} & "
             r"\textbf{Behavioral Characteristics / Risk} \\",
             r"\hline"]
    for row in PLACEMENT_ROWS_EN:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_en_placement.tex", lines)


def en_dp(dp):
    lines = [r"\begin{table*}[t]",
             r"\caption{Results of the two dynamic programming algorithms on the four "
             r"configurations ($r_{\text{rescue}}=0$, $\gamma=0.95$).}",
             r"\label{tab:dp}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Configuration} & \textbf{Algorithm} & \textbf{Success Rate} & "
             r"\textbf{Time (s)} & \textbf{Iterations} & \textbf{$V^*(s_0)$} & "
             r"\textbf{Return} & \textbf{Steps} \\",
             r"\hline"]
    for key in SET_ORDER:
        for name in ("Value Iteration", "Policy Iteration"):
            m = dp[f"{key}|{name}|0.95"]
            lines.append(f"{SLAB[key]} & {name} & {m['success_rate']:.2f} & "
                         f"{m['training_time']:.2f} & {m['iterations']} & "
                         f"{m['v_start']:.2f} & {m['mean_reward']:.1f} & "
                         f"{m['mean_steps']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_en_dp.tex", lines)


def en_vipi(dp):
    lines = [r"\begin{table}[t]",
             r"\caption{Comparison of value functions and greedy policies between "
             r"Value Iteration and Policy Iteration.}",
             r"\label{tab:vipi}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lccc}", r"\hline",
             r"\textbf{Config.} & \textbf{$\max|V_{\text{VI}}-V_{\text{PI}}|$} & "
             r"\textbf{Policy Match} & \textbf{States} \\",
             r"\hline"]
    for key in SET_ORDER:
        d = dp[key]
        n = thousands(d["n_states"])
        lines.append(f"{SLAB[key]} & {sci(d['max_abs_value_difference'])} & "
                     f"{d['same_policy_states']} / {n} & {n} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_en_vipi.tex", lines)


def en_td_perf(summary):
    lines = [r"\begin{table*}[t]",
             r"\caption{Performance evaluation of the four model-free algorithms across "
             r"30 seeds ($r_{\text{rescue}}=0$, greedy policy).}",
             r"\label{tab:td_perf}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llcccc}", r"\hline",
             r"\textbf{Configuration} & \textbf{Algorithm} & \textbf{Success Rate} & "
             r"\textbf{Return} & \textbf{Discounted Return} & \textbf{Steps} \\",
             r"\hline"]
    for key in SET_ORDER:
        for algo in TD:
            m = summary[f"{key}|{algo}"]
            lines.append(
                f"{SLAB[key]} & {algo} & {m['success_rate']:.2f} $\\pm$ "
                f"{m['success_rate_std']:.2f} & {m['mean_reward']:.1f} $\\pm$ "
                f"{m['mean_reward_std']:.1f} & "
                f"{m['mean_discounted_reward']:.2f} $\\pm$ "
                f"{m['mean_discounted_reward_std']:.2f} & "
                f"{m['mean_steps']:.1f} $\\pm$ {m['mean_steps_std']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_en_td_perf.tex", lines)


def en_td_behavior(summary):
    lines = [r"\begin{table}[t]",
             r"\caption{Rescue tile utilization behavior over 30 seeds: deadlock rate, "
             r"mean activations, and rescues per episode ($r_{\text{rescue}}=0$).}",
             r"\label{tab:td_behavior}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llccc}", r"\hline",
             r"\textbf{Config.} & \textbf{Algorithm} & \textbf{Deadlock} & "
             r"\textbf{Activ.} & \textbf{Rescue} \\",
             r"\hline"]
    for key in SET_ORDER:
        for algo in TD:
            m = summary[f"{key}|{algo}"]
            lines.append(f"{SLAB[key]} & {algo} & {m['deadlock_rate']:.3f} & "
                         f"{m['mean_rescue_activations']:.2f} & "
                         f"{m['mean_rescue_uses']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_en_td_behavior.tex", lines)


def en_ablation(data):
    with_tile = data["ablation_with"]
    without = data["ablation_without"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Ablation study comparing configurations with and without rescue "
             r"tile across all maps (30 seeds, $r_{\text{rescue}}=0$).}",
             r"\label{tab:ablation}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Configuration} & \textbf{Algorithm} & \textbf{Success} & "
             r"\textbf{Return} & \textbf{Steps} & \textbf{Success} & \textbf{Return} & "
             r"\textbf{Steps} \\",
             r"\hline",
             r"\multicolumn{2}{c}{} & \multicolumn{3}{c}{\textbf{With Rescue Tile}} & "
             r"\multicolumn{3}{c}{\textbf{Without Rescue Tile}} \\",
             r"\hline"]
    for key in SET_ORDER:
        for algo in TD:
            k = f"{key}|{algo}"
            a = with_tile[k]
            b = without[key][k]
            lines.append(f"{SLAB[key]} & {algo} & {a['success_rate']:.2f} & "
                         f"{a['mean_reward']:.1f} & {a['mean_steps']:.1f} & "
                         f"{b['success_rate']:.2f} & {b['mean_reward']:.1f} & "
                         f"{b['mean_steps']:.1f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    write("tables_en_ablation.tex", lines)


def en_rescue_train(st, rows, data):
    n = int(data.get("seeds", 30))
    lines = [r"\begin{table*}[t]",
             r"\caption{Rescue tile impact on learning dynamics on Setting 3 across four "
             r"TD/MC algorithms (30 seeds, 1000 episodes). ``Ep. to 0.9'' denotes the "
             r"first episode where the rolling 50-episode success rate reaches 0.9; "
             r"Timeout denotes episode truncation rate.}",
             r"\label{tab:rescue_train}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcccc}", r"\hline",
             r"\textbf{Algorithm} & \textbf{Condition} & \textbf{Training Success} & "
             r"\textbf{Ep. to 0.9} & \textbf{Deadlock} & \textbf{Timeout} \\",
             r"\hline"]
    for algo in RP_ALGOS:
        for cond in RP_CONDS:
            s = st[(algo, cond)]
            rs = [r for r in rows if r["algo"] == algo and r["cond"] == cond]
            to = float(np.mean([np.mean(r["timeout"]) for r in rs]))
            lines.append(f"{algo} & {RP_EN[cond]} & {s['success']:.3f} $\\pm$ "
                         f"{s['success_std']:.3f} & {s['e2t']:.0f} $\\pm$ "
                         f"{s['e2t_std']:.0f} & {s['deadlock']:.3f} & "
                         f"{to:.3f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    write("tables_en_rescue_train.tex", lines)


def en_sens(sens):
    lines = [r"\begin{table}[t]",
             r"\caption{Sensitivity analysis of learning rate $\alpha$ and discount "
             r"factor $\gamma$ for Q-Learning on Setting 3 (30 seeds, 600 episodes, "
             r"$r_{\text{rescue}}=0$).}",
             r"\label{tab:sens}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Parameter} & \textbf{Success Rate} & \textbf{Return} & "
             r"\textbf{Discounted} & \textbf{Steps} \\",
             r"\hline"]
    for p in ("0.05", "0.1", "0.2", "0.4"):
        m = sens["alpha"][p]
        lines.append(f"$\\alpha={float(p):.2f}$ & {m['success_rate']:.2f} $\\pm$ "
                     f"{m['success_rate_std']:.2f} & {m['mean_reward']:.1f} $\\pm$ "
                     f"{m['mean_reward_std']:.1f} & "
                     f"{m['mean_discounted_reward']:.2f} $\\pm$ "
                     f"{m['mean_discounted_reward_std']:.2f} & "
                     f"{m['mean_steps']:.1f} $\\pm$ {m['mean_steps_std']:.1f} \\\\")
    lines.append(r"\hline")
    for p in ("0.8", "0.9", "0.95", "0.99"):
        m = sens["gamma"][p]
        lines.append(f"$\\gamma={float(p):.2f}$ & {m['success_rate']:.2f} $\\pm$ "
                     f"{m['success_rate_std']:.2f} & {m['mean_reward']:.1f} $\\pm$ "
                     f"{m['mean_reward_std']:.1f} & "
                     f"{m['mean_discounted_reward']:.2f} $\\pm$ "
                     f"{m['mean_discounted_reward_std']:.2f} & "
                     f"{m['mean_steps']:.1f} $\\pm$ {m['mean_steps_std']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_en_sens.tex", lines)


def en_sens_reward(sens):
    lines = [r"\begin{table}[t]",
             r"\caption{Sensitivity to reward formulation for Q-Learning on Setting 3 "
             r"(30 seeds, 600 episodes).}",
             r"\label{tab:sens-reward}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Formulation} & \textbf{Success} & \textbf{Return} & "
             r"\textbf{Steps} & \textbf{Rescues} \\",
             r"\hline"]
    for key in ("r=0", "r=+10", "r=-10", "box-focused", "deadlock-heavy"):
        m = sens["reward"][key]
        lines.append(f"{REWARD_LABEL_EN[key]} & {m['success_rate']:.2f} $\\pm$ "
                     f"{m['success_rate_std']:.2f} & {m['mean_reward']:.1f} & "
                     f"{m['mean_steps']:.1f} & {m['mean_rescue_uses']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_en_sens-reward.tex", lines)


def en_epsilon(data):
    lines = [r"\begin{table*}[t]",
             r"\caption{Exploration schedule sensitivity: comparing final greedy "
             r"performance under different $\varepsilon$-decay rates on Setting 3 "
             r"(30 seeds, 1000 episodes, $r_{\text{rescue}}=0$).}",
             r"\label{tab:epsilon}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccc}", r"\hline",
             r"\textbf{Decay Rate} & \textbf{Floor Episode} & \textbf{Final Training "
             r"$\varepsilon$} & \textbf{Q-Learning} & \textbf{SARSA} & "
             r"\textbf{Expected SARSA} & \textbf{Monte Carlo Control} \\",
             r"\hline"]
    for decay in EPS_DECAYS:
        block = data["decays"][decay]
        cells = []
        for algo in TD:
            m = block[algo]
            cells.append(f"{m['success_rate']:.3f} $\\pm$ "
                         f"{m['success_rate_std']:.3f}")
        final = max(0.02, float(decay) ** data["episodes"])
        lines.append(f"{float(decay):.4f} & {EPS_FLOOR_EN[decay]} & {final:.4f} & " +
                     " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_en_epsilon.tex", lines)


def main():
    os.makedirs(REP, exist_ok=True)

    old = load("study_results.json")
    main_all = load("study_all.json")
    sens = load("sens_r0.json")
    eps = load("eps_r0.json")
    rows = rescue_rows()
    st = rescue_stats(rows)

    # Ban tieng Viet.
    vi_setting(old["dp"])
    vi_placement()
    vi_vipi(old["dp"])
    vi_dp(main_all["dp"])
    vi_td_perf(main_all["td_summary"])
    vi_td_behavior(main_all["td_summary"])
    vi_ablation(main_all)
    vi_sens(sens)
    vi_epsilon(eps)
    vi_rescue_train(st, rows)

    # Ban tieng Anh.
    en_setting(old["dp"])
    en_placement()
    en_dp(main_all["dp"])
    en_vipi(old["dp"])
    en_td_perf(main_all["td_summary"])
    en_td_behavior(main_all["td_summary"])
    en_ablation(main_all)
    en_rescue_train(st, rows, eps)
    en_sens(sens)
    en_sens_reward(sens)
    en_epsilon(eps)

    print("Da sinh xong cac bang trong", REP)


if __name__ == "__main__":
    main()
