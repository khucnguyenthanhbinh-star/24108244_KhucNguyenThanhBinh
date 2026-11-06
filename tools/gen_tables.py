# -*- coding: utf-8 -*-
"""Sinh toan bo bang so lieu cua bao cao.

    python tools/gen_tables.py

Doc cac tep trong saved_models/ va ghi ra report/:

    tables_vi_setting.tex       bon cau hinh ban do
    tables_vi_placement.tex     danh gia vi tri o cuu ho
    tables_vi_dp.tex            ket qua quy hoach dong
    tables_vi_vipi.tex          so sanh Value Iteration va Policy Iteration
    tables_vi_td_perf.tex       hieu nang bon thuat toan TD
    tables_vi_td_behavior.tex   hanh vi su dung o cuu ho
    tables_vi_ablation.tex      so sanh co o va khong o
    tables_vi_rescue_train.tex  toc do hoc khi co o cuu ho
    tables_vi_sens.tex          do nhay alpha va gamma
    tables_vi_sens-reward.tex   do nhay cau hinh phan thuong
    tables_vi_epsilon.tex       quet he so giam epsilon

Bon bang (setting, placement, vipi, epsilon) duoc dung ngay trong tep nay.
Cac bang con lai goi lai gen_all_maps, gen_r0_report va gen_rescue_all de
chi co mot duong sinh so lieu duy nhat.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
REP = os.path.join(ROOT, "report")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import gen_all_maps
import gen_r0_report
import gen_rescue_all

SET_ORDER = ["setting1", "setting2", "setting3", "setting4"]
SLAB = {"setting1": "Setting 1", "setting2": "Setting 2",
        "setting3": "Setting 3", "setting4": "Setting 4"}
TD = ["Q-Learning", "SARSA", "Expected SARSA", "Monte Carlo Control"]

# Mo ta va kich thuoc cua bon ban do (khong doi giua cac lan chay).
SETTING_ROWS = [
    ("setting1", "Bản đồ nhỏ cơ sở", r"$5\times6$", r"81\,000"),
    ("setting2", "Hành lang hẹp", r"$6\times7$", r"222\,264"),
    ("setting3", "Tường ngăn phân vùng", r"$6\times7$", r"222\,264"),
    ("setting4", "Mở rộng, hốc cứu hộ riêng", r"$6\times8$", r"331\,776"),
]

PLACEMENT_ROWS = [
    (r"Cách Start 2 bước", "Có", "Thấp", r"Lệch khỏi đường đi tối ưu"),
    (r"Ngay sát Start", "Có", r"$\approx 0$", r"Trở thành lượt thử miễn phí"),
    (r"Ngẫu nhiên", "Có", "Đổi", r"Có thể nằm ngoài vùng học"),
    (r"Kích hoạt sẵn ($u=1$)", "Không", "0", r"Bỏ điều kiện kích hoạt trước"),
]

EPS_DECAYS = ["0.9", "0.99", "0.995", "0.9992"]


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


def table_setting(dp):
    lines = [r"\begin{table*}[t]",
             r"\caption{Bốn cấu hình bản đồ và kích thước không gian trạng thái.}",
             r"\label{tab:setting}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrrrr}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{Đặc điểm} & \textbf{Lưới} & "
             r"\textbf{Số state tiềm năng} & \textbf{State tới được} & "
             r"\textbf{Trạng thái kết thúc} \\"]
    for key, desc, grid, potential in SETTING_ROWS:
        d = dp.get(key, {})
        n_states = d.get("n_states")
        n_term = d.get("n_terminal_states")
        n_states = thousands(n_states) if n_states is not None else "-"
        n_term = thousands(n_term) if n_term is not None else "-"
        lines.append(f"{SLAB[key]} & {desc} & {grid} & {potential} & "
                     f"{n_states} & {n_term} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write("tables_vi_setting.tex", lines)


def table_placement():
    lines = [r"\begin{table}[t]",
             r"\caption{Đánh giá định tính các phương án thiết kế vị trí ô cứu hộ.}",
             r"\label{tab:placement}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lccl}", r"\hline",
             r"\textbf{Phương án vị trí} & \textbf{Đúng đề} & \textbf{Chi phí} & "
             r"\textbf{Đặc điểm hành vi / Rủi ro} \\",
             r"\hline"]
    for row in PLACEMENT_ROWS:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_placement.tex", lines, newline="\n")


def table_vipi(dp):
    lines = [r"\begin{table}[t]",
             r"\caption{So sánh giá trị và chính sách giữa Value Iteration và Policy Iteration.}",
             r"\label{tab:vipi}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccc}", r"\hline",
             r"\textbf{Cấu hình} & \textbf{$\max|V_{\text{VI}}-V_{\text{PI}}|$} & "
             r"\textbf{Trùng khớp chính sách} & \textbf{Số state} \\",
             r"\hline"]
    for key in SET_ORDER:
        d = dp[key]
        n_states = thousands(d["n_states"])
        lines.append(f"{SLAB[key]} & {sci(d['max_abs_value_difference'])} & "
                     f"{d['same_policy_states']} / {n_states} & {n_states} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    write("tables_vi_vipi.tex", lines)


def table_td_behavior(summary):
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


def table_epsilon(data):
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


def main():
    os.makedirs(REP, exist_ok=True)
    os.makedirs(os.path.join(REP, "figures"), exist_ok=True)

    # Bang mo ta va bang lay tu ket qua quy hoach dong.
    table_setting(load("study_results.json")["dp"])
    table_placement()
    table_vipi(load("study_results.json")["dp"])

    # Hieu nang, hanh vi, ablation va quy hoach dong tren bon ban do.
    gen_all_maps.main()
    table_td_behavior(load("study_all.json")["td_summary"])

    # Do nhay tham so tren 30 seed.
    gen_r0_report.sens_table(load("sens_r0.json"))

    # Quet he so giam epsilon.
    table_epsilon(load("eps_r0.json"))

    # O cuu ho va toc do hoc (doc rescue_probe.json + rescue_probe_rest.json).
    gen_rescue_all.main()

    print("Da sinh xong cac bang trong", REP)


if __name__ == "__main__":
    main()
