"""Giao diện dòng lệnh thực thi mô phỏng và chạy thử các thuật toán Sokoban RL."""

import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sokoban import experiments, viz

ALGO_MAP = {
    "1": ("Q-Learning", "q_learning"),
    "2": ("SARSA", "sarsa"),
    "3": ("Expected SARSA", "expected_sarsa"),
    "4": ("Monte Carlo Control", "monte_carlo"),
    "5": ("Value Iteration", "vi"),
    "6": ("Policy Iteration", "pi"),
}

SETTING_MAP = {
    "1": ("setting1", "Bản đồ 1: Cơ sở (5x6)"),
    "2": ("setting2", "Bản đồ 2: Hành lang hẹp (6x7)"),
    "3": ("setting3", "Bản đồ 3: Phân vùng cứu hộ (6x7)"),
    "4": ("setting4", "Bản đồ 4: Mở rộng (6x8)"),
}


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_header() -> None:
    print("=" * 74)
    print("   SOKOBAN HAI THÙNG VỚI CƠ CHẾ Ô CỨU HỘ (REINFORCEMENT LEARNING)")
    print("=" * 74)


def run_demo(algo_choice: str, setting_choice: str) -> None:
    algo_name, algo_slug = ALGO_MAP.get(algo_choice, ("Q-Learning", "q_learning"))
    setting_key, setting_desc = SETTING_MAP.get(setting_choice, ("setting1", "Bản đồ 1: Cơ sở (5x6)"))

    clear_screen()
    print_header()
    print(f"\n   -> Thuật toán : {algo_name}")
    print(f"   -> Cấu hình   : {setting_desc}\n")

    env = experiments.make_env(setting_key)
    agent = experiments.build_agent(algo_name, env, seed=0)

    # Nạp mô hình đã lưu sẵn hoặc huấn luyện nhanh nếu chưa có
    model_path = os.path.join(experiments.MODELS_DIR, f"{algo_slug}_{setting_key}.npz")
    if os.path.exists(model_path):
        print(f"   [+] Đang nạp mô hình huấn luyện sẵn từ: {model_path} ...")
        agent.load(model_path)
    else:
        print("   [!] Chưa có file trọng số sẵn, đang huấn luyện nhanh...")
        t0 = time.time()
        if algo_name in ["Value Iteration", "Policy Iteration"]:
            agent.train()
        else:
            agent.train(num_episodes=600, eval_interval=600)
        print(f"   [+] Huấn luyện hoàn tất trong {time.time() - t0:.1f}s.")

    # Mô phỏng từng bước
    res = viz.run_episode(env, agent)
    frames = res["frames"]

    print("\n" + "=" * 74)
    print("   MÔ PHỎNG TỪNG BƯỚC ĐI CỦA TÁC TỬ (GREEDY POLICY)")
    print("=" * 74)

    for i, grid in enumerate(frames):
        print(f"\n[Bước {i}]")
        for row in grid:
            print("   " + "".join(row))

    print("\n" + "=" * 74)
    print(f"   KẾT QUẢ MÔ PHỎNG:")
    print(f"   - Trạng thái giải : {'Thành công (Solved)' if res['solved'] else ('Bế tắc (Deadlock)' if res['deadlock'] else 'Hết bước')}")
    print(f"   - Tổng số bước đi : {res['steps']}")
    print(f"   - Tổng phần thưởng: {res['reward']:.2f}")
    print("=" * 74)


def main() -> None:
    while True:
        clear_screen()
        print_header()
        print("""
   CHỌN THUẬT TOÁN ĐỂ MÔ PHỎNG:
   [1]  Q-Learning
   [2]  SARSA
   [3]  Expected SARSA
   [4]  Monte Carlo Control (Last-Visit)
   [5]  Value Iteration (Quy hoạch động)
   [6]  Policy Iteration (Quy hoạch động)
   [0]  Thoát
        """)
        algo_choice = input("   Chọn thuật toán [0-6]: ").strip()
        if algo_choice == "0":
            print("\n   Tạm biệt!\n")
            break
        if algo_choice not in ALGO_MAP:
            continue

        print("""
   CHỌN BẢN ĐỒ THỬ NGHIỆM:
   [1]  Bản đồ 1: Cơ sở (5x6)
   [2]  Bản đồ 2: Hành lang hẹp (6x7)
   [3]  Bản đồ 3: Phân vùng cứu hộ (6x7)
   [4]  Bản đồ 4: Mở rộng (6x8)
        """)
        setting_choice = input("   Chọn bản đồ [1-4] (mặc định 1): ").strip() or "1"
        if setting_choice not in SETTING_MAP:
            setting_choice = "1"

        run_demo(algo_choice, setting_choice)
        input("\n   Nhấn Enter để tiếp tục thử nghiệm khác...")


if __name__ == "__main__":
    main()
