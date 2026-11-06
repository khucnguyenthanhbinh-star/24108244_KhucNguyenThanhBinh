"""
Chương trình chính của đồ án học tăng cường - đề tài 16.

Chạy: python main.py
"""

import argparse
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sokoban import experiments


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_header() -> None:
    print("=" * 74)
    print("   HỌC TĂNG CƯỜNG - ĐỀ TÀI 16")
    print("   SOKOBAN HAI THÙNG VỚI CƠ CHẾ Ô CỨU HỘ")
    print("=" * 74)


def pause() -> None:
    input("\n   Nhấn Enter để quay lại menu...")


def main_menu() -> None:
    while True:
        clear_screen()
        print_header()
        print("""
   [1]  Huấn luyện tất cả thuật toán trên 4 setting
   [2]  Bảng so sánh kết quả giữa các thuật toán
   [3]  Phân tích độ nhạy theo alpha, gamma và phần thưởng
   [4]  Sinh hình ảnh, nhiệt đồ giá trị và file GIF
   [5]  Demo sáu thuật toán trên một setting
   [6]  Chạy thử một episode bằng chính sách đã học
   [7]  Chạy bộ kiểm thử
   [0]  Thoát
        """)
        choice = input("   Chọn [0-7]: ").strip()

        if choice == "1":
            clear_screen()
            print_header()
            experiments.run_all_training()
            pause()
        elif choice == "2":
            clear_screen()
            print_header()
            experiments.evaluate_comparison()
            pause()
        elif choice == "3":
            clear_screen()
            print_header()
            experiments.sensitivity_analysis()
            pause()
        elif choice == "4":
            clear_screen()
            print_header()
            experiments.make_figures()
            pause()
        elif choice == "5":
            clear_screen()
            print_header()
            print("   Setting: 1) nhỏ  2) hành lang  3) có ô cứu hộ  4) nâng cao")
            s = input("   Chọn setting [1-4] (mặc định 1): ").strip() or "1"
            experiments.demo_all_algorithms(setting_key=f"setting{s}")
            pause()
        elif choice == "6":
            clear_screen()
            print_header()
            print("   Thuật toán: 1) Q-Learning  2) SARSA  3) Expected SARSA  4) Monte Carlo")
            a = input("   Chọn thuật toán [1-4] (mặc định 1): ").strip() or "1"
            algo = {"1": "Q-Learning", "2": "SARSA", "3": "Expected SARSA",
                    "4": "Monte Carlo Control"}.get(a, "Q-Learning")
            print("   Setting: 1) nhỏ  2) hành lang  3) có ô cứu hộ  4) nâng cao")
            s = input("   Chọn setting [1-4] (mặc định 1): ").strip() or "1"
            experiments.demo_episode(algo=algo, setting_key=f"setting{s}")
            pause()
        elif choice == "7":
            clear_screen()
            print_header()
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                                    cwd=os.path.dirname(os.path.abspath(__file__)))
            print("   Kết quả:", "đạt" if result.returncode == 0 else "có lỗi")
            pause()
        elif choice == "0":
            print("\n   Cảm ơn bạn đã sử dụng chương trình.\n")
            break
        else:
            print("   Lựa chọn không hợp lệ.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sokoban hai thùng - đề tài 16")
    parser.add_argument("--train", action="store_true", help="huấn luyện toàn bộ")
    parser.add_argument("--compare", action="store_true", help="in bảng so sánh")
    parser.add_argument("--sensitivity", action="store_true", help="phân tích độ nhạy")
    parser.add_argument("--figures", action="store_true", help="sinh hình ảnh và GIF")
    parser.add_argument("--demo", action="store_true", help="chạy thử một episode")
    parser.add_argument("--demo-all", action="store_true",
                        help="chạy thử sáu thuật toán trên một setting")
    parser.add_argument("--setting", type=int, default=1, help="setting 1 đến 4")
    parser.add_argument("--episodes", type=int, default=experiments.DEFAULT_EPISODES)
    parser.add_argument("--seeds", type=int, default=experiments.DEFAULT_SEEDS)
    args = parser.parse_args()

    if args.train:
        experiments.run_all_training(episodes=args.episodes, seeds=args.seeds)
    elif args.compare:
        experiments.evaluate_comparison(episodes=args.episodes, seeds=args.seeds)
    elif args.sensitivity:
        experiments.sensitivity_analysis(episodes=min(args.episodes, 600))
    elif args.figures:
        experiments.make_figures(episodes=args.episodes)
    elif args.demo:
        experiments.demo_episode(episodes=args.episodes)
    elif args.demo_all:
        experiments.demo_all_algorithms(setting_key=f"setting{args.setting}",
                                        episodes=args.episodes)
    else:
        main_menu()


if __name__ == "__main__":
    main()
