"""
Chạy toàn bộ thực nghiệm của đồ án.

    python tools/run_study.py --seeds 30 --episodes 1000

Kết quả được lưu vào saved_models/study_results.json.
"""

import argparse
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sokoban import experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Thực nghiệm đầy đủ - Đề tài 16")
    parser.add_argument("--seeds", type=int, default=30, help="số seed cho mỗi cấu hình")
    parser.add_argument("--episodes", type=int, default=1000, help="số episode huấn luyện")
    parser.add_argument("--workers", type=int, default=0, help="số tiến trình (0 = tất cả lõi)")
    parser.add_argument("--out", default="study_results.json", help="tên file kết quả")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="bỏ qua phân tích độ nhạy tham số")
    args = parser.parse_args()

    experiments.run_full_study(episodes=args.episodes, seeds=args.seeds,
                               workers=args.workers, out_file=args.out,
                               skip_sensitivity=args.skip_sensitivity)


if __name__ == "__main__":
    main()
