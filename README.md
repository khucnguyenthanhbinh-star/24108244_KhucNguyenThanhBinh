# Sokoban Hai Thùng với Cơ Chế Ô Cứu Hộ (Two-Box Sokoban with Rescue Tile)

Mã nguồn bài tập lớn môn **Học tăng cường** — **Đề tài 16: Sokoban hai thùng và ô cứu hộ**.

---

## 1. Giới thiệu đề tài

Bài toán yêu cầu điều khiển tác tử đẩy hai thùng hàng vào đúng hai ô đích trên bản đồ lưới ô vuông. Điểm phức tạp của Sokoban là tính chất hành động bất khả đảo (deadlock): một cú đẩy sai vào góc tường sẽ làm bài toán vĩnh viễn không thể giải được.

Môi trường tích hợp cơ chế **ô cứu hộ dùng một lần**:
- Tác tử chỉ kích hoạt được cơ chế nếu chủ động ghé qua ô cứu hộ trước khi xảy ra bế tắc.
- Khi đẩy thùng vào thế kẹt, hệ thống hoàn tác cú đẩy vừa thực hiện, đưa người chơi và thùng về vị trí liền trước.
- Trạng thái hai thùng được chuẩn hóa theo thứ tự từ điển ($b_1 \le b_2$) để triệt tiêu 50% không gian trạng thái đối xứng.

Toàn bộ bài toán được giải quyết bằng **sáu thuật toán học tăng cường dạng bảng (Tabular RL)**, không sử dụng mạng nơ-ron xấp xỉ hàm hay thư viện bên ngoài (gymnasium/pygame).

---

## 2. Cài đặt môi trường

Mã nguồn được viết bằng Python chuẩn và chỉ phụ thuộc vào ba thư viện tính toán cơ bản:

```bash
pip install -r requirements.txt
```

*(Yêu cầu: `numpy`, `matplotlib`, `pillow`)*

---

## 3. Hướng dẫn chạy chương trình

### Giao diện Menu tương tác
Chạy chương trình chính qua giao diện dòng lệnh:

```bash
python main.py
```

Hệ thống cung cấp menu chức năng trực quan:
- `[1]` Huấn luyện tất cả thuật toán trên 4 cấu hình bản đồ
- `[2]` Xuất bảng so sánh kết quả thực nghiệm
- `[3]` Phân tích độ nhạy tham số ($\alpha$, $\gamma$, hàm phần thưởng)
- `[4]` Sinh hình ảnh nhiệt đồ giá trị và đường đi chính sách
- `[5]` Demo so sánh sáu thuật toán chạy đua trên một bản đồ
- `[6]` Chạy thử mô phỏng từng bước (step-by-step) của một thuật toán
- `[7]` Thực thi toàn bộ bài kiểm thử tự động
- `[0]` Thoát

### Chạy trực tiếp bằng tham số dòng lệnh (CLI Arguments)
```bash
# Huấn luyện nhanh 6 thuật toán:
python main.py --train --episodes 1000 --seeds 30

# In bảng so sánh kết quả:
python main.py --compare

# Chạy demo 6 thuật toán trên Setting 1:
python main.py --demo-all --setting 1

# Chạy trực quan một episode theo chính sách tham lam:
python main.py --demo --setting 3
```

---

## 4. Cấu trúc mã nguồn

```
RL_Sokoban_Source_Code/
├── sokoban/                   # Gói mã nguồn cốt lõi
│   ├── base_env.py            # Giao diện trừu tượng MDPEnv, Discrete
│   ├── env.py                 # Môi trường Sokoban 2 thùng, StateEncoder, DeadlockDetector
│   ├── mdp.py                 # Đồ thị chuyển trạng thái tất định cho Dynamic Programming
│   ├── experiments.py         # Quy trình huấn luyện, đánh giá, kiểm thử độ nhạy
│   ├── viz.py                 # Công cụ trực quan hóa bàn cờ, nhiệt đồ V(s) và mũi tên policy
│   └── agents/                # Thư mục cài đặt 6 thuật toán Tabular RL
│       ├── base.py            # Lớp cơ sở TabularAgent
│       ├── dp.py              # Value Iteration, Policy Iteration
│       └── td.py              # Q-Learning, SARSA, Expected SARSA, Monte Carlo Control
├── maps/                      # Bốn cấu hình bản đồ thực nghiệm
│   ├── map_setting1_small.txt      # Setting 1: Bản đồ nhỏ cơ sở (5x6)
│   ├── map_setting2_corridor.txt   # Setting 2: Hành lang hẹp (6x7)
│   ├── map_setting3_rescue.txt     # Setting 3: Tường ngăn phân vùng (6x7)
│   └── map_setting4_advanced.txt   # Setting 4: Mở rộng, hốc cứu hộ riêng (6x8)
├── tests/                     # Bộ kiểm thử tự động
│   ├── test_env.py            # Kiểm thử tính đúng đắn của môi trường, deadlock, rescue
│   └── test_algorithms.py     # Kiểm thử sự hội tụ của 6 thuật toán
├── saved_models/              # Dữ liệu thực nghiệm lưu trữ dạng JSON (30 seeds)
├── main.py                    # Điểm khởi chạy chương trình chính
├── requirements.txt           # Danh mục thư viện phụ thuộc
└── README.md                  # Tài liệu hướng dẫn sử dụng
```

---

## 5. Kết quả thực nghiệm tóm tắt (30 Random Seeds)

| Cấu hình | Value Iteration | Policy Iteration | Q-Learning | SARSA | Expected SARSA | Monte Carlo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting 1** ($5\times 6$) | 1.00 (12.0 bước) | 1.00 (12.0 bước) | 1.00 (12.1 bước) | 1.00 (12.0 bước) | 1.00 (12.0 bước) | 1.00 (12.4 bước) |
| **Setting 2** ($6\times 7$) | 1.00 (12.0 bước) | 1.00 (12.0 bước) | 1.00 (13.6 bước) | 1.00 (13.0 bước) | 1.00 (13.0 bước) | 0.97 (17.3 bước) |
| **Setting 3** ($6\times 7$) | 1.00 (15.0 bước) | 1.00 (15.0 bước) | 1.00 (22.4 bước) | 1.00 (24.7 bước) | 1.00 (24.9 bước) | 0.97 (20.6 bước) |
| **Setting 4** ($6\times 8$) | 1.00 (12.0 bước) | 1.00 (12.0 bước) | 1.00 (12.9 bước) | 1.00 (12.1 bước) | 1.00 (12.1 bước) | 0.93 (20.9 bước) |

*Ghi chú: Kết quả ghi nhận tại cấu hình phần thưởng chuẩn ($r_{\text{rescue}}=0$). Tỉ lệ thành công của 3 thuật toán TD đạt 1.00 tuyệt đối trên cả 4 bản đồ.*

---

## 6. Chạy bộ kiểm thử tự động

Dự án bao gồm 36 ca kiểm thử tự động xác nhận tính đúng đắn của logic chuyển trạng thái, hàm phần thưởng, tính Markov và sự hội tụ của thuật toán:

```bash
python -m pytest tests/ -v
```

*(Kết quả đạt: 36/36 passed)*
