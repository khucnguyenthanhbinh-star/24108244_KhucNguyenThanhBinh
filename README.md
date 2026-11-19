# Sokoban Hai Thùng với Cơ Chế Ô Cứu Hộ (Two-Box Sokoban with Rescue Tile)

Báo cáo bài tập lớn môn **Học tăng cường** — **Đề tài 16**.

- **Sinh viên thực hiện:** Khúc Nguyễn Thanh Bình (MSSV: 24108244)
- **Giảng viên hướng dẫn:** ThS. Vũ Hoàng Diệu
- **Đơn vị:** Khoa Điện – Điện tử, Trường Kỹ thuật, Trường Đại học Phenikaa

---

## 1. Giới thiệu bài toán

Bài toán yêu cầu điều khiển tác tử đẩy hai thùng hàng vào đúng hai ô đích trên bản đồ lưới ô vuông. Điểm phức tạp của Sokoban là tính chất hành động một chiều không thể đảo ngược (deadlock): một cú đẩy sai vào góc tường sẽ làm bài toán vĩnh viễn không thể giải được.

Môi trường tích hợp cơ chế **ô cứu hộ dùng một lần**:
- Tác tử chỉ kích hoạt được cơ chế nếu chủ động ghé qua ô cứu hộ trước khi xảy ra bế tắc.
- Khi đẩy thùng vào thế kẹt, hệ thống hoàn tác cú đẩy vừa thực hiện, đưa người chơi và thùng về vị trí liền trước.
- Trạng thái hai thùng được chuẩn hóa theo thứ tự từ điển ($b_1 \le b_2$) để triệt tiêu 50% không gian trạng thái đối xứng.

Toàn bộ bài toán được giải quyết bằng **sáu thuật toán học tăng cường dạng bảng (Tabular RL)**:
1. **Quy hoạch động (Dynamic Programming):** Value Iteration, Policy Iteration.
2. **Học sai phân thời gian (Temporal Difference):** Q-Learning, SARSA, Expected SARSA.
3. **Phương pháp Monte Carlo:** Last-Visit Monte Carlo Control.

---

## 2. Cài đặt môi trường

Chương trình chỉ sử dụng Python thuần và các thư viện tính toán cơ bản:

```bash
pip install -r requirements.txt
```

*(Yêu cầu: `numpy`, `matplotlib`, `pillow`)*

---

## 3. Chạy mô phỏng (Demo)

Khởi chạy chương trình mô phỏng từng bước:

```bash
python main.py
```

Chương trình cung cấp menu tương tác trực quan:
- Lựa chọn 1 trong 6 thuật toán RL.
- Lựa chọn 1 trong 4 cấu hình bản đồ.
- Tự động nạp trọng số đã huấn luyện sẵn từ thư mục `saved_models/` (hoặc tự học nhanh nếu chưa có) và hiển thị trực quan từng bước đi của nhân vật đến khi giải xong ván cờ.

---

## 4. Cấu trúc mã nguồn

```
RL_Sokoban_Source_Code/
├── sokoban/                   # Thư mục mã nguồn cốt lõi
│   ├── base_env.py            # Giao diện môi trường trừu tượng Discrete, MDPEnv
│   ├── env.py                 # Môi trường Sokoban 2 thùng, cơ chế hoàn tác và phát hiện bế tắc
│   ├── mdp.py                 # Xây dựng ma trận chuyển trạng thái cho Quy hoạch động
│   ├── experiments.py         # Quy trình huấn luyện, đánh giá và xây dựng agent
│   ├── viz.py                 # Công cụ trực quan hóa bàn cờ, xuất hình và chạy ván cờ
│   └── agents/                # Cài đặt sáu thuật toán Tabular RL
│       ├── base.py            # Lớp cơ sở TabularAgent (quản lý bảng Q, epsilon-greedy)
│       ├── dp.py              # Value Iteration, Policy Iteration
│       └── td.py              # Q-Learning, SARSA, Expected SARSA, Monte Carlo
├── maps/                      # Bốn cấu hình bản đồ thực nghiệm
│   ├── map_setting1_small.txt      # Bản đồ 1: Cơ sở (5x6)
│   ├── map_setting2_corridor.txt   # Bản đồ 2: Hành lang hẹp (6x7)
│   ├── map_setting3_rescue.txt     # Bản đồ 3: Phân vùng cứu hộ (6x7)
│   └── map_setting4_advanced.txt   # Bản đồ 4: Mở rộng (6x8)
├── saved_models/              # 24 mô hình trọng số đã huấn luyện sẵn (.npz) cho 6 thuật toán trên 4 map
├── main.py                    # Giao diện dòng lệnh chạy mô phỏng trực quan
├── requirements.txt           # Danh mục thư viện phụ thuộc
└── README.md                  # Hướng dẫn sử dụng
```

---

## 5. Tóm tắt kết quả thực nghiệm (30 Random Seeds)

Kết quả đánh giá trên 4 cấu hình bản đồ tại mức thưởng cứu hộ cơ sở ($r_{\text{rescue}} = 0$):

| Bản đồ | Kích thước | Dynamic Programming (VI / PI) | Q-Learning | SARSA | Expected SARSA | Monte Carlo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bản đồ 1** | 5x6 | 1.00 (12.0 bước) | 1.00 (12.1 bước) | 1.00 (12.0 bước) | 1.00 (12.0 bước) | 1.00 (12.4 bước) |
| **Bản đồ 2** | 6x7 | 1.00 (12.0 bước) | 1.00 (13.6 bước) | 1.00 (13.0 bước) | 1.00 (13.0 bước) | 0.97 (17.3 bước) |
| **Bản đồ 3** | 6x7 | 1.00 (15.0 bước) | 1.00 (22.4 bước) | 1.00 (24.7 bước) | 1.00 (24.9 bước) | 0.97 (20.6 bước) |
| **Bản đồ 4** | 6x8 | 1.00 (12.0 bước) | 1.00 (12.9 bước) | 1.00 (12.1 bước) | 1.00 (12.1 bước) | 0.93 (20.9 bước) |

*Ghi chú: Giá trị trong ngoặc biểu thị số bước đi trung bình của tác tử khi hoàn thành bài toán. Các thuật toán TD đạt tỉ lệ giải quyết thành công tuyệt đối (1.00) trên toàn bộ các bản đồ.*
