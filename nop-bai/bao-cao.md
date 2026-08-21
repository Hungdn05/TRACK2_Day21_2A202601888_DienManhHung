# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

|              |                                                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Họ và tên | Điền Mạnh Hùng                                                                                                             |
| MSSV         | 2A202601888                                                                                                                    |
| Lớp / Khóa | K3B                                                                                                                            |
| Repo GitHub  | [github.com/Hungdn05/TRACK2_Day21_2A202601888_DienManhHung](https://github.com/Hungdn05/TRACK2_Day21_2A202601888_DienManhHung)  |
| Ngày nộp   | 2026-08-21                                                                                                                     |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

Kết quả từ 4 lần chạy thí nghiệm trên MLflow:

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score         | accuracy |
| ---------- | ------------ | ------------- | --------- | ---------------- | -------- |
| 1          | 100          | 0.1           | 3         | 0.7109           | 0.8780   |
| 2          | 50           | 0.05          | 2         | 0.6051           | 0.8460   |
| 3          | 200          | 0.15          | 5         | 0.7005           | 0.8700   |
| 4          | 150          | 0.1           | 4         | **0.7156** | 0.8760   |

**Bộ siêu tham số đã chọn:** `n_estimators=150`, `learning_rate=0.1`, `max_depth=4`.

**Lý do:** Bộ tham số này đạt F1 score cao nhất (0.7156) trong 4 lần thí nghiệm, vượt ngưỡng 0.65 của lab. Quan sát thấy accuracy cao nhất (0.8780) lại ở thí nghiệm 1 với F1 = 0.7109, trong khi thí nghiệm 4 có F1 cao hơn nhưng accuracy thấp hơn. Điều này chứng minh accuracy không phản ánh đúng chất lượng mô hình phân loại trên dữ liệu mất cân bằng. Giữa n_estimators và learning_rate có quan hệ đánh đổi: thí nghiệm 2 với n_estimators=50, lr=0.05 cho F1 thấp nhất (0.6051), trong khi thí nghiệm 4 với cân bằng hơn n_estimators=150, lr=0.1 cho kết quả tốt nhất.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng nghiêm trọng: chỉ 24.8% số mẫu thuộc lớp thu nhập cao (>50K). Hệ quả là một mô hình "luôn trả lời thu nhập thấp" cho mọi đầu vào sẽ đạt accuracy = 0.752, trông có vẻ tốt nhưng thực tế hoàn toàn vô dụng vì không phát hiện được trường hợp thu nhập cao nào. F1 score của lớp dương đo khả năng cân bằng giữa precision (độ chính xác khi dự đoán dương) và recall (khả năng phát hiện tất cả trường hợp dương), phản ánh đúng chất lượng mô hình trong bài toán mất cân bằng. Không nên dùng average="weighted" hay average="macro" vì các giá trị đó bị lớp đa số kéo lên cao, che giấu việc mô hình bỏ sót phần lớn trường hợp thu nhập cao.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn                                                                                                | Nguyên nhân                                                         | Cách giải quyết                                                   |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Cài đặt dependencies thất bại do Python 3.14 không tương thích với phiên bản numpy cố định | requirements.txt yêu cầu numpy==2.0.0rc1 không có cho Python 3.14 | Sửa requirements.txt thành version flexible (dùng >= thay vì ==) |
| F1 score thí nghiệm 2 thấp hơn ngưỡng 0.65                                                          | n_estimators=50 và learning_rate=0.05 quá yếu cho mô hình        | Tăng n_estimators và learning_rate để cải thiện                |
| Quan sát thấy accuracy và F1 không tương quan tuyến tính                                          | Dữ liệu mất cân bằng 75/25 khiến accuracy gây hiểu lầm       | Tập trung theo dõi F1 score thay vì accuracy                      |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

|                                  | f1_score | accuracy |
| -------------------------------- | -------- | -------- |
| Bước 2 (chỉ `train_batch1`)  | 0.7156   | 0.8760   |
| Bước 3 (thêm `train_batch2`) | 0.7248   | 0.8800   |

**Nhận xét:** Khi tăng dữ liệu huấn luyện từ 22.361 lên 44.722 mẫu, F1 tăng 0.0092 và
accuracy tăng 0.0040. Hai batch được lấy ngẫu nhiên từ cùng nguồn nên có phân phối tương tự;
mức tăng nhỏ này là hợp lý và không nên hiểu rằng thêm dữ liệu luôn làm chỉ số tăng.
