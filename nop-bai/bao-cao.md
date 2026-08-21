# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

**Họ tên:** Điền Mạnh Hùng · **MSSV:** 2A202601888 · **Lớp:** K3B ·
**Repo:** [TRACK2_Day21_2A202601888_DienManhHung](https://github.com/Hungdn05/TRACK2_Day21_2A202601888_DienManhHung) · **Ngày:** 21/08/2026

## 1. Thực nghiệm và lựa chọn mô hình

| Run | n_estimators | learning_rate | max_depth | F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.10 | 3 | 0.7109 | **0.8780** |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.15 | 5 | 0.7005 | 0.8700 |
| 4 | 150 | 0.10 | 4 | **0.7156** | 0.8760 |

Chọn run 4 vì có F1 cao nhất và vượt gate 0.65. Run 1 có accuracy cao hơn nhưng F1 thấp
hơn, cho thấy accuracy không đủ để chọn model trên dữ liệu lệch lớp. Chỉ 24,8% mẫu là lớp
thu nhập cao; model luôn đoán lớp thấp vẫn đạt accuracy 0.752 nhưng F1 lớp dương bằng 0.
Vì vậy gate dùng F1 lớp dương để cân bằng precision và recall, không dùng weighted/macro F1.

## 2. Khó khăn và kết quả huấn luyện liên tục

| Khó khăn | Cách xử lý |
|---|---|
| Dependency từng lệch giữa CI và VM | Khóa `scikit-learn==1.7.2`, đồng nhất nơi train/serve |
| Credential DVC và bucket artifact khác cơ chế | Dùng secret riêng, credential chỉ tồn tại trên runner/VM |
| Model kém có thể ghi đè production trước gate | Upload candidate; chỉ promote sau minimum gate và regression gate |

| Dữ liệu train | F1 | Accuracy |
|---|---:|---:|
| Bước 2: 22.361 mẫu | 0.7156 | 0.8760 |
| Bước 3: 44.722 mẫu | **0.7248** | **0.8800** |

Thêm batch 2 làm F1 tăng 0.0092 và accuracy tăng 0.0040. Hai batch cùng nguồn và phân phối
tương tự nên mức tăng nhỏ là hợp lý; thêm dữ liệu không bảo đảm metric luôn tăng.

## 3. Bonus

| Bonus | Kết quả |
|---|---|
| Remote MLflow | CI nhận URI/username/token qua GitHub Secrets và ghi run lên DagsHub |
| Decision threshold | Quét 0.10–0.90; trên dữ liệu Bước 3, F1 tăng **0.7248 → 0.7446** tại threshold **0.45** |
| Precision/Recall | Tự sinh `detail.txt`; lớp cao đạt precision 0.8037, recall 0.6935 |
| Chống regression | Candidate 0.7345 bị chặn khi production đạt 0.7446; production không bị ghi đè |
| Data drift | Tỷ lệ dương 0.2478, lệch 0.0002 so với 0.248; cảnh báo nếu lệch trên 0.05 |

Với giả định API phục vụ chiến dịch ưu đãi cao cấp, false positive làm lãng phí chi phí tiếp
cận nên precision thấp tốn kém hơn; recall vẫn cần theo dõi để không bỏ sót quá nhiều khách
hàng phù hợp. `detail.txt` báo cáo cả hai thay vì tối ưu một chỉ số đơn lẻ.
