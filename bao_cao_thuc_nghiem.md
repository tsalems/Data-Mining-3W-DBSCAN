# THIẾT KẾ THỰC NGHIỆM VÀ KẾT QUẢ ĐÁNH GIÁ

---

## 1. Bộ dữ liệu thực nghiệm

Thực nghiệm được tiến hành trên **8 bộ dữ liệu** gồm cả dữ liệu tổng hợp (synthetic) và dữ liệu thực tế (real-world), đại diện cho nhiều dạng phân bố khác nhau về hình dạng cụm, mật độ và số chiều. Các bộ dữ liệu được lấy từ kho UCI Machine Learning Repository và các bài báo gốc liên quan.

**Bảng 1. Thông tin các bộ dữ liệu thực nghiệm**

| Tên dataset | Số điểm dữ liệu | Số đặc trưng | Số cụm thực | Loại |
|---|---|---|---|---|
| 4C | 1.250 | 2 | 4 | Tổng hợp 2D |
| Pathbased | 300 | 2 | 3 | Tổng hợp 2D |
| Aggregation | 788 | 2 | 7 | Tổng hợp 2D |
| Compound | 399 | 2 | 6 | Tổng hợp 2D |
| Flame | 240 | 2 | 2 | Tổng hợp 2D |
| IRIS | 150 | 4 | 3 | Thực tế |
| Glass | 214 | 9 | 6 | Thực tế |
| Seeds | 210 | 7 | 3 | Thực tế |

**Mô tả đặc điểm:**

- **4C**: Dữ liệu 2 chiều gồm 4 cụm hình dạng tùy ý, mật độ không đồng đều. Được sử dụng trong bài báo gốc 3W-DBSCAN (Yu, 2019).
- **Pathbased**: Gồm 3 cụm với dạng đường cong, trong đó có cụm bao quanh các cụm khác — thách thức cho thuật toán dựa trên bán kính cố định.
- **Aggregation**: 7 cụm dày đặc với khoảng cách các cụm biến thiên, một số cụm nằm sát nhau.
- **Compound**: 6 cụm đa dạng hình dạng, bao gồm cụm lồng nhau và cụm có mật độ khác biệt lớn.
- **Flame**: 2 cụm hình ngọn lửa với mật độ tương đối đồng đều, có phần đuôi kéo dài. Là bài kiểm tra điển hình cho tính linh hoạt về hình dạng.
- **IRIS**: Bộ dữ liệu hoa iris kinh điển, 3 loài với 4 đặc trưng hình thái. Hai lớp có ranh giới không tuyến tính.
- **Glass**: Phân loại 6 loại thủy tinh dựa trên thành phần hóa học (9 đặc trưng). Mật độ phân bố không đồng đều giữa các lớp.
- **Seeds**: Dữ liệu hạt lúa mì, 3 giống với 7 đặc trưng hình học. Các cụm có hình dạng elipsoid, mật độ biến thiên.

Tất cả dữ liệu được chuẩn hóa về khoảng [0, 1] bằng **MinMaxScaler** trước khi đưa vào thuật toán.

---

## 2. Các độ đo đánh giá

### 2.1 Độ đo cho phân cụm cứng (Hard Clustering)

Được sử dụng để đánh giá 3W-DBSCAN, CE3-KMeans và DScale-DBSCAN. Mỗi điểm dữ liệu được gán nhãn dứt khoát vào một cụm dựa trên **Lower Bound** (chỉ tính vùng lõi POS) hoặc **Upper Bound** (tính cả vùng biên BND).

**Accuracy (ACC):** Tỉ lệ phân loại đúng sau khi khớp nhãn tối ưu bằng thuật toán Hungarian.

$$\text{ACC} = \frac{\text{Số điểm phân loại đúng}}{n}$$

**Normalized Mutual Information (NMI):** Đo mức độ tương đồng thông tin giữa nhãn dự đoán và nhãn thực, chuẩn hóa về khoảng [0, 1].

$$\text{NMI}(Y, C) = \frac{2 \cdot I(Y; C)}{H(Y) + H(C)}$$

trong đó $I(Y;C)$ là thông tin tương hỗ, $H(\cdot)$ là entropy.

**F1-score (Macro):** Trung bình F1 trên tất cả các cụm, đánh giá cân bằng giữa Precision và Recall kể cả khi số lượng điểm giữa các cụm mất cân đối.

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 2.2 Độ đo cho phân cụm mềm (Soft Clustering)

Được sử dụng để đánh giá chất lượng phân vùng 3 chiều (POS / BND / NEG) của 3W-DBSCAN và LE3W-DBSCAN.

Gọi $POS_k$, $BND_k$ lần lượt là tập vùng tích cực và vùng biên của cụm $k$, $c$ là số cụm, $n$ là tổng số điểm.

**Gamma (γ) — Tỉ lệ bao phủ toàn cục:**

$$\gamma = \frac{\sum_k |POS_k|}{n}$$

Phản ánh tỉ lệ điểm được phân vào vùng lõi (POS) so với toàn bộ dữ liệu. Gamma càng cao, các cụm càng gọn và ít điểm mơ hồ.

**Alpha (α) — Độ thuần khiết trung bình theo cụm:**

$$\alpha = \frac{1}{c} \sum_k \frac{|POS_k|}{|POS_k| + |BND_k|}$$

Trung bình tỉ lệ POS trên tổng điểm được phân (POS + BND) trong mỗi cụm. Alpha cao nghĩa là mỗi cụm có vùng lõi rõ ràng, vùng biên nhỏ.

**Alpha-star (α\*) — Độ thuần khiết toàn cục:**

$$\alpha^* = \frac{\sum_k |POS_k|}{\sum_k (|POS_k| + |BND_k|)}$$

Tương tự alpha nhưng tính trên toàn bộ tập thay vì trung bình từng cụm, đặc biệt hữu ích khi các cụm có kích thước chênh lệch lớn.

Ba chỉ số soft metrics đều nằm trong khoảng [0, 1], **giá trị càng cao thể hiện phân vùng 3 chiều càng chặt chẽ và có ý nghĩa hơn**.

---

## 3. Kết quả thực nghiệm và đánh giá

### 3.1 Thực nghiệm 3W-DBSCAN: So sánh với CE3-KMeans và DScale-DBSCAN

Thực nghiệm so sánh ba thuật toán phân cụm 3 chiều:
- **CE3-KMeans**: Baseline dựa trên K-Means mở rộng 3 chiều (CE3).
- **3W-DBSCAN** (Yu, 2019): Thuật toán DBSCAN kết hợp lý thuyết tập thô 3 chiều.
- **DScale-DBSCAN**: Biến thể cải tiến của 3W-DBSCAN sử dụng bán kính đa tỉ lệ.

Tham số tối ưu cho từng dataset được xác định thông qua tìm kiếm lưới (grid search) theo NMI trên tập upper bound.

**Bảng 2. Kết quả so sánh hiệu năng phân cụm trên 8 bộ dữ liệu**

| Dataset | Độ đo | CE3-KMeans (LB) | 3W-DBSCAN (LB) | CE3-KMeans (UB) | 3W-DBSCAN (UB) | DScale-DBSCAN (UB) |
|---|---|---|---|---|---|---|
| **4C** | ACC | 0.5696 | 0.8216 | 0.5960 | **0.9040** | 0.8808 |
| | NMI | 0.4233 | 0.7241 | 0.4350 | **0.7982** | 0.7915 |
| | F1 | 0.5270 | 0.6630 | 0.6464 | **0.7114** | 0.6972 |
| **IRIS** | ACC | 0.9667 | 0.7400 | **0.9733** | 0.9667 | 0.8600 |
| | NMI | **0.9178** | 0.6901 | 0.9144 | 0.8851 | 0.7921 |
| | F1 | 0.7346 | 0.6330 | **0.9733** | 0.9666 | 0.6923 |
| **Glass** | ACC | 0.3972 | 0.3598 | 0.4486 | **0.4579** | 0.3972 |
| | NMI | 0.3198 | 0.3465 | 0.3003 | 0.3162 | **0.3881** |
| | F1 | 0.3107 | 0.2216 | **0.3670** | 0.3634 | 0.2741 |
| **Seeds** | ACC | 0.8429 | 0.0143 | **0.8905** | 0.6571 | 0.0381 |
| | NMI | 0.6595 | 0.0267 | **0.6724** | 0.5193 | 0.0656 |
| | F1 | 0.6662 | 0.0209 | **0.8920** | 0.5339 | 0.0541 |
| **Pathbased** | ACC | 0.7133 | 0.8700 | 0.7200 | **0.9700** | 0.9600 |
| | NMI | 0.5113 | 0.7934 | 0.5006 | 0.8756 | **0.9019** |
| | F1 | 0.5155 | 0.6954 | 0.6777 | **0.9700** | 0.7331 |
| **Aggregation** | ACC | 0.8185 | 0.7424 | 0.8287 | 0.8680 | **0.8503** |
| | NMI | 0.7999 | 0.8048 | 0.8200 | **0.9278** | 0.9091 |
| | F1 | 0.7370 | 0.6819 | 0.8115 | **0.8122** | 0.7074 |
| **Compound** | ACC | 0.4511 | 0.7970 | 0.5363 | **0.8897** | 0.8722 |
| | NMI | 0.6011 | 0.8616 | 0.6781 | 0.8607 | **0.9209** |
| | F1 | 0.3507 | 0.6632 | 0.4544 | 0.7243 | **0.7394** |
| **Flame** | ACC | 0.7583 | 0.9250 | 0.7958 | 0.9667 | **0.9833** |
| | NMI | 0.3742 | 0.7947 | 0.3801 | 0.7961 | **0.9356** |
| | F1 | 0.5432 | 0.6374 | 0.7943 | **0.9644** | 0.6597 |

*LB = Lower Bound (chỉ vùng POS), UB = Upper Bound (POS + BND). Giá trị tốt nhất mỗi dòng được in đậm.*

> **[CHÈN HÌNH]** Hình 1–8: Biểu đồ trực quan 4 panels cho từng dataset (Ground Truth / DScale-DBSCAN / CE3-KMeans / 3W-DBSCAN).  
> Đường dẫn: `figures/3W/Figure_4C.png`, `figures/3W/Figure_IRIS.png`, ..., `figures/3W/Figure_Flame.png`

**Nhận xét:**

- **3W-DBSCAN** vượt trội CE3-KMeans trên hầu hết các bộ dữ liệu có hình dạng cụm phức tạp (4C, Pathbased, Compound, Flame), đặc biệt trên upper bound.
- **CE3-KMeans** có ưu thế trên dữ liệu hình cầu đồng đều (IRIS, Seeds) do giả định cụm hình cầu phù hợp.
- **DScale-DBSCAN** đạt kết quả tốt nhất về NMI trên Pathbased, Compound và Flame nhờ cơ chế bán kính đa tỉ lệ xử lý tốt dữ liệu có mật độ biến thiên.
- **Seeds** là trường hợp đặc biệt: cả 3W-DBSCAN lẫn DScale-DBSCAN thất bại hoàn toàn (ACC < 0.04), cho thấy eps toàn cục không phù hợp với phân bố mật độ không đồng đều của dataset này.

**Phân tích độ nhạy tham số Eta:**

> **[CHÈN HÌNH]** Hình 9: Biểu đồ F1 theo eta trên tất cả các bộ dữ liệu (Figure 10 trong code).  
> Đường dẫn: `figures/3W/Figure_10_F1_vs_eta.png`

Kết quả cho thấy eta = 0.20 là lựa chọn ổn định cho đa số dataset. Các bộ dữ liệu có đặc trưng khác nhau (IRIS, Glass, Seeds) hưởng lợi từ các giá trị eta trong khoảng [0.15, 0.30].

---

### 3.2 Thực nghiệm LE3W-DBSCAN: So sánh bán kính cục bộ và soft metrics

LE3W-DBSCAN cải tiến 3W-DBSCAN bằng cách thay thế bán kính toàn cục (eps) bằng **bán kính cục bộ** tự động học được từ mật độ điểm hạt nhân của từng cụm.

#### 3.2.1 Bán kính cục bộ học được

**Bảng 3. Bán kính cục bộ eps theo từng cụm (LE3W-DBSCAN)**

| Dataset | Số cụm | eps_C1 | eps_C2 | eps_C3 | eps_C4 | eps_C5 | eps_C6 | eps_C7 |
|---|---|---|---|---|---|---|---|---|
| Aggregation | 7 | 0.0490 | 0.0546 | 0.0567 | 0.0634 | 0.0702 | 0.0673 | 0.0629 |
| Compound | 6 | 0.0651 | 0.0409 | 0.0719 | 0.0478 | 0.0706 | 0.1797 | — |
| Pathbased | 3 | 0.0529 | 0.0685 | 0.1433 | — | — | — | — |
| Flame | 2 | 0.1150 | 0.1238 | — | — | — | — | — |
| IRIS | 3 | 0.1759 | 0.2058 | 0.2318 | — | — | — | — |
| Seeds | 3 | 0.2589 | 0.2774 | 0.3553 | — | — | — | — |
| 4C | 4 | 0.0370 | 0.0359 | 0.0341 | 0.1225 | — | — | — |
| Glass | 5 | 0.0595 | 0.0694 | 0.1202 | 0.2268 | 0.3235 | — | — |

**Nhận xét:** Bán kính cục bộ phản ánh rõ sự chênh lệch mật độ giữa các cụm. Ví dụ trên **Glass**, eps dao động từ 0.0595 đến 0.3235 (gấp hơn 5 lần) — điều mà một eps toàn cục không thể biểu diễn. Trên **Compound**, cụm 6 có eps = 0.1797 gấp ~4 lần cụm 2 (0.0409), tương ứng cụm thưa so với cụm dày đặc.

#### 3.2.2 So sánh soft metrics

**Bảng 4. Kết quả soft metrics: 3W-DBSCAN vs LE3W-DBSCAN**

| Dataset | α (3W) | α (LE3W) | γ (3W) | γ (LE3W) | α\* (3W) | α\* (LE3W) |
|---|---|---|---|---|---|---|
| Aggregation | 0.892 | **0.975** | 0.869 | **0.991** | 0.861 | **0.982** |
| Compound | 0.606 | **0.943** | 0.802 | **0.972** | 0.775 | **0.960** |
| Pathbased | 0.830 | **0.976** | 0.870 | **0.980** | 0.834 | **0.977** |
| **Flame** | **0.879** | 0.806 | **0.925** | 0.842 | **0.892** | 0.815 |
| IRIS | 0.669 | **0.888** | 0.740 | **0.893** | 0.661 | **0.887** |
| Seeds | 0.016 | **0.593** | 0.014 | **0.743** | 0.014 | **0.653** |
| 4C | 0.535 | **0.874** | 0.858 | **0.903** | 0.824 | **0.883** |
| Glass | 0.287 | **0.834** | 0.706 | **0.827** | 0.493 | **0.819** |

*Giá trị tốt hơn được in đậm.*

> **[CHÈN HÌNH]** Hình 10–17: Biểu đồ so sánh trực quan 3W-DBSCAN vs LE3W-DBSCAN cho từng dataset (tương đương Figure 8–11 trong bài báo LE3W-DBSCAN).  
> Đường dẫn: `figures/LE3W/Figure_Aggregation.png`, ..., `figures/LE3W/Figure_Glass.png`

**Nhận xét:**

- **LE3W-DBSCAN vượt trội 3W-DBSCAN trên 7/8 bộ dữ liệu** theo cả 3 chỉ số soft metrics.
- **Seeds** là cải thiện ấn tượng nhất: alpha tăng từ 0.016 lên 0.593 (gấp ~37 lần). 3W-DBSCAN thất bại hoàn toàn do eps toàn cục không phù hợp với mật độ ellipsoid biến thiên của Seeds.
- **Glass** cũng cải thiện đáng kể: alpha từ 0.287 lên 0.834 (+0.547), tương ứng bán kính cục bộ xử lý tốt 5 cụm có mật độ trải từ rất dày đến rất thưa.
- **Flame** là ngoại lệ duy nhất: 3W-DBSCAN giữ ưu thế (alpha 0.879 > 0.806). Phân tích chi tiết ở mục 4.

---

## 4. So sánh tổng hợp

### 4.1 Đánh giá toàn diện theo dataset

| Dataset | Đặc điểm | Thuật toán tốt nhất | Lý do |
|---|---|---|---|
| 4C | Hình dạng tùy ý, mật độ không đồng đều | LE3W-DBSCAN | Bán kính cục bộ thích nghi tốt |
| IRIS | Hình cầu, mật độ gần đồng đều | CE3-KMeans | Giả định hình cầu phù hợp |
| Glass | Nhiều lớp, mật độ rất khác nhau | LE3W-DBSCAN | Bán kính cục bộ xử lý dải mật độ rộng |
| Seeds | Hình ellipsoid, mật độ biến thiên | LE3W-DBSCAN | Eps toàn cục của 3W-DBSCAN thất bại hoàn toàn |
| Pathbased | Hình dạng đường cong | LE3W-DBSCAN / DScale-DBSCAN | Không gian cục bộ phù hợp cấu trúc đường cong |
| Aggregation | Nhiều cụm, khoảng cách biến thiên | LE3W-DBSCAN | Bán kính riêng cho từng cụm |
| Compound | Cụm lồng nhau, mật độ khác biệt | LE3W-DBSCAN / DScale-DBSCAN | Cả hai xử lý tốt sự không đồng đều |
| Flame | Đồng đều, có đuôi kéo dài | 3W-DBSCAN | Eps toàn cục đủ tốt; LE3W bị BND inflation |

### 4.2 Điểm mạnh và hạn chế

**LE3W-DBSCAN — Điểm mạnh:**
- Tự động học bán kính phù hợp cho từng cụm, không cần điều chỉnh eps thủ công.
- Hiệu quả cao trên dữ liệu có **mật độ không đồng đều** giữa các cụm (Seeds, Glass, Compound).
- Soft metrics vượt trội nhờ BND phản ánh đúng bản chất mơ hồ cục bộ của từng cụm.

**LE3W-DBSCAN — Hạn chế:**
- Với dữ liệu có mật độ đồng đều (Flame), bán kính cục bộ không mang lại lợi thế so với eps toàn cục.
- Cơ chế gán toàn bộ điểm nhiễu vào BND của cụm gần nhất làm phình vùng biên khi dữ liệu có nhiều noise tự nhiên.
- Độ phức tạp tính toán cao hơn do phải tính ma trận khoảng cách đầy đủ.

**3W-DBSCAN — Phù hợp khi:**
- Các cụm có mật độ tương đương nhau.
- Dữ liệu có nhiều điểm noise cần giữ nguyên (không muốn gán vào BND).
- Cần kiểm soát tường minh vùng biên qua tham số eta.

### 4.3 Kết luận

Kết quả thực nghiệm trên 8 bộ dữ liệu xác nhận rằng:

1. **3W-DBSCAN** cải thiện đáng kể so với CE3-KMeans trên các bộ dữ liệu hình dạng tùy ý, đặc biệt ở upper bound — chứng minh lý thuyết tập thô 3 chiều phù hợp với phân cụm mật độ.

2. **LE3W-DBSCAN** tiếp tục cải thiện 3W-DBSCAN trên **7/8 dataset** theo soft metrics, đặc biệt ấn tượng trên Seeds (+37 lần alpha) và Glass (+0.547 alpha). Bán kính cục bộ là cải tiến cốt lõi giải quyết hạn chế lớn nhất của 3W-DBSCAN là phụ thuộc eps toàn cục.

3. Trường hợp ngoại lệ **Flame** (3W-DBSCAN tốt hơn) có giải thích lý thuyết rõ ràng: khi mật độ đồng đều, bán kính cục bộ không tạo thêm giá trị và cơ chế noise→BND của LE3W gây BND inflation. Đây là giới hạn có căn cứ, không phải lỗi thuật toán.

---

## 5. Kết luận

Báo cáo này trình bày và đánh giá hai thuật toán phân cụm dựa trên lý thuyết quyết định ba chiều: **3W-DBSCAN** (Yu et al., 2019) và **LE3W-DBSCAN** (Shen et al., 2023).

3W-DBSCAN mở rộng DBSCAN bằng cách biểu diễn mỗi cụm thông qua một cặp tập lồng nhau (lower bound – upper bound), phân loại điểm dữ liệu thành ba vùng: thuộc chắc chắn (POS), mơ hồ (BND) và không thuộc (NEG). Cách biểu diễn này phù hợp hơn với bản chất nhận thức của con người và xử lý tốt dữ liệu có ranh giới cụm không rõ ràng. Thực nghiệm trên 8 bộ dữ liệu cho thấy 3W-DBSCAN vượt trội CE3-KMeans và DScale-DBSCAN trên phần lớn dataset có hình dạng tùy ý.

LE3W-DBSCAN khắc phục hạn chế cốt lõi của 3W-DBSCAN — sự phụ thuộc vào eps toàn cục — bằng cách học bán kính cục bộ riêng cho từng cụm dựa trên khoảng cách k-láng giềng. Kết quả thực nghiệm xác nhận LE3W-DBSCAN cải thiện đáng kể chất lượng phân vùng ba chiều trên 7/8 bộ dữ liệu, với mức tăng đặc biệt rõ rệt trên các dataset có mật độ phân bố không đồng đều như Seeds và Glass. Trường hợp ngoại lệ duy nhất là Flame — dataset có mật độ đồng đều — trong đó eps toàn cục đã đủ hiệu quả và cơ chế gán noise của LE3W gây phình vùng biên không cần thiết.

Hướng phát triển tiếp theo có thể tập trung vào: (1) tự động hóa việc chọn tham số $k$ và MinPts cho LE3W-DBSCAN; (2) giảm độ phức tạp tính toán $O(n^2)$ của ma trận khoảng cách đầy đủ bằng cấu trúc chỉ mục không gian; (3) mở rộng khung ba chiều cho dữ liệu luồng (streaming) và dữ liệu chiều cao.

---

## Tài liệu tham khảo

[1] H. Yu, L. Chen, J. Yao, X. Wang, "A three-way clustering method based on an improved DBSCAN algorithm," *Physica A: Statistical Mechanics and its Applications*, vol. 535, p. 122289, 2019. https://doi.org/10.1016/j.physa.2019.122289

[2] Q. Shen, Q. Zhang, M. Gao, Y. Dai, "Three-way DBSCAN Algorithm Based on Local Eps," *Computer Science*, vol. 50, no. 6, pp. 100–108, 2023. https://doi.org/10.11896/jsjkx.220800074

[3] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise," in *Proc. 2nd Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 1996, pp. 226–231.

[4] Y. Yao, "Three-way decision and granular computing," *International Journal of Approximate Reasoning*, vol. 103, pp. 107–123, 2018.

[5] P. Wang, Y. Yao, "Ce3: A three-way clustering method based on mathematical morphology," *Knowledge-Based Systems*, vol. 155, pp. 54–65, 2018.

[6] Y. Zhu, K. M. Ting, M. Angelova, "A distance scaling method to improve density-based clustering," in *Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)*, Springer, 2018, pp. 389–400.

[7] Y. Zhu, K. M. Ting, M. J. Carman, "Density-ratio based clustering for discovering clusters with varying densities," *Pattern Recognition*, vol. 60, pp. 983–997, 2016.

[8] A. Lancichinetti, S. Fortunato, J. Kertész, "Detecting the overlapping and hierarchical community structure in complex networks," *New Journal of Physics*, vol. 11, no. 3, p. 033015, 2009.

[9] D. Dua, C. Graff, "UCI Machine Learning Repository," University of California, Irvine, School of Information and Computer Sciences, 2019. [Online]. Available: http://archive.ics.uci.edu/ml
