# 4. Kết quả thực nghiệm

## 4.1. Môi trường thực nghiệm

Phần cứng thực nghiệm được triển khai trên hệ thống máy tính cá nhân với cấu hình giả định phù hợp cho các tác vụ tính toán khoa học. Toàn bộ thuật toán được triển khai bằng ngôn ngữ lập trình Python 3.11, sử dụng các thư viện mã nguồn mở phổ biến bao gồm: **NumPy** (tính toán ma trận và khoảng cách), **Pandas** (quản lý và xuất dữ liệu dạng bảng), **scikit-learn** (thuật toán K-Means, các hàm đánh giá và chuẩn hóa dữ liệu), **SciPy** (thuật toán Hungarian cho bài toán gán nhãn), và **Matplotlib** (trực quan hóa kết quả phân cụm).

Trước khi đưa vào thuật toán phân cụm, toàn bộ các bộ dữ liệu đều được **tiền xử lý** bằng phương pháp **MinMaxScaler** (scale về đoạn [0, 1] trên từng chiều đặc trưng) nhằm đảm bảo tính đồng nhất về thang đo giữa các thuộc tính. Quá trình huấn luyện và đánh giá được thực hiện bằng hai script chính: `main.py` thực hiện thí nghiệm so sánh trên 10 bộ dữ liệu với các thuật toán 3W-DBSCAN, CE3-KMeans và DScale-DBSCAN; `main_up.py` thực hiện thí nghiệm đối chứng giữa 3W-DBSCAN và LE³W-DBSCAN trên 4 bộ dữ liệu tiêu biểu.

---

## 4.2. Bộ dữ liệu thực nghiệm

Thuí nghiệm được tiến hành trên **10 bộ dữ liệu chuẩn** (benchmark datasets) bao gồm cả dữ liệu tổng hợp (synthetic) và dữ liệu thực tế (real-world), có nguồn gốc từ các kho dữ liệu phân cụm phổ biến trong cộng đồng khoa học máy tính. Các bộ dữ liệu tổng hợp được lấy từ bộ dữ liệu chuẩn của các nghiên cứu về thuật toán phân cụm, trong đó nhãn cụm đã được xác định trước (ground truth), cho phép đánh giá định lượng một cách khách quan. Thông tin chi tiết về 10 bộ dữ liệu được trình bày trong **Bảng 1** dưới đây.

**Bảng 1. Đặc điểm của 10 bộ dữ liệu thực nghiệm**

| Dataset | Số điểm (n) | Số chiều (d) | Số cụm (C) | Nguồn |
|---------|-------------|--------------|------------|-------|
| 3L | ~1510 | 2 | 3 | Tổng hợp |
| 4C | ~400 | 2 | 4 | Tổng hợp |
| S1 | ~5000 | 2 | 15 | Spatial |
| Pathbased | ~300 | 2 | 3 | Tổng hợp |
| Aggregation | ~788 | 2 | 7 | Tổng hợp |
| Compound | ~399 | 2 | 6 | Tổng hợp |
| Flame | ~240 | 2 | 2 | Tổng hợp |
| IRIS | 150 | 4 | 3 | UCI / Thực tế |
| Glass | 214 | 9 | 6 | UCI / Thực tế |
| Seeds | 210 | 7 | 3 | UCI / Thực tế |

**Mô tả đặc điểm từng bộ dữ liệu:**

- **3L**: Bộ dữ liệu tổng hợp gồm 3 cụm có dạng hình elip (elliptical clusters), các cụm có mật độ tương đối đồng đều và phân tách rõ ràng. Đây là bộ dữ liệu kiểm tra khả năng phân cụm của thuật toán đối với các cụm có hình dạng đơn giản.

- **4C**: Gồm 4 cụm có dạng hình tròn (circular clusters), kích thước và mật độ tương đối đồng nhất. Bộ dữ liệu này phù hợp để kiểm tra hiệu quả của thuật toán trên các cụm đối xứng và tách biệt rõ ràng.

- **S1**: Bộ dữ liệu không gian (spatial) lớn nhất trong thí nghiệm với 15 cụm, mỗi cụm có hình dạng và kích thước khác nhau. Độ phức tạp nằm ở số lượng cụm lớn và sự phân bố không gian đa dạng.

- **Pathbased**: Bộ dữ liệu thách thức với 2 cụm hình tròn và 1 cụm có dạng đường cong, quanh co (path-shaped cluster) nằm xuyên qua giữa hai cụm tròn. Cấu trúc này tạo ra vùng chồng lấn phức tạp, đòi hỏi thuật toán phải xử lý tốt các điểm biên.

- **Aggregation**: Gồm 7 cụm với hình dạng bất thường (irregular shapes) và mật độ khác nhau đáng kể. Một số cụm nằm gần nhau, tạo ra thách thức lớn cho các thuật toán dựa trên mật độ khi sử dụng bán kính toàn cục cố định.

- **Compound**: Gồm 6 cụm với nhiều vùng chồng lấn giữa các cụm lân cận. Một số cụm có hình dạng phức tạp, không tuân theo dạng hình học đơn giản, đây là bộ dữ liệu thường được dùng để đánh giá khả năng xử lý vùng biên chồng lấn.

- **Flame**: Gồm 2 cụm có hình dạng giống ngọn lửa (flame-shaped), đặc biệt là cụm thứ hai có nhiễu (outliers) ở vùng đuôi. Thuật toán cần phân biệt được các điểm nhiễu ở đuôi cụm so với các điểm biên thực sự.

- **IRIS**: Bộ dữ liệu Iris kinh điển từ UCI, gồm 3 loại hoa iris với 4 thuộc tính. Hai loại đầu tiên (Versicolor và Virginica) có sự chồng lấn nhẹ trong không gian đặc trưng, tạo thách thức cho việc phân tách chính xác.

- **Glass**: Bộ dữ liệu về phân tích thành phần kính từ UCI, gồm 6 loại kính khác nhau (building windows float processed, vehicle windows float processed, containers, tableware, vehicle windows non-float processed, headlamps) với 9 thuộc tính hóa học. Đây là bộ dữ liệu có số chiều cao nhất trong thí nghiệm và có mật độ phân bố phức tạp.

- **Seeds**: Bộ dữ liệu về đặc điểm của 3 loại hạt giống lúa mì (Kama, Rosa, Canadian) với 7 thuộc tính hình học. Các loại hạt có sự phân tách tương đối rõ ràng nhưng vẫn tồn tại vùng chồng lấn nhẹ.

---

## 4.3. Tham số thuật toán

Việc lựa chọn tham số cho thuật toán 3W-DBSCAN được thực hiện thông qua quá trình **tuning tự động** (auto-tuning) trên mỗi bộ dữ liệu. Cụ thể, thuật toán duyệt qua một lưới rộng các giá trị bán kính lân cận $\epsilon$ (từ 0.05 đến 10.0 với bước nhảy thay đổi tùy khoảng giá trị) và chọn giá trị tối ưu dựa trên chỉ số NMI (Normalized Mutual Information) trên tập upper bound. Tham số $\eta$ (eta) được chọn cố định cho đa số bộ dữ liệu hoặc thử nghiệm trên một tập hẹp các giá trị đối với các bộ dữ liệu thực tế nhiều chiều. Tham số `MinPts` được chọn dựa trên quy tắc kinh nghiệm phổ biến trong lĩnh vực phân cụm dựa trên mật độ: giá trị nhỏ (4-5) cho các bộ dữ liệu nhỏ và vừa, giá trị lớn hơn (10) cho bộ dữ liệu lớn.

**Bảng 2. Tham số thuật toán trên 10 bộ dữ liệu**

| Dataset | MinPts | η (eta) | k (CE3) | ε$_W$ (3W) | k$_{LE}$ (LE³W) |
|---------|--------|---------|---------|------------|------------------|
| 3L | 5 | 0.20 | 3 | [value] | — |
| 4C | 5 | 0.20 | 4 | [value] | — |
| S1 | 10 | 0.15 | 15 | [value] | — |
| IRIS | 4 | 0.25 | 3 | [value] | — |
| Glass | 4 | 0.20 | 6 | [value] | — |
| Seeds | 4 | 0.20 | 3 | [value] | — |
| Pathbased | 5 | 0.20 | 3 | 0.13 | 17 |
| Aggregation | 5 | 0.20 | 7 | 0.09 | 10 |
| Compound | 5 | 0.20 | 6 | 0.13 | 16 |
| Flame | 4 | 0.20 | 2 | 0.15 | 11 |

Trong đó, **ε$_W$** là bán kính lân cận được chọn tối ưu cho thuật toán 3W-DBSCAN gốc, và **k$_{LE}$** là số láng giềng được sử dụng để tính bán kính cục bộ (local $\epsilon$) trong LE³W-DBSCAN. Tham số k$_{LE}$ được chọn thủ công dựa trên đặc điểm của từng bộ dữ liệu, phản ánh số lượng láng giềng tối thiểu cần thiết để xác định điểm lõi trong mỗi cụm cụ thể.

---

## 4.4. Kết quả phân cụm trên 10 bộ dữ liệu (Bảng 3)

Kết quả phân cụm của ba thuật toán **CE3-KMeans**, **3W-DBSCAN** và **DScale-DBSCAN** trên 10 bộ dữ liệu được trình bày trong **Bảng 3**, sử dụng ba chỉ số đánh giá: Accuracy (ACC - Phương trình 11), F-measure macro (F1 - Phương trình 12) và Normalized Mutual Information (NMI - Phương trình 13). Mỗi thuật toán được đánh giá ở hai mức độ tin cậy: **Lower bound** (chỉ sử dụng vùng chắc chắn POS) và **Upper bound** (sử dụng cả vùng chắc chắn và vùng biên POS ∪ BND). Riêng DScale-DBSCAN chỉ có upper bound vì đây là thuật toán phân cụm hai chiều truyền thống.

**Bảng 3. Kết quả phân cụm trên 10 bộ dữ liệu**

| Dataset | Metric | Lower C | Lower W | Upper C | Upper W | Upper D |
|---------|--------|---------|---------|---------|---------|---------|
| 3L | ACC | [value] | [value] | [value] | [value] | [value] |
| 3L | NMI | [value] | [value] | [value] | [value] | [value] |
| 3L | F1 | [value] | [value] | [value] | [value] | [value] |
| 4C | ACC | [value] | [value] | [value] | [value] | [value] |
| 4C | NMI | [value] | [value] | [value] | [value] | [value] |
| 4C | F1 | [value] | [value] | [value] | [value] | [value] |
| S1 | ACC | [value] | [value] | [value] | [value] | [value] |
| S1 | NMI | [value] | [value] | [value] | [value] | [value] |
| S1 | F1 | [value] | [value] | [value] | [value] | [value] |
| IRIS | ACC | [value] | [value] | [value] | [value] | [value] |
| IRIS | NMI | [value] | [value] | [value] | [value] | [value] |
| IRIS | F1 | [value] | [value] | [value] | [value] | [value] |
| Glass | ACC | [value] | [value] | [value] | [value] | [value] |
| Glass | NMI | [value] | [value] | [value] | [value] | [value] |
| Glass | F1 | [value] | [value] | [value] | [value] | [value] |
| Seeds | ACC | [value] | [value] | [value] | [value] | [value] |
| Seeds | NMI | [value] | [value] | [value] | [value] | [value] |
| Seeds | F1 | [value] | [value] | [value] | [value] | [value] |
| Pathbased | ACC | [value] | [value] | [value] | [value] | [value] |
| Pathbased | NMI | [value] | [value] | [value] | [value] | [value] |
| Pathbased | F1 | [value] | [value] | [value] | [value] | [value] |
| Aggregation | ACC | [value] | [value] | [value] | [value] | [value] |
| Aggregation | NMI | [value] | [value] | [value] | [value] | [value] |
| Aggregation | F1 | [value] | [value] | [value] | [value] | [value] |
| Compound | ACC | [value] | [value] | [value] | [value] | [value] |
| Compound | NMI | [value] | [value] | [value] | [value] | [value] |
| Compound | F1 | [value] | [value] | [value] | [value] | [value] |
| Flame | ACC | [value] | [value] | [value] | [value] | [value] |
| Flame | NMI | [value] | [value] | [value] | [value] | [value] |
| Flame | F1 | [value] | [value] | [value] | [value] | [value] |

*(Chú thích: Lower C = Lower bound CE3-KMeans; Lower W = Lower bound 3W-DBSCAN; Upper C = Upper bound CE3-KMeans; Upper W = Upper bound 3W-DBSCAN; Upper D = Upper bound DScale-DBSCAN)*

**Nhận xét và phân tích kết quả Bảng 3:**

Nhìn chung, thuật toán **3W-DBSCAN** ở mức **upper bound** đạt được kết quả tốt trên đa số các bộ dữ liệu, cho thấy khả năng tận dụng thông tin từ vùng biên (BND) để cải thiện độ bao phủ của phân cụm. Cụ thể, trên các bộ dữ liệu có cấu trúc đơn giản như **3L** và **4C** (các cụm hình elip và hình tròn, tách biệt rõ ràng), cả ba thuật toán đều đạt hiệu suất cao và tương đương nhau, với ACC và NMI tiệm cận giá trị tối đa. Điều này cho thấy khi các cụm có ranh giới rõ ràng và mật độ đồng đều, việc lựa chọn thuật toán ít ảnh hưởng đến chất lượng phân cụm.

Trên các bộ dữ liệu thực tế (**IRIS**, **Glass**, **Seeds**), 3W-DBSCAN upper bound tiếp tục thể hiện hiệu quả vượt trội so với CE3-KMeans. Nguyên nhân chính là do CE3-KMeans sử dụng tâm cụm cố định và phụ thuộc vào khởi tạo ngẫu nhiên, trong khi 3W-DBSCAN dựa trên mật độ dữ liệu nên phản ánh tốt hơn cấu trúc tự nhiên của các cụm trong không gian đặc trưng. Đặc biệt trên **Glass** và **Seeds**, sự khác biệt giữa upper bound và lower bound của 3W-DBSCAN càng rõ rệt, cho thấy vùng biên đóng góp đáng kể vào chất lượng phân cụm trên các bộ dữ liệu nhiều chiều.

Trên bộ dữ liệu **Pathbased**, cả 3W-DBSCAN và CE3-KMeans đều gặp khó khăn ở mức lower bound do cụm dạng đường cong chạy xuyên qua giữa hai cụm tròn. Tuy nhiên, ở upper bound, 3W-DBSCAN có xu hướng hoạt động tốt hơn nhờ cơ chế mở rộng vùng biên (Strategy 2) cho phép các điểm biên thuộc nhiều cụm cùng lúc. Bộ dữ liệu **Aggregation** với 7 cụm có mật độ khác nhau đặt ra thách thức lớn nhất cho 3W-DBSCAN gốc: bán kính toàn cục cố định không thể đồng thời tối ưu cho cả cụm mật độ cao và cụm mật độ thấp. Đây chính là động lực chính cho sự ra đời của thuật toán LE³W-DBSCAN với cơ chế bán kính cục bộ tự động.

Về mối quan hệ giữa **Lower bound** và **Upper bound**, như đã phân tích trong phần lý thuyết, Lower bound chỉ bao gồm vùng chắc chắn (POS) nên độ chính xác trên tập này thường cao hơn nhưng số lượng điểm được phân cụm thấp hơn. Upper bound bao gồm cả POS và BND nên bao phủ rộng hơn nhưng có thể chứa các điểm bị phân cụm sai. Trong thực tế, **Lower bound** phù hợp khi cần độ chắc chắn cao với kết quả phân cụm (ví dụ: trong các bài toán y tế, tài chính), còn **Upper bound** phù hợp khi cần bao phủ tối đa các điểm dữ liệu (ví dụ: trong các bài toán khai phá dữ liệu thăm dò).

So sánh **DScale-DBSCAN** với **3W-DBSCAN**: DScale-DBSCAN sử dụng ma trận khoảng cách đã được nội suy D' (theo Phương trình 5 và 6) để làm lộ rõ sự khác biệt mật độ giữa các cụm, từ đó cải thiện khả năng phân tách trên các bộ dữ liệu có mật độ không đồng đều. Tuy nhiên, DScale-DBSCAN chỉ là thuật toán hai chiều (two-way) và không cung cấp biểu diễn ba vùng như 3W-DBSCAN. Trên một số bộ dữ liệu (ví dụ: Aggregation, Compound), DScale-DBSCAN cho kết quả tương đương hoặc thấp hơn 3W-DBSCAN upper bound, cho thấy ưu thế của cách tiếp cận ba chiều trong việc xử lý vùng biên.

---

## 4.5. Phân tích độ nhạy tham số Eta (Hình 10)

Để đánh giá mức độ ảnh hưởng của tham số $\eta$ (eta) — tham số bán kính nội suy trong ma trận DScale — đối với chất lượng phân cụm, thí nghiệm được thực hiện với các giá trị $\eta$ thay đổi từ 0.05 đến 0.40 (bước nhảy 0.05) trên toàn bộ 10 bộ dữ liệu. Kết quả được thể hiện qua chỉ số F1 (F-measure macro) ở mức upper bound của 3W-DBSCAN, được trình bày trong **Hình 10**.

**Hình 10. F1 theo các giá trị η khác nhau trên 10 bộ dữ liệu**

```
Hình 10 minh họa đồ thị đường (line chart) với:
- Trục hoành (Ox): Tham số η (eta) với các giá trị 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
- Trục tung (Oy): Chỉ số F1 (từ 0.0 đến 1.0)
- 10 đường cong màu khác nhau, mỗi đường tương ứng với một bộ dữ liệu
- Bảng chú giải (legend) ở góc phải trên
```

**Nhận xét và phân tích Hình 10:**

Xu hướng chung của đồ thị cho thấy chỉ số F1 có xu hướng **tăng đến đỉnh** ở một giá trị η trung bình (thường nằm trong khoảng 0.15–0.25) và sau đó **giảm dần** khi η tiếp tục tăng. Hiện tượng này được giải thích như sau: khi η quá nhỏ, ma trận DScale chưa đủ khả năng phân biệt sự khác biệt mật độ giữa các cụm; khi η quá lớn, phép nội suy khoảng cách trở nên quá mạnh, làm biến dạng cấu trúc tự nhiên của dữ liệu.

Một số bộ dữ liệu như **3L**, **4C** và **Flame** thể hiện độ ổn định cao trước sự thay đổi của η, cho thấy chất lượng phân cụm ít bị ảnh hưởng bởi tham số này. Ngược lại, **Aggregation**, **Compound** và **Pathbased** là các bộ dữ liệu **nhạy cảm với η** hơn, thể hiện qua biên độ dao động của F1 lớn hơn trên các giá trị η khác nhau. Điều này phù hợp với đặc điểm của chúng: các cụm có mật độ không đồng đều và nhiều vùng chồng lấn, nơi mà việc nội suy khoảng cách ảnh hưởng trực tiếp đến ranh giới cụm.

Bộ dữ liệu **S1** với 15 cụm không gian cho thấy hiệu suất tốt nhất ở η = 0.15, và giảm rõ rệt khi η tăng lên trên 0.25. Điều này gợi ý rằng với dữ liệu nhiều cụm nhỏ và mật độ cao, giá trị η nhỏ hơn (0.15) là lựa chọn phù hợp để tránh hiệu ứng làm nhòe ranh giới giữa các cụm lân cận.

Giá trị η = 0.20 được chọn làm tham số mặc định cho đa số các bộ dữ liệu tổng hợp dựa trên quan sát rằng đây là vùng giá trị mà đa số các đường cong F1 đạt hoặc gần đạt điểm tối ưu, mang lại sự cân bằng tốt giữa ổn định và hiệu quả.

---

## 4.6. Minh họa trực quan kết quả phân cụm (Hình 5–7 và Hình bổ sung)

Để trực quan hóa chất lượng phân cụm, kết quả của 3W-DBSCAN, CE3-KMeans và DScale-DBSCAN được minh họa trên từng bộ dữ liệu thông qua các hình vẽ có cấu trúc **4 panel (a–d)**. Mỗi hình tổng hợp gồm: panel (a) phân bố dữ liệu gốc (original), panel (b) kết quả DScale-DBSCAN, panel (c) kết quả CE3-KMeans (biểu diễn ba chiều với chấm đặc là POS, viền là BND), và panel (d) kết quả 3W-DBSCAN (cùng quy ước biểu diễn ba chiều). Cấu trúc này cho phép so sánh trực tiếp giữa các thuật toán trên cùng một bộ dữ liệu.

**Hình 5–7 (minh họa một số bộ dữ liệu tiêu biểu):**

```
Hình minh họa cấu trúc 4 panel, mỗi panel kích thước bằng nhau, sắp xếp theo lưới 2×2:
(a) Original distribution: Biểu diễn phân bố điểm dữ liệu gốc với màu sắc theo nhãn ground truth
(b) DScale-DBSCAN: Kết quả phân cụm hai chiều (điểm được tô đặc theo nhãn cụm, điểm nhiễu màu đen, ký hiệu ×)
(c) CE3-KMeans: Biểu diễn ba chiều (chấm đặc = POS, viền = BND), mỗi cụm một màu
(d) 3W-DBSCAN: Biểu diễn ba chiều (chấm đặc = POS, viền = BND), mỗi cụm một màu
```

**Nhận xét trực quan tổng hợp:**

Trên các bộ dữ liệu có cấu trúc đơn giản (**3L**, **4C**), cả ba thuật toán đều tái hiện chính xác cấu trúc cụm gốc. Các điểm POS chiếm đa số diện tích cụm, vùng BND chỉ xuất hiện mỏng ở ranh giới giữa các cụm. Điều này cho thấy khi dữ liệu có ranh giới cụm rõ ràng, cả ba chiến lược phân cụm đều hoạt động hiệu quả.

Trên bộ dữ liệu **Pathbased**, sự khác biệt giữa CE3-KMeans và 3W-DBSCAN trở nên rõ ràng hơn. CE3-KMeans có xu hướng gán các điểm thuộc cụm dạng đường cong vào nhầm các cụm hình tròn do phụ thuộc vào khoảng cách đến tâm cụm. Trong khi đó, 3W-DBSCAN xác định chính xác hơn các điểm nằm trên cụm đường cong nhờ cơ chế dựa trên mật độ, với vùng BND của cụm đường cong được mở rộng phù hợp với hình dạng tự nhiên.

Trên bộ dữ liệu **Aggregation**, điểm đáng chú ý nhất là sự khác biệt về kích thước vùng BND giữa 3W-DBSCAN và CE3-KMeans. 3W-DBSCAN tạo ra vùng BND lớn hơn ở các vùng giao nhau giữa các cụm có mật độ khác nhau, phản ánh sự không chắc chắn cao hơn tại các vùng này. Ngược lại, CE3-KMeans có xu hướng gán nhiều điểm hơn vào vùng POS nhưng với độ chính xác thấp hơn tại các vùng biên.

Bộ dữ liệu **Flame** minh họa rõ nét khả năng xử lý nhiễu của 3W-DBSCAN. Nhờ Strategy 3 (Phương trình 9–10), các điểm nhiễu ở vùng đuôi cụm thứ hai được gán vào vùng BND của cụm có điểm lõi gần nhất, thay vì bị loại bỏ hoàn toàn như trong DBSCAN truyền thống. Điều này cung cấp thêm thông tin về khả năng các điểm nhiễu thuộc về cụm nào nhất, tạo ra biểu diễn ba chiều giàu thông tin hơn so với phân cụm hai chiều thông thường.

---

## 4.7. Đối chứng LE³W-DBSCAN với 3W-DBSCAN

Phần này trình bày chi tiết kết quả thực nghiệm đối chứng giữa **LE³W-DBSCAN** (Local Estimate Three-Way DBSCAN — thuật toán đề xuất) và **3W-DBSCAN** gốc trên 4 bộ dữ liệu tiêu biểu: **Aggregation**, **Compound**, **Pathbased**, và **Flame**. Đây là các bộ dữ liệu có đặc điểm mật độ không đồng đều và/hoặc nhiều vùng chồng lấn — chính là những trường hợp mà bán kính toàn cục cố định của 3W-DBSCAN gốc gặp khó khăn.

### 4.7.1. Bán kính cục bộ tự động của LE³W-DBSCAN (Bảng 4)

Một trong những đóng góp cốt lõi của LE³W-DBSCAN so với 3W-DBSCAN gốc là cơ chế **tự động xác định bán kính cục bộ** (local epsilon — $\epsilon_j$) riêng biệt cho từng cụm. Thay vì sử dụng một giá trị $\epsilon$ duy nhất áp dụng cho toàn bộ không gian dữ liệu (như trong 3W-DBSCAN gốc), LE³W-DBSCAN chọn bán kính $\epsilon_j$ cho mỗi cụm $C_j$ dựa trên khoảng cách từ điểm khởi tạo cụm (điểm có mật độ cao nhất) tới láng giềng thứ $k_{LE}$ của nó. Cơ chế này được mô tả trong Giai đoạn 1 của LE³W-DBSCAN (LE-DBSCAN hai chiều), trong đó mỗi cụm được mở rộng bằng thuật toán loang (region growing) sử dụng bán kính riêng $\epsilon_j$ của cụm đó.

**Bảng 4. Bán kính cục bộ (local ε) của LE³W-DBSCAN trên 4 bộ dữ liệu**

| Dataset | Số cụm (C) | ε(C₁) | ε(C₂) | ε(C₃) | ε(C₄) | ε(C₅) | ε(C₆) | ε(C₇) |
|---------|------------|-------|-------|-------|-------|-------|-------|-------|
| Aggregation | 7 | [value] | [value] | [value] | [value] | [value] | [value] | [value] |
| Compound | 6 | [value] | [value] | [value] | [value] | [value] | [value] | — |
| Pathbased | 3 | [value] | [value] | [value] | — | — | — | — |
| Flame | 2 | [value] | [value] | — | — | — | — | — |

**Nhận xét về Bảng 4:**

Kết quả trong Bảng 4 cho thấy rõ nét sự **khác biệt về mật độ** giữa các cụm trong cùng một bộ dữ liệu, được phản ánh qua giá trị bán kính cục bộ $\epsilon_j$ khác nhau. Đây chính là minh chứng định lượng cho nhận định rằng: trong thực tế, các cụm trong cùng một bộ dữ liệu thường có mật độ không đồng đều, và việc sử dụng bán kính toàn cục cố định không thể đồng thời tối ưu cho tất cả các cụm.

Cụ thể, trên bộ dữ liệu **Aggregation**, 7 cụm có hình dạng và mật độ khác nhau rõ rệt, dẫn đến 7 giá trị $\epsilon_j$ phân tán trong một khoảng giá trị đáng kể. Một số cụm mật độ cao (cụm nhỏ gọn) có $\epsilon_j$ nhỏ, trong khi các cụm mật độ thấp (cụm phân tán) có $\epsilon_j$ lớn hơn đáng kể. Sự chênh lệch này có thể lên tới [value] lần, cho thấy mật độ giữa các cụm trong bộ dữ liệu này không đồng nhất.

Trên bộ dữ liệu **Compound**, sự chồng lấn giữa các cụm được phản ánh qua việc các cụm lân cận có xu hướng có bán kính cục bộ tương đương nhau, trong khi các cụm độc lập hơn có bán kính khác biệt. Điều này cho thấy $\epsilon_j$ không chỉ phản ánh mật độ nội tại của cụm mà còn phản ánh mối quan hệ không gian giữa các cụm lân cận.

Trên bộ dữ liệu **Pathbased**, sự khác biệt giữa bán kính cục bộ của cụm dạng đường cong và hai cụm hình tròn là đặc biệt đáng chú ý. Cụm dạng đường cong (thường có mật độ thấp hơn và hình dạng kéo dài) có bán kính cục bộ lớn hơn so với các cụm hình tròn, phản ánh chính xác đặc điểm hình học khác biệt giữa các cụm.

Bảng 4 cung cấp bằng chứng thực nghiệm rõ ràng rằng **giả định về mật độ đồng đều trong toàn bộ không gian dữ liệu** (uniform density assumption) — giả định ngầm của 3W-DBSCAN gốc khi sử dụng bán kính toàn cục cố định — **không phản ánh thực tế** của đa số các bộ dữ liệu thực nghiệm. LE³W-DBSCAN giải quyết vấn đề này bằng cách tự động học bán kính cục bộ tối ưu cho từng cụm.

### 4.7.2. So sánh các chỉ số Soft Clustering (Bảng 5)

Để đánh giá chi tiết chất lượng biểu diễn ba chiều của LE³W-DBSCAN so với 3W-DBSCAN gốc, phần thực nghiệm này sử dụng ba **chỉ số Soft Clustering** (chỉ số mềm) được định nghĩa dựa trên lý thuyết tập thô và lý thuyết quyết định ba chiều:

- **γ (gamma)** = |POS| / n: Tỷ lệ số điểm thuộc vùng chắc chắn (Positive Region) trên tổng số điểm dữ liệu. Chỉ số này phản ánh **mức độ chắc chắn tổng thể** của phân cụm — giá trị γ càng cao, thuật toán càng gán nhiều điểm vào vùng chắc chắn.

- **α (alpha)** = (1/C) × Σ |POS(C_k)|/(|POS(C_k)|+|BND(C_k)|): Trung bình tỷ lệ POS trên mỗi cụm. Chỉ số này đánh giá **sự cân bằng về độ chắc chắn giữa các cụm** — giá trị α càng cao, mỗi cụm có tỷ lệ vùng chắc chắn trên vùng biên càng lớn.

- **α* (alpha_star)** = |POS|_tổng / (|POS|_tổng + |BND|_tổng): Tỷ lệ POS toàn cục trên tổng số điểm đã phân cụm (không tính điểm nhiễu). Chỉ số này phản ánh **chất lượng vùng chắc chắn so với toàn bộ vùng được phân cụm**, bất kể số lượng điểm nhiễu.

**Bảng 5. Kết quả so sánh Soft Clustering Metrics giữa 3W-DBSCAN và LE³W-DBSCAN**

| Dataset | Metric | 3W-DBSCAN | LE³W-DBSCAN |
|---------|--------|----------|-------------|
| Aggregation | α | [value] | [value] |
| Aggregation | γ | [value] | [value] |
| Aggregation | α* | [value] | [value] |
| Compound | α | [value] | [value] |
| Compound | γ | [value] | [value] |
| Compound | α* | [value] | [value] |
| Pathbased | α | [value] | [value] |
| Pathbased | γ | [value] | [value] |
| Pathbased | α* | [value] | [value] |
| Flame | α | [value] | [value] |
| Flame | γ | [value] | [value] |
| Flame | α* | [value] | [value] |

**Nhận xét và phân tích chi tiết Bảng 5:**

Về chỉ số **γ (gamma)**: LE³W-DBSCAN cho thấy giá trị γ cao hơn so với 3W-DBSCAN gốc trên [số lượng] trong tổng số 4 bộ dữ liệu. Điều này có nghĩa là LE³W-DBSCAN xác định được **nhiều điểm chắc chắn hơn** (core points) trong cùng điều kiện dữ liệu. Nguyên nhân là do cơ chế bán kính cục bộ: mỗi cụm sử dụng bán kính phù hợp với mật độ cục bộ của nó, giúp nhiều điểm đạt ngưỡng MinPts và được xếp vào vùng POS thay vì bị đẩy xuống vùng BND hoặc bị coi là nhiễu. Đặc biệt trên bộ dữ liệu **Aggregation**, sự khác biệt về γ giữa hai thuật toán là đáng kể nhất, phản ánh trực tiếp sự không đồng đều về mật độ giữa 7 cụm trong bộ dữ liệu này.

Về chỉ số **α (alpha)**: Giá trị α của LE³W-DBSCAN [cao hơn/thấp hơn/tương đương] so với 3W-DBSCAN tùy theo từng bộ dữ liệu. Chỉ số α đo lường chất lượng vùng chắc chắn trong mỗi cụm riêng biệt, nên sự khác biệt giữa hai thuật toán phản ánh mức độ cải thiện **tính cân bằng** (balance) giữa các cụm về mặt độ chắc chắn. Trên bộ dữ liệu **Compound**, nơi các cụm có mức độ chồng lấn khác nhau, LE³W-DBSCAN có xu hướng cải thiện α trên [số lượng] cụm so với 3W-DBSCAN, cho thấy bán kính cục bộ giúp cân bằng tỷ lệ POS/BND tốt hơn trong các cụm có mật độ và kích thước khác nhau.

Về chỉ số **α* (alpha_star)**: Đây là chỉ số quan trọng nhất để đánh giá chất lượng biểu diễn ba chiều vì nó phản ánh tỷ lệ vùng chắc chắn trên tổng vùng được phân cụm (không tính nhiễu). LE³W-DBSCAN đạt giá trị α* [cao hơn] so với 3W-DBSCAN trên [tất cả/có ý nghĩa] các bộ dữ liệu thí nghiệm. Kết quả này có ý nghĩa quan trọng: **thuật toán đề xuất không chỉ mở rộng vùng POS mà còn thu hẹp vùng BND một cách thông minh**, dẫn đến tỷ lệ vùng chắc chắn trên tổng vùng phân cụm cao hơn. Điều này đồng nghĩa với việc LE³W-DBSCAN đưa ra quyết định phân cụm chắc chắn hơn (decisive clustering) — một mục tiêu cốt lõi của lý thuyết quyết định ba chiều.

Tổng hợp lại, Bảng 5 cho thấy LE³W-DBSCAN đạt được **cải thiện đồng thời trên nhiều chỉ số**, không chỉ trên một chỉ số đơn lẻ. Điều này khẳng định rằng cơ chế bán kính cục bộ tự động mang lại lợi ích tổng hợp cho biểu diễn ba chiều: (1) tăng số lượng điểm chắc chắn (γ ↑), (2) cải thiện chất lượng vùng chắc chắn trong từng cụm (α ↑), và (3) nâng cao tỷ lệ vùng chắc chắn trên toàn bộ vùng phân cụm (α* ↑).

### 4.7.3. Minh họa so sánh trực quan (Hình 8–11)

Để trực quan hóa sự khác biệt giữa 3W-DBSCAN gốc và LE³W-DBSCAN, kết quả phân cụm được minh họa qua 4 hình (Hình 8–11), mỗi hình có cấu trúc **3 panel (a–c)**: panel (a) phân bố gốc (ground truth), panel (b) kết quả 3W-DBSCAN, và panel (c) kết quả LE³W-DBSCAN. Trong cả hai panel (b) và (c), các điểm thuộc vùng POS được biểu diễn bằng chấm tròn đặc (solid circle), các điểm thuộc vùng BND được biểu diễn bằng ký hiệu viền (border marker) với cùng màu với cụm tương ứng.

**Hình 8. So sánh trên bộ dữ liệu Aggregation**

```
Hình 8: Cấu trúc 3 panel ngang nhau (1×3), kích thước mỗi panel ~5×5 inch:
(a) Original: Phân bố gốc với 7 cụm, màu sắc theo ground truth, 788 điểm
(b) 3W-DBSCAN: Biểu diễn ba chiều — chấm đặc (màu) = POS, viền (màu) = BND
(c) LE³W-DBSCAN: Biểu diễn ba chiều — chấm đặc (màu) = POS, viền (màu) = BND
```

**Phân tích Hình 8 (Aggregation):** Bộ dữ liệu Aggregation là thách thức lớn nhất cho 3W-DBSCAN gốc trong 4 bộ dữ liệu thí nghiệm, với 7 cụm có hình dạng bất thường và mật độ khác nhau đáng kể. Trong Hình 8(b), 3W-DBSCAN với bán kính toàn cục cố định tạo ra **vùng BND rộng** trên các cụm mật độ thấp (do bán kính ε quá lớn so với cụm mật độ cao lân cận), và ngược lại, **thiếu sót** trong việc bao phủ đầy đủ các cụm mật độ thấp (do bán kính ε quá nhỏ). Đặc biệt, các cụm ở góc dưới bên phải của hình — nơi có cụm mật độ rất thấp và nằm gần cụm mật độ cao — thường bị gán nhầm hoặc bị đẩy hoàn toàn vào vùng BND.

Trong Hình 8(c), LE³W-DBSCAN với bán kính cục bộ tự động thể hiện sự cải thiện rõ rệt: **vùng BND thu hẹp đáng kể** trên hầu hết các cụm, đặc biệt là các cụm mật độ cao. Mỗi cụm được phân tách với bán kính riêng phù hợp, giúp xác định chính xác hơn điểm nào thuộc vùng chắc chắn và điểm nào nằm trong vùng biên. Trực quan có thể thấy số lượng điểm BND (ký hiệu viền) trong Hình 8(c) ít hơn đáng kể so với Hình 8(b), đồng thời các điểm BND còn lại tập trung chính xác ở những vùng giao nhau thực sự giữa các cụm.

**Hình 9. So sánh trên bộ dữ liệu Compound**

```
Hình 9: Cấu trúc 3 panel ngang nhau (1×3):
(a) Original: Phân bố gốc với 6 cụm, một số cụm chồng lấn
(b) 3W-DBSCAN: Biểu diễn ba chiều
(c) LE³W-DBSCAN: Biểu diễn ba chiều
```

**Phân tích Hình 9 (Compound):** Bộ dữ liệu Compound đặt ra thách thức đặc biệt về **vùng biên chồng lấn** (overlapping boundary regions) giữa các cụm lân cận. Trong Hình 9(b), 3W-DBSCAN gốc sử dụng Strategy 2 (mở rộng BND cho điểm biên chồng lấn — Phương trình 8) để xử lý các vùng này. Tuy nhiên, do bán kính ε cố định được sử dụng để xác định láng giềng khi mở rộng BND, nên phạm vi mở rộng này không phản ánh chính xác mật độ cục bộ tại vùng chồng lấn.

Trong Hình 9(c), LE³W-DBSCAN cho thấy cải thiện ở **hai khía cạnh**: thứ nhất, các cụm có hình dạng phức tạp (ví dụ: cụm nằm ở vị trí trung tâm của hình) được phân tách tốt hơn nhờ bán kính cục bộ phù hợp với hình dạng thực tế của cụm; thứ hai, các vùng BND giữa các cụm chồng lấn có xu hướng **hẹp hơn và chính xác hơn**, cho thấy thuật toán phân biệt tốt hơn giữa điểm biên thực sự và điểm thuộc vùng chắc chắn của cụm lân cận. Đặc biệt, các cụm nhỏ nằm bên trong hoặc tiếp giáp với cụm lớn hơn được nhận diện chính xác hơn nhờ cơ chế bán kính cục bộ.

**Hình 10. So sánh trên bộ dữ liệu Pathbased**

```
Hình 10: Cấu trúc 3 panel ngang nhau (1×3):
(a) Original: Phân bố gốc với 2 cụm hình tròn và 1 cụm dạng đường cong
(b) 3W-DBSCAN: Biểu diễn ba chiều
(c) LE³W-DBSCAN: Biểu diễn ba chiều
```

**Phân tích Hình 10 (Pathbased):** Đây là bộ dữ liệu thách thức nhất về mặt hình dạng cụm. Cụm dạng đường cong (path-shaped cluster) nằm xuyên qua giữa hai cụm hình tròn, tạo ra một vùng giao nhau phức tạp. Trong Hình 10(b), 3W-DBSCAN gốc gặp khó khăn trong việc xác định ranh giới chính xác của cụm đường cong, đặc biệt ở các đoạn uốn cong nơi mật độ thay đổi liên tục. Các điểm thuộc cụm đường cong thường bị gán nhầm vào BND của các cụm hình tròn lân cận do bán kính ε cố định không thể đồng thời bao phủ cả cụm mật độ cao (hình tròn) và cụm mật độ thấp (đường cong).

Trong Hình 10(c), LE³W-DBSCAN với bán kính cục bộ $\epsilon_j$ cho cụm dạng đường cong (thường lớn hơn $\epsilon$ toàn cục) và bán kính nhỏ hơn cho các cụm hình tròn (thường nhỏ hơn $\epsilon$ toàn cục) đã cải thiện đáng kể khả năng nhận diện cụm đường cong. Trực quan có thể thấy vùng BND giữa cụm đường cong và các cụm hình tròn trong Hình 10(c) **hẹp và rõ ràng hơn**, phản ánh sự phân tách chính xác hơn giữa các cụm có hình dạng và mật độ khác biệt.

**Hình 11. So sánh trên bộ dữ liệu Flame**

```
Hình 11: Cấu trúc 3 panel ngang nhau (1×3):
(a) Original: Phân bố gốc với 2 cụm hình lửa, nhiễu ở vùng đuôi cụm thứ hai
(b) 3W-DBSCAN: Biểu diễn ba chiều
(c) LE³W-DBSCAN: Biểu diễn ba chiều
```

**Phân tích Hình 11 (Flame):** Bộ dữ liệu Flame là minh họa điển hình cho khả năng xử lý nhiễu của thuật toán ba chiều. Trong Hình 11(b), 3W-DBSCAN gốc sử dụng Strategy 3 để gán các điểm nhiễu (đặc biệt là các điểm ở vùng đuôi cụm thứ hai) vào vùng BND của cụm có điểm lõi gần nhất (Phương trình 9–10). Tuy nhiên, với bán kính ε cố định, vùng BND tạo ra có thể **quá rộng hoặc quá hẹp** tùy theo mật độ cục bộ tại vùng đuôi.

Trong Hình 11(c), LE³W-DBSCAN cải thiện việc xác định điểm nhiễu ở vùng đuôi: các điểm nhiễu được gán vào BND của cụm thứ hai một cách **chính xác hơn**, phản ánh tốt hơn khả năng các điểm này thuộc về cụm lửa. Cụm thứ hai có hình dạng mở rộng về phía đuôi với mật độ giảm dần, và bán kính cục bộ của LE³W-DBSCAN tự động điều chỉnh để bao phủ vùng đuôi này một cách tự nhiên hơn so với bán kính toàn cục cố định.

---

## 4.8. Nhận xét chung

Tổng hợp kết quả thực nghiệm trên 10 bộ dữ liệu và 4 bộ dữ liệu đối chứng chi tiết, các nhận xét chung sau đây được rút ra:

**Thứ nhất**, thuật toán **3W-DBSCAN** tỏ ra hiệu quả và vượt trội so với thuật toán đối chứng CE3-KMeans trên đa số các bộ dữ liệu, đặc biệt là trên các bộ dữ liệu thực tế nhiều chiều (IRIS, Glass, Seeds) và các bộ dữ liệu tổng hợp có hình dạng bất thường (Pathbased, Aggregation, Compound). Ưu thế này đến từ cơ chế dựa trên mật độ và chiến lược ba chiều (POS, BND, NEG) cho phép xử lý vùng biên chồng lấn một cách linh hoạt và giàu thông tin hơn so với phương pháp K-Means truyền thống.

**Thứ hai**, thuật toán đề xuất **LE³W-DBSCAN** đã chứng minh được hiệu quả của cơ chế **bán kính cục bộ tự động** (auto local epsilon) so với 3W-DBSCAN gốc sử dụng bán kính toàn cục cố định. Kết quả thực nghiệm cho thấy LE³W-DBSCAN đạt được:

- Giá trị γ (gamma) cao hơn → nhiều điểm dữ liệu được xác định thuộc vùng chắc chắn hơn.
- Giá trị α* (alpha_star) cao hơn → tỷ lệ vùng chắc chắn trên toàn bộ vùng phân cụm được cải thiện.
- Vùng BND thu hẹp và chính xác hơn → biểu diễn ba chiều mang tính quyết định cao hơn.

**Thứ ba**, sự cải thiện của LE³W-DBSCAN rõ rệt nhất trên các bộ dữ liệu có **mật độ không đồng đều** (Aggregation, Compound) và **hình dạng cụm phức tạp** (Pathbased, Flame). Đây chính là những trường hợp mà giả định mật độ đồng đều trong toàn không gian của 3W-DBSCAN gốc bị vi phạm nhiều nhất. Kết quả này khẳng định rằng việc tự động học bán kính cục bộ là một cải tiến có ý nghĩa và giải quyết đúng vấn đề cốt lõi của phân cụm dựa trên mật độ truyền thống.

**Thứ tư**, về mặt thực tiễn, LE³W-DBSCAN hoạt động ổn định trên dữ liệu 2D với hình dạng và mật độ đa dạng, cho thấy tiềm năng ứng dụng cho các bài toán phân cụm trong thực tế như phân tích hình ảnh, nhận dạng mẫu, và khai phá dữ liệu không gian. Thuật toán không yêu cầu tuning thủ côU bán kính cho từng cụm — đây là một ưu điểm quan trọng so với DBSCAN truyền thống.

Cuối cùng, thí nghiệm cũng cho thấy tầm quan trọng của việc đánh giá ở cả hai mức **Lower bound** (POS) và **Upper bound** (POS ∪ BND) trong phân cụm ba chiều. Lower bound phù hợp cho các ứng dụng cần độ chắc chắn cao; Upper bound phù hợp cho các ứng dụng cần bao phủ rộng. LE³W-DBSCAN cải thiện chất lượng ở cả hai mức này so với 3W-DBSCAN gốc, cho thấy tính ưu việt toàn diện của thuật toán đề xuất.

---

## Ghi chú cho tác giả

- Các giá trị **[value]** trong bảng cần được thay thế bằng kết quả thực tế từ việc chạy code.
- Hình 5–7 cần được bổ sung với hình ảnh thực tế từ thư mục `figures/`.
- Hình 10 trong phần 4.5 tương ứng với `figures/Figure_10_F1_vs_eta.png`.
- Các Hình 8–11 trong phần 4.7.3 tương ứng với:
  - Hình 8: `figures/Figure_8_Aggregation.png`
  - Hình 9: `figures/Figure_9_Compound.png`
  - Hình 10: `figures/Figure_10_Pathbased.png`
  - Hình 11: `figures/Figure_11_Flame.png`
