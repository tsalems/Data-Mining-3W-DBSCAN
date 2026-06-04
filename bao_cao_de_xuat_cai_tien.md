# CHƯƠNG 3: ĐỀ XUẤT CẢI TIẾN — THUẬT TOÁN LE3W-DBSCAN

> **Nguồn tham khảo chính:** Shen Qiuping, Zhang Qinghua, Gao Man, Dai Yongyang, *"Three-way DBSCAN Algorithm Based on Local Eps"*, Computer Science, 2023, 50(6): 100-108.

---

## 3.1. Hạn chế của 3W-DBSCAN

Dù thuật toán 3W-DBSCAN (Yu et al., 2019) đã đạt được những kết quả đáng kể, nó vẫn tồn tại hai hạn chế cơ bản cần được khắc phục:

### 3.1.1. Hạn chế về xử lý mật độ không đồng đều

3W-DBSCAN sử dụng **hàm co giãn khoảng cách (DScale)** để chuẩn hóa dữ liệu nhằm xử lý các cụm có mật độ khác nhau. Tuy nhiên, phương pháp này có nhược điểm:

- **Tham số $\eta$ quá nhạy cảm:** Hàm co giãn phụ thuộc vào tham số $\eta$ (bán kính lân cận). Thực nghiệm cho thấy với các bộ dữ liệu khác nhau, giá trị $\eta$ tối ưu biến động mạnh trong khoảng $[0.15, 0.3]$, rất khó xác định tự động.
- **Chuẩn hóa toàn cục:** DScale biến đổi **toàn bộ** ma trận khoảng cách theo một hàm duy nhất — cách tiếp cận toàn cục (global) này không thể thích ứng tốt khi từng cụm có đặc trưng mật độ riêng biệt.
- **Phụ thuộc vào kết quả ban đầu:** Chất lượng của biểu diễn ba chiều hoàn toàn phụ thuộc vào kết quả phân cụm hai chiều ban đầu của DBSCAN cải tiến.

### 3.1.2. Hạn chế về biểu diễn ba chiều quá mờ

Trong chiến lược xây dựng vùng biên của 3W-DBSCAN:

- **Toàn bộ điểm biên đều vào BND:** Mọi điểm có $S(x) = 0$ (điểm biên) đều được đưa vào vùng biên, bất kể tất cả láng giềng của nó đều thuộc cùng một cụm duy nhất — điều này dẫn đến vùng biên bị **phình to không cần thiết**.
- **Kết quả ba chiều quá mờ:** Khi quá nhiều điểm được đặt vào vùng biên, ý nghĩa của sự "mơ hồ" bị pha loãng. Vùng dương (POS) trở nên quá nhỏ, mất đi khả năng biểu diễn phần "chắc chắn" của cụm.
- **Dùng $\epsilon$ toàn cục để kiểm tra chồng lấp:** Khi mở rộng vùng biên (Chiến lược 2 trong 3W-DBSCAN), thuật toán sử dụng $\epsilon$ toàn cục để kiểm tra láng giềng — không phù hợp với dữ liệu đa mật độ vì $\epsilon$ quá nhỏ hoặc quá lớn với các cụm khác nhau.

> **[Chèn hình ảnh]:** Hình 1 trong bài báo Shen et al. (2023) — *"Examples of multi-density dataset"* — minh hoạ bộ dữ liệu Compound với cụm $C_1$ có mật độ thấp hơn rõ rệt so với các cụm còn lại, và vùng chồng lấp giữa $C_3$ và $C_4$ không có ranh giới rõ ràng.

---

## 3.2. Ý tưởng cải tiến

Để giải quyết hai hạn chế trên, Shen et al. (2023) đề xuất thuật toán **LE3W-DBSCAN** (Three-way DBSCAN Algorithm Based on **L**ocal **E**ps), với hai cải tiến cốt lõi:

| Hạn chế của 3W-DBSCAN | Cải tiến trong LE3W-DBSCAN |
|-----------------------|---------------------------|
| DScale toàn cục, tham số $\eta$ nhạy cảm | **Bán kính cục bộ (Local Eps)**: mỗi cụm có $\epsilon_j$ riêng, tự động tính từ k-kNN |
| Phân cụm không đảm bảo thứ tự mật độ | **Nguyên tắc mật độ giảm dần**: ưu tiên cụm mật độ cao trước |
| Toàn bộ điểm biên vào BND | **Tái phân loại dựa trên nhãn láng giềng**: điểm biên có thể được nâng lên POS |
| Dùng $\epsilon$ toàn cục kiểm tra chồng lấp | Dùng $\epsilon_j$ **cục bộ** của từng cụm để kiểm tra |

Thuật toán được chia thành hai giai đoạn:
1. **LE-DBSCAN**: phân cụm hai chiều với bán kính cục bộ
2. **LE3W-DBSCAN**: chuyển kết quả hai chiều thành biểu diễn ba chiều dựa trên nhãn láng giềng

---

## 3.3. Các định nghĩa và khái niệm mới

### 3.3.1. Hàm mật độ

**Định nghĩa 1 (Hàm mật độ):** Trong LE-DBSCAN, mật độ của điểm $p$ được định nghĩa là **trung bình khoảng cách từ $p$ đến $\text{MinPts}$ điểm lân cận gần nhất** của nó:

$$\text{dens}(p) = \frac{1}{\text{MinPts}} \sum_{i=1}^{\text{MinPts}} d(p, q_i) $$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\text{dens}(p)$ | Giá trị **mật độ** của điểm $p$ — giá trị càng nhỏ, mật độ càng cao |
| $\text{MinPts}$ | Tham số số lượng điểm tối thiểu, như trong DBSCAN gốc |
| $q_i$ | Điểm lân cận thứ $i$ gần nhất của $p$ |
| $d(p, q_i)$ | Khoảng cách Euclidean từ $p$ đến $q_i$ |

Lưu ý: giá trị $\text{dens}(p)$ nhỏ đồng nghĩa với việc $p$ có nhiều điểm gần nhau → **mật độ cao**. Ngược lại, $\text{dens}(p)$ lớn → khoảng cách trung bình lớn → **mật độ thấp**.

### 3.3.2. Khoảng cách $k$-lân cận ($k$-NN Distance)

**Định nghĩa 2 ($k$-lân cận khoảng cách):** Với tập đối tượng $X$ và điểm bất kỳ $p \in X$, xếp tất cả khoảng cách từ các điểm trong $X$ đến $p$ theo thứ tự tăng dần: $d_1 \leq d_2 \leq \ldots \leq d_k \leq \ldots \leq d_n$. Khoảng cách $k$-lân cận của $p$ là:

$$\text{dist}_k(p) = d_k $$

với $1 \leq k \leq n$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\text{dist}_k(p)$ | Khoảng cách từ $p$ đến điểm hàng xóm **thứ $k$** gần nhất |
| $k$ | Tham số thứ tự lân cận — **thay thế** cho tham số $\epsilon$ trong DBSCAN gốc |
| $d_k$ | Khoảng cách được xếp hạng thứ $k$ trong danh sách khoảng cách từ $p$ đến tất cả điểm trong $X$ |

**Ý nghĩa:** Nếu $p$ nằm trong vùng **mật độ cao** (nhiều điểm gần nhau), $\text{dist}_k(p)$ nhỏ. Nếu $p$ nằm trong vùng **mật độ thấp**, $\text{dist}_k(p)$ lớn.

### 3.3.3. Bán kính cục bộ (Local Eps)

**Định nghĩa 3 (Bán kính cục bộ):** Khi thực hiện LE-DBSCAN trên tập $X$, với mỗi cụm $C_j$, bán kính cục bộ $\text{Eps}_j$ được tính dựa trên mật độ của cụm đó. Bán kính cục bộ là $k$-khoảng cách lân cận của **điểm đầu tiên** được chọn để khởi tạo cụm $C_j$:

$$\text{Eps}_j = \text{dist}_k(p_j^{(1)}) $$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\text{Eps}_j$ | **Bán kính cục bộ** của cụm $C_j$ — đặc trưng cho mật độ riêng của cụm đó |
| $p_j^{(1)}$ | Điểm **đầu tiên** được chọn để khởi tạo cụm $C_j$ (điểm có mật độ cao nhất trong tập chưa phân loại) |
| $\text{dist}_k(p_j^{(1)})$ | Khoảng cách $k$-lân cận của điểm $p_j^{(1)}$ |
| $k$ | Tham số thứ tự lân cận — thông thường $k > \text{MinPts}$ và $k < n$ |

**Tại sao dùng điểm đầu tiên?** Vì điểm đầu tiên luôn là điểm có **mật độ cao nhất** trong tập chưa phân loại (theo nguyên tắc mật độ giảm dần). Khoảng cách $k$-lân cận của nó phản ánh mật độ điển hình của toàn cụm.

**So sánh với 3W-DBSCAN:** 3W-DBSCAN dùng một $\epsilon$ toàn cục sau khi co giãn DScale; LE3W-DBSCAN dùng $\text{Eps}_j$ khác nhau cho từng cụm → thích ứng tự nhiên với mật độ không đồng đều.

**Chuyển đổi tham số:** Thay vì phải chọn tham số $\epsilon$ nhạy cảm (trong DBSCAN gốc) hoặc tham số $\eta$ nhạy cảm (trong 3W-DBSCAN), người dùng chỉ cần chọn tham số $k$ (thứ tự lân cận), vốn ít nhạy cảm hơn nhiều.

### 3.3.4. Nguyên tắc mật độ giảm dần (Density Decreasing Principle)

**Định nghĩa 4:** Các cụm được xây dựng **theo thứ tự mật độ từ cao xuống thấp**: khi bắt đầu xây dựng cụm mới, điểm khởi đầu $p$ luôn là điểm có **mật độ cao nhất** trong tập các điểm chưa được phân loại.

$$p_j^{(1)} = \arg\max_{p \in X_{\text{unclassified}}} \text{dens}(p) \quad \Leftrightarrow \quad p_j^{(1)} = \arg\min_{p \in X_{\text{unclassified}}} \text{dens}_{\text{value}}(p) $$

**Giải thích:** Mật độ cao ↔ $\text{dens}(p)$ (trung bình khoảng cách đến $k$ láng giềng) **nhỏ** nhất.

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $X_{\text{unclassified}}$ | Tập hợp các điểm **chưa được phân loại** vào bất kỳ cụm nào |
| $p_j^{(1)}$ | Điểm khởi đầu cho cụm $C_j$ — điểm có mật độ cao nhất trong tập chưa phân loại |

**Tại sao quan trọng?** Nếu bắt đầu từ cụm mật độ thấp, bán kính cục bộ của nó sẽ rất lớn, có thể "nuốt" các cụm mật độ cao lân cận vào cùng cụm. Nguyên tắc mật độ giảm dần đảm bảo các cụm dày đặc được phát hiện trước, không bị ảnh hưởng bởi các cụm thưa hơn.

### 3.3.5. Định nghĩa điểm lõi, biên, nhiễu theo Local Eps

Với bán kính cục bộ $\text{Eps}_j$, ba loại điểm trong cụm $C_j$ được định nghĩa lại:

$$S_j(p) = \begin{cases}
1 & |N_{\text{Eps}_j}(p)| \geq \text{MinPts} \\
0 & 1 < |N_{\text{Eps}_j}(p)| < \text{MinPts} \\
-1 & |N_{\text{Eps}_j}(p)| = 1
\end{cases} \quad (5)$$

Trong đó: $S_j(p) = 1$ là điểm lõi, $S_j(p) = 0$ là điểm biên, $S_j(p) = -1$ là điểm nhiễu.

**Giải thích:** $N_{\text{Eps}_j}(p) = \{q \in X \mid d(p, q) \leq \text{Eps}_j\}$ là vùng $\text{Eps}_j$-lân cận **cục bộ** của điểm $p$ trong cụm $C_j$.

---

## 3.4. Thuật toán LE-DBSCAN (Phân cụm hai chiều với bán kính cục bộ)

LE-DBSCAN là nền tảng của LE3W-DBSCAN. Nó giải quyết vấn đề mật độ không đồng đều của DBSCAN gốc thông qua bán kính cục bộ và nguyên tắc mật độ giảm dần.

**Các bước thực hiện:**

**Bước 1:** Nhập tham số $\text{MinPts}$ và thứ tự lân cận $k$.

**Bước 2:** Tính **mật độ của tất cả điểm** theo hàm mật độ (Công thức 1) — lấy trung bình khoảng cách đến $\text{MinPts}$ điểm lân cận gần nhất.

**Bước 3:** Áp dụng **nguyên tắc mật độ giảm dần** — chọn điểm chưa phân loại có **mật độ cao nhất** (Công thức 4) làm điểm khởi đầu $p$ của cụm mới $C_j$.

**Bước 4:** Tính **bán kính cục bộ** $\text{Eps}_j$ của $C_j$ = khoảng cách $k$-lân cận của điểm $p$ (Công thức 3).

**Bước 5:** Dựa trên $\text{MinPts}$ và $\text{Eps}_j$, xác định tập điểm lõi, biên và nhiễu (Công thức 5). Từ điểm $p$, mở rộng cụm bằng cách tìm tất cả điểm tiếp cận được theo mật độ trong phạm vi $\text{Eps}_j$.

**Bước 6:** Tăng $j$ lên, lặp lại Bước 2–5 cho đến khi tất cả điểm (trừ nhiễu) đã được phân loại.

**Kết quả:** Các cụm $C_1, C_2, \ldots, C_c$ được tìm ra theo thứ tự từ mật độ cao đến thấp. Bảng 2 trong bài báo cho thấy bán kính cục bộ của từng cụm tăng dần theo thứ tự phát hiện, xác nhận tính hợp lệ của nguyên tắc mật độ giảm dần.

---

## 3.5. Tái phân loại ba chiều dựa trên nhãn láng giềng

Sau khi LE-DBSCAN cho kết quả hai chiều, LE3W-DBSCAN thực hiện **tái phân loại** điểm biên và điểm nhiễu dựa trên nhãn của các điểm lân cận.

### 3.5.1. Nguyên tắc tái phân loại

Điểm cốt lõi của cải tiến này so với 3W-DBSCAN:

> **Trong 3W-DBSCAN:** *Tất cả điểm biên* ($S(x) = 0$) đều được đưa vào vùng biên BND.
>
> **Trong LE3W-DBSCAN:** Điểm biên chỉ được đưa vào BND **khi thực sự có sự mơ hồ** (láng giềng thuộc nhiều cụm khác nhau). Nếu tất cả láng giềng thuộc cùng cụm → điểm được **nâng lên POS** (vùng dương).

### 3.5.2. Bốn bước tái phân loại

**Bước 1 — Chạy LE-DBSCAN:**
Thực hiện LE-DBSCAN để có kết quả phân cụm hai chiều $\{C_1, C_2, \ldots, C_c\}$.

**Bước 2 — Xử lý điểm lõi:**
Với mỗi điểm lõi $p$ ($|N_{\text{Eps}_j}(p)| \geq \text{MinPts}$) thuộc cụm $C_j$:

$$p \in C_j^P $$

Điểm lõi luôn được đưa vào **vùng dương** vì mật độ cao thể hiện sự chắc chắn trong cụm.

**Bước 3 — Tái phân loại điểm biên:**
Với điểm biên $p$ ($1 < |N_{\text{Eps}_j}(p)| < \text{MinPts}$) thuộc cụm $C_j$, tính vùng $\text{Eps}_j$-lân cận $N_{\text{Eps}_j}(p)$:

**Trường hợp (7a):** Nếu $\forall q \in N_{\text{Eps}_j}(p): q \in C_j$, thì:

$$\Rightarrow p \in C_j^P $$

**Trường hợp (7b):** Nếu $\exists q \in N_{\text{Eps}_j}(p): q \in C_m,\ m \neq j$, thì:

$$p \in C_j^B \quad \wedge \quad p \in C_m^B $$

**Giải thích Công thức (7a) và (7b):**

| Điều kiện | Hành động | Lý do |
|-----------|-----------|-------|
| **Toàn bộ** láng giềng $q$ thuộc cùng cụm $C_j$ | Đưa $p$ vào **POS($C_j$)** | Không có sự mơ hồ — $p$ chắc chắn thuộc $C_j$ |
| **Có ít nhất một** láng giềng $q$ thuộc cụm $C_m \neq j$ | Đưa $p$ vào **BND($C_j$) và BND($C_m$)** | $p$ nằm ở vùng chồng lấp giữa hai cụm |

**Bước 4 — Xử lý điểm nhiễu:**
Với mỗi điểm nhiễu $p$ ($|N_{\text{Eps}_j}(p)| = 1$), tìm điểm lõi gần nhất $q$:

$$q = \arg\min_{q \in \text{AllPOS}} d(p, q) $$

Nếu $q \in C_j^P$, thì:

$$p \in C_j^B $$

**Ý nghĩa:** Điểm nhiễu không đủ mật độ để thuộc POS, nhưng do gần với một cụm xác định, nó được gán vào BND của cụm đó — thể hiện quan hệ mơ hồ với cụm gần nhất.

### 3.5.3. So sánh trực tiếp với 3W-DBSCAN

| Khía cạnh | 3W-DBSCAN (Yu 2019) | LE3W-DBSCAN (Shen 2023) |
|-----------|---------------------|--------------------------|
| Xử lý mật độ không đều | DScale toàn cục (tham số $\eta$) | Bán kính cục bộ $\text{Eps}_j$ (tham số $k$) |
| Thứ tự phân cụm | Ngẫu nhiên | Mật độ giảm dần |
| Điểm biên → POS | Không (luôn → BND) | **Có**, nếu toàn bộ láng giềng cùng cụm |
| Kiểm tra chồng lấp | Dùng $\epsilon$ toàn cục | Dùng $\text{Eps}_j$ cục bộ |
| Vùng biên | Có thể quá lớn | Chính xác hơn, chỉ giữ điểm thực sự mơ hồ |

---

## 3.6. Thuật toán LE3W-DBSCAN hoàn chỉnh

### 3.6.1. Mã giả của thuật toán

```
Algorithm 1: LE3W-DBSCAN

Input : Tập dữ liệu X = {x₁, x₂, ..., xₙ}
        Tham số MinPts (số điểm tối thiểu)
        Tham số k (thứ tự lân cận để tính bán kính cục bộ)

Output: Kết quả ba chiều Cⱼ = (CⱼP, CⱼB), 1 ≤ j ≤ c
        Trong đó CⱼP: vùng dương, CⱼB: vùng biên của cụm Cⱼ

─────────────────────────────────────────────────────────
//============ GIAI ĐOẠN 1: LE-DBSCAN ============//

BƯỚC 1: Tính ma trận khoảng cách
  DistanceMatrix ← getDistanceMatrix(X)
  // Khoảng cách Euclidean giữa mọi cặp điểm
  // Độ phức tạp: O(n²)

BƯỚC 2: Tính mật độ tất cả điểm
  DensityMatrix ← getDensityMatrix(MinPts, DistanceMatrix)
  // dens(p) = trung bình khoảng cách đến MinPts điểm gần nhất
  // Độ phức tạp: O(n²)

BƯỚC 3: Phân cụm theo nguyên tắc mật độ giảm dần
  clusterId ← 0
  WHILE còn tồn tại điểm chưa phân loại:
    clusterId ← clusterId + 1

    // Bước 3a: Chọn điểm mật độ cao nhất chưa phân loại
    pointId ← getMostDensePoint(DensityMatrix)

    // Bước 3b: Tính bán kính cục bộ cho cụm mới
    Eps_clusterId ← getEps(pointId, DistanceMatrix, k)
    //  = dist_k(pointId) — khoảng cách k-lân cận

    // Bước 3c: Mở rộng cụm từ pointId
    twoWayResult ← expandCluster(
        pointId, clusterId, Eps_clusterId, MinPts
    )
    // Tìm tất cả điểm tiếp cận được theo mật độ
    // trong phạm vi Eps_clusterId
  END WHILE
  // Kết quả: c cụm {C₁, C₂, ..., Cc} + tập điểm nhiễu

//============ GIAI ĐOẠN 2: Tái phân loại ba chiều ============//

BƯỚC 4: Tái phân loại từng điểm dựa trên nhãn láng giềng
  FOR i = 0 TO n-1:
    clusterId ← getClassLabel(twoWayResult, xᵢ)
    // Lấy nhãn cụm của xᵢ từ kết quả hai chiều

    Eps_i ← Eps_clusterId  // Bán kính cục bộ của cụm chứa xᵢ

    threeWayResult ← Reclassify(xᵢ, Eps_i)
    /*
      Nếu xᵢ là điểm lõi:
        → xᵢ ∈ POS(Cⱼ)                          // Công thức (6)

      Nếu xᵢ là điểm biên:
        Tính N_Eps_j(xᵢ) = {q | d(xᵢ,q) ≤ Eps_j}
        IF ∀q ∈ N_Eps_j(xᵢ): q ∈ Cⱼ:
          → xᵢ ∈ POS(Cⱼ)                         // Công thức (7)
        ELSE (∃q ∈ N_Eps_j(xᵢ): q ∈ Cₘ, m≠j):
          → xᵢ ∈ BND(Cⱼ) VÀ xᵢ ∈ BND(Cₘ)        // Công thức (8)

      Nếu xᵢ là điểm nhiễu:
        q ← argmin_{q ∈ AllPOS} d(xᵢ, q)          // Công thức (9 — NCN)
        IF q ∈ Cⱼ: → xᵢ ∈ BND(Cⱼ)               // Công thức (9)
    */
  END FOR

BƯỚC 5: Xây dựng kết quả cuối
  FOR mỗi cụm Cⱼ:
    CⱼP ← tập hợp điểm trong vùng dương của Cⱼ
    CⱼB ← tập hợp điểm trong vùng biên của Cⱼ

RETURN threeWayResult = {(C₁P, C₁B), (C₂P, C₂B), ..., (CcP, CcB)}
─────────────────────────────────────────────────────────
```

### 3.6.2. Lưu đồ thuật toán

> **[Chèn hình ảnh]:** Hình 2 trong bài báo Shen et al. (2023) — *"Flowchart of LE3W-DBSCAN"* — thể hiện toàn bộ quy trình hai giai đoạn: giai đoạn LE-DBSCAN (phân cụm hai chiều với bán kính cục bộ) và giai đoạn tái phân loại ba chiều.

Dưới đây là mô tả lưu đồ dạng văn bản:

```
     ┌──────────────────────────────────────┐
     │  BẮT ĐẦU                             │
     │  Input: X, MinPts, k                 │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │  Tính ma trận khoảng cách            │
     │  DistanceMatrix ← getDistMatrix(X)   │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │  Tính mật độ tất cả điểm             │
     │  dens(p) = avg dist đến MinPts NN    │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │  Còn điểm chưa phân loại?            │
     └──────────┬──────────────┬────────────┘
                │ Có            │ Không
                ▼               │
     ┌──────────────────────┐   │
     │  Chọn điểm mật độ    │   │
     │  cao nhất → p        │   │
     └──────────┬───────────┘   │
                │               │
                ▼               │
     ┌──────────────────────┐   │
     │  Tính Eps_j =        │   │
     │  dist_k(p) [C.t. 3]  │   │
     └──────────┬───────────┘   │
                │               │
                ▼               │
     ┌──────────────────────┐   │
     │  Mở rộng cụm Cⱼ      │   │
     │  từ p với Eps_j      │   │
     │  và MinPts           │   │
     └──────────┬───────────┘   │
                │               │
                └───────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────┐
     │  Kết quả 2 chiều: C₁,...,Cc + nhiễu  │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────────────┐
     │  FOR mỗi điểm xᵢ:                           │
     │    - Điểm lõi → POS(Cⱼ)  [Công thức 6]     │
     │    - Điểm biên:                             │
     │       Tất cả NN ∈ Cⱼ → POS(Cⱼ) [C.t. 7a]  │
     │       Có NN ∈ Cₘ(m≠j) → BND(Cⱼ)∩BND(Cₘ)  │
     │                          [Công thức 7b]     │
     │    - Điểm nhiễu → BND(cụm NN lõi gần nhất) │
     │                   [Công thức 8, 9]          │
     └────────────────┬─────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │  KẾT THÚC                            │
     │  Output: {(C₁P,C₁B),...,(CcP,CcB)}  │
     └──────────────────────────────────────┘
```

### 3.6.3. Phân tích độ phức tạp thời gian

| Bước | Thao tác | Độ phức tạp |
|------|----------|------------|
| Bước 1 | Tính ma trận khoảng cách | $O(n^2)$ |
| Bước 2 | Tính mật độ tất cả điểm | $O(n^2)$ |
| Bước 3 | LE-DBSCAN (while loop với mở rộng cụm) | $O(n^2)$ worst case |
| Bước 4 | Tái phân loại $n$ điểm, mỗi điểm kiểm tra láng giềng | $O(n^2)$ |

**Độ phức tạp tổng thể của LE3W-DBSCAN:** $O(n^2)$ — tương đương với 3W-DBSCAN.

---

## 3.7. Bộ dữ liệu thực nghiệm

Bài báo Shen et al. (2023) sử dụng **11 bộ dữ liệu** gồm **5 bộ dữ liệu nhân tạo** và **6 bộ dữ liệu UCI thực tế**.

### Bảng 1: Thông tin 11 bộ dữ liệu thực nghiệm

| STT | Tên bộ dữ liệu | Số cụm | Số chiều | Số mẫu | Loại |
|-----|----------------|--------|----------|--------|------|
| 1 | Aggregation | 7 | 2 | 788 | Nhân tạo |
| 2 | Compound | 6 | 2 | 399 | Nhân tạo |
| 3 | Pathbased | 3 | 2 | 300 | Nhân tạo |
| 4 | Jain | 2 | 2 | 373 | Nhân tạo |
| 5 | Flame | 2 | 2 | 240 | Nhân tạo |
| 6 | Dermatology | 6 | 34 | 366 | UCI thực tế |
| 7 | Ecoli | 5 | 7 | 210 | UCI thực tế |
| 8 | Iris | 3 | 4 | 150 | UCI thực tế |
| 9 | Seeds | 3 | 7 | 197 | UCI thực tế |
| 10 | Wine | 3 | 13 | 178 | UCI thực tế |
| 11 | Haberman | 2 | 3 | 306 | UCI thực tế |

### 3.7.1. Mô tả bộ dữ liệu nhân tạo

**Aggregation (7 cụm, 2D, 788 điểm):** Bảy cụm hình tròn với mật độ tương đối đồng đều. Thách thức nằm ở các vùng kết nối giữa $C_2$–$C_3$ và $C_4$–$C_5$, nơi các điểm có mức độ chồng lấp cao.

**Compound (6 cụm, 2D, 399 điểm):** Bộ dữ liệu khó nhất trong nhóm nhân tạo — hỗn hợp hình dạng và mật độ: cụm $C_1$ có mật độ thấp rõ rệt so với 5 cụm còn lại, cụm $C_3$ và $C_4$ không có ranh giới rõ ràng. Thích hợp để kiểm tra nguyên tắc mật độ giảm dần.

> **[Chèn hình ảnh]:** Hình 4 trong bài báo — *"Experimental results of two-way clustering of Compound"* — so sánh kết quả phân cụm hai chiều của DBSCAN, KR-DBSCAN, 3W-DBSCAN và LE-DBSCAN trên Compound. Đặc biệt cụm $C_1$ (mật độ thấp) được gán đúng trong LE-DBSCAN (trở thành $C_6$ theo thứ tự mật độ giảm dần).

**Pathbased (3 cụm, 2D, 300 điểm):** Hai cụm Gaussian cộng một cụm tròn được nối với nhau bằng các chuỗi điểm mỏng.

**Jain (2 cụm, 2D, 373 điểm):** Hai cụm hình dạng bất quy tắc — một cụm hình cung bao xung quanh một cụm dạng elip.

**Flame (2 cụm, 2D, 240 điểm):** Phân phối giống ngọn lửa, hai cụm kết nối với nhau.

### 3.7.2. Mô tả bộ dữ liệu UCI thực tế

**Dermatology (6 lớp, 34 chiều, 366 mẫu):** Dữ liệu chẩn đoán các bệnh da liễu. Số chiều lớn nhất (34) trong tập thực nghiệm — kiểm tra khả năng xử lý dữ liệu chiều cao.

**Ecoli (5 lớp, 7 chiều, 210 mẫu):** Dữ liệu chuỗi protein E.coli, các lớp có kích thước không cân bằng đáng kể.

**Iris (3 lớp, 4 chiều, 150 mẫu):** Bộ dữ liệu cổ điển — ba loài hoa Iris với một cụm tách biệt hoàn toàn và hai cụm chồng lấp.

**Seeds (3 lớp, 7 chiều, 197 mẫu):** Thông số hình học của ba giống hạt lúa mì. Lưu ý: bài báo này dùng 197 mẫu (khác với 210 mẫu trong yu2019 — có thể do loại bỏ outlier).

**Wine (3 lớp, 13 chiều, 178 mẫu):** Phân tích hóa học của rượu vang từ 3 vùng trồng nho khác nhau.

**Haberman (2 lớp, 3 chiều, 306 mẫu):** Dữ liệu sống sót sau phẫu thuật ung thư vú — hai lớp mất cân bằng (225 sống sót / 81 không sống sót).

---

## 3.8. Các độ đo đánh giá

Bài báo Shen et al. (2023) sử dụng hai nhóm chỉ số: **chỉ số phân cụm cứng** (đánh giá LE-DBSCAN) và **chỉ số phân cụm mềm** (đánh giá LE3W-DBSCAN).

### 3.8.1. Chỉ số phân cụm cứng

**1. Accuracy (ACC)** — tương tự Chương 2, đo tỷ lệ đối tượng được gán đúng cụm.

**2. Chỉ số Rand hiệu chỉnh — ARI (Adjusted Rand Index)**

$$\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]} $$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\text{RI}$ | **Rand Index** — tỷ lệ cặp điểm được phân loại nhất quán (cùng cụm hoặc khác cụm trong cả kết quả dự đoán lẫn ground truth) |
| $E[\text{RI}]$ | Giá trị kỳ vọng của RI khi phân cụm ngẫu nhiên |
| $\max(\text{RI})$ | Giá trị RI tối đa có thể đạt được |
| $\text{ARI} \in [-1, 1]$ | ARI = 1: hoàn toàn khớp; ARI = 0: tương đương phân cụm ngẫu nhiên; ARI < 0: tệ hơn ngẫu nhiên |

**Ưu điểm của ARI:** Khắc phục nhược điểm của RI — RI luôn dương ngay cả khi phân cụm ngẫu nhiên; ARI chuẩn hóa về kỳ vọng nên phản ánh chính xác hơn.

**3. Normalized Mutual Information (NMI)** — tương tự Chương 2, đã trình bày tại Công thức (13).

### 3.8.2. Chỉ số phân cụm mềm (Soft Clustering Metrics)

Đây là nhóm chỉ số **mới** so với Chapter 2, dùng để đánh giá đặc trưng của kết quả phân cụm ba chiều — cụ thể là **chất lượng của vùng dương** so với vùng biên.

**1. Tỷ lệ vùng dương trung bình $\gamma$ (Maji et al.)**

$$\gamma = \frac{1}{n} \sum_{j=1}^{c} |C_j^P| $$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\gamma$ | Tỷ lệ **trung bình trên toàn bộ dữ liệu** của các điểm thuộc vùng dương |
| $n$ | Tổng số điểm trong tập dữ liệu |
| $c$ | Tổng số cụm |
| $C_j^P$ | Vùng dương của cụm $C_j$ |
| $|C_j^P|$ | Số lượng điểm trong vùng dương của cụm $C_j$ |
| $\gamma \in [0, 1]$ | Giá trị càng lớn → phần lớn điểm có sự xác định rõ ràng → phân cụm tốt hơn |

$\gamma$ đo lường **tỷ lệ điểm "chắc chắn"** trong toàn bộ dữ liệu. Giá trị $\gamma$ cao nghĩa là ít điểm mơ hồ, kết quả ba chiều không quá mờ.

**2. Chất lượng trung bình từng cụm $\alpha$ (Zhang et al.)**

$$\alpha = \frac{1}{c} \sum_{j=1}^{c} \frac{|C_j^P|}{|C_j^P| + |C_j^B|} $$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\alpha$ | Trung bình của **tỷ lệ vùng dương** trên từng cụm |
| $|C_j^P|$ | Số điểm trong vùng dương của cụm $C_j$ |
| $|C_j^B|$ | Số điểm trong vùng biên của cụm $C_j$ |
| $\frac{|C_j^P|}{|C_j^P| + |C_j^B|}$ | Tỷ lệ điểm "chắc chắn" trong cụm $C_j$ — bằng 1 nếu không có vùng biên |
| $\alpha \in [0, 1]$ | Giá trị càng lớn → mỗi cụm càng có nhiều điểm chắc chắn → tốt hơn |

$\alpha$ xét từng cụm một rồi lấy trung bình — phù hợp khi các cụm có kích thước không đồng đều.

**3. Chất lượng tổng thể $\alpha^*$ (Zhang et al.)**

$$\alpha^* = \frac{\sum_{j=1}^{c} |C_j^P|}{\sum_{j=1}^{c} \left(|C_j^P| + |C_j^B|\right)} $$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\alpha^*$ | Tỷ lệ vùng dương **trên toàn bộ** tất cả cụm (không trung bình mà tính tổng thể) |
| Tử số $\sum |C_j^P|$ | Tổng số điểm thuộc vùng dương của **tất cả cụm** |
| Mẫu số $\sum (|C_j^P| + |C_j^B|)$ | Tổng số điểm thuộc vùng dương **hoặc** vùng biên của tất cả cụm |
| $\alpha^* \in [0, 1]$ | Giá trị càng lớn → kết quả ba chiều càng xác định → tốt hơn |

**So sánh $\alpha$ và $\alpha^*$:**
- $\alpha$ tính **trung bình không trọng số** qua các cụm → bị ảnh hưởng bởi cụm nhỏ
- $\alpha^*$ tính **tổng trực tiếp** → trọng số tỷ lệ với kích thước cụm

**Tóm tắt ý nghĩa ba chỉ số phân cụm mềm:**

> Cả ba chỉ số $\alpha$, $\alpha^*$, $\gamma$ đều đo **mức độ "rõ ràng"** của kết quả ba chiều. Giá trị cao = ít điểm mơ hồ = vùng dương lớn và vùng biên nhỏ = thuật toán phân biệt được nhiều điểm chắc chắn hơn, không để quá nhiều điểm vào vùng biên một cách không cần thiết.

---

## 3.9. Kết quả thực nghiệm

### 3.9.1. Kết quả phân cụm hai chiều (LE-DBSCAN)

Bảng 2 trong bài báo gốc cung cấp bán kính cục bộ $\text{Eps}_j$ mà LE-DBSCAN tính được cho từng cụm:

**Bảng 2: Bán kính cục bộ tính được bởi LE-DBSCAN**

| STT | Bộ dữ liệu | $\text{Eps}_{C_1}$ | $\text{Eps}_{C_2}$ | $\text{Eps}_{C_3}$ | $\text{Eps}_{C_4}$ | $\text{Eps}_{C_5}$ | $\text{Eps}_{C_6}$ | $\text{Eps}_{C_7}$ |
|-----|------------|----------|----------|----------|----------|----------|----------|----------|
| 1 | Aggregation | 0.0513 | 0.0573 | 0.0633 | 0.0690 | 0.0735 | 0.0735 | 0.0635 |
| 2 | Compound | 0.0495 | 0.0556 | 0.0391 | 0.0363 | 0.0401 | **0.1588** | — |
| 3 | Pathbased | 0.0561 | 0.0733 | **0.1998** | — | — | — | — |
| 4 | Jain | 0.0666 | **0.0949** | — | — | — | — | — |
| 5 | Flame | 0.0628 | 0.0680 | — | — | — | — | — |
| 6 | Dermatology | 1.0628 | 1.5580 | 1.4744 | 1.7321 | 1.7046 | **2.3572** | — |
| 7 | Ecoli | 0.1726 | 0.2411 | 0.4304 | 0.4982 | **1.0266** | — | — |
| 8 | Iris | 0.3498 | 0.2927 | **0.6089** | — | — | — | — |
| 9 | Seeds | 0.3092 | 0.3993 | **0.5265** | — | — | — | — |
| 10 | Wine | 0.5970 | 0.9734 | **1.1905** | — | — | — | — |
| 11 | Haberman | 0.1926 | **0.4931** | — | — | — | — | — |

*Các giá trị in đậm là bán kính cục bộ lớn nhất — tương ứng với cụm có mật độ thấp nhất, được phát hiện sau cùng.*

**Nhận xét:** Trên các bộ dữ liệu mật độ không đồng đều (Compound, Pathbased, Ecoli, Wine, Haberman), bán kính cục bộ **tăng dần** theo thứ tự phát hiện — xác nhận tính đúng đắn của nguyên tắc mật độ giảm dần. Trên Flame (mật độ đồng đều), hai bán kính gần bằng nhau (0.0628 ≈ 0.0680).

**Bảng 3: Kết quả phân cụm hai chiều (Hard Clustering)**

| STT | Bộ dữ liệu | DBSCAN | KR-DBSCAN | 3W-DBSCAN | **LE-DBSCAN** |
|-----|------------|--------|-----------|-----------|---------------|
| | | ACC / ARI / NMI | ACC / ARI / NMI | ACC / ARI / NMI | ACC / ARI / NMI |
| 1 | Aggregation | 0.992 / 0.989 / 0.983 | 0.997 / 0.995 / 0.992 | 0.997 / 0.995 / 0.992 | **0.997 / 0.995 / 0.992** |
| 2 | Compound | 0.822 / 0.939 / 0.896 | 0.925 / 0.889 / 0.896 | 0.782 / 0.760 / 0.777 | **0.952 / 0.947 / 0.918** |
| 3 | Pathbased | 0.847 / 0.707 / 0.738 | 0.967 / 0.901 / 0.870 | 0.707 / 0.548 / 0.689 | **0.957 / 0.872 / 0.847** |
| 4 | Jain | 0.756 / 0.934 / 0.833 | **1.000 / 1.000 / 1.000** | **1.000 / 1.000 / 1.000** | **1.000 / 1.000 / 1.000** |
| 5 | Flame | 0.938 / 0.892 / 0.813 | 0.996 / 0.983 / 0.963 | 0.992 / 0.967 / 0.927 | **0.996 / 0.983 / 0.963** |
| 6 | Dermatology | 0.489 / 0.371 / 0.574 | 0.738 / 0.728 / 0.860 | 0.658 / 0.600 / 0.707 | **0.825 / 0.741 / 0.797** |
| 7 | Ecoli | 0.489 / 0.432 / 0.441 | 0.596 / 0.392 / 0.514 | **0.679 / 0.528 / 0.586** | 0.554 / 0.639 / 0.584 |
| 8 | Iris | 0.740 / 0.612 / 0.640 | 0.840 / 0.760 / 0.792 | 0.907 / 0.759 / 0.806 | **0.907 / 0.857 / 0.866** |
| 9 | Seeds | 0.652 / 0.502 / 0.582 | 0.600 / 0.460 / 0.553 | 0.624 / 0.489 / 0.605 | **0.738 / 0.490 / 0.534** |
| 10 | Wine | 0.562 / 0.439 / 0.564 | 0.360 / 0.461 / 0.599 | 0.612 / 0.462 / 0.600 | **0.764 / 0.632 / 0.627** |
| 11 | Haberman | 0.621 / 0.159 / 0.063 | 0.745 / 0.100 / 0.055 | 0.735 / 0.151 / 0.059 | **0.758 / 0.159 / 0.071** |

*Giá trị in đậm là kết quả tốt nhất trên mỗi bộ dữ liệu.*

### 3.9.2. Kết quả phân cụm ba chiều (LE3W-DBSCAN)

**Bảng 4: Kết quả phân cụm ba chiều (Soft Clustering Metrics)**

| STT | Bộ dữ liệu | 3W-DBSCAN $\alpha$ | **LE3W-DBSCAN $\alpha$** | 3W-DBSCAN $\alpha^*$ | **LE3W-DBSCAN $\alpha^*$** | 3W-DBSCAN $\gamma$ | **LE3W-DBSCAN $\gamma$** |
|-----|------------|-------------------|--------------------------|---------------------|---------------------------|-------------------|--------------------------|
| 1 | Aggregation | 0.889 | **0.977** | 0.845 | **0.982** | 0.854 | **0.991** |
| 2 | Compound | 0.824 | **0.916** | 0.962 | 0.958 | 0.962 | 0.960 |
| 3 | Pathbased | 0.807 | **0.955** | 0.891 | **0.954** | 0.903 | **0.977** |
| 4 | Jain | 0.995 | **1.000** | 0.997 | **1.000** | 0.997 | **1.000** |
| 5 | Flame | 0.919 | **0.924** | 0.925 | **0.938** | 0.929 | **0.942** |
| 6 | Dermatology | 0.258 | **0.858** | 0.314 | **0.867** | 0.314 | **0.929** |
| 7 | Ecoli | 0.343 | **0.609** | 0.524 | **0.624** | 0.532 | **0.783** |
| 8 | Iris | 0.474 | **0.921** | 0.487 | **0.923** | 0.487 | **0.960** |
| 9 | Seeds | 0.550 | **0.626** | **0.734** | 0.638 | **0.762** | 0.790 |
| 10 | Wine | 0.595 | **0.883** | 0.709 | **0.884** | 0.713 | **0.944** |
| 11 | Haberman | 0.959 | **0.970** | 0.974 | **0.987** | 0.987 | **0.993** |

*Giá trị in đậm là kết quả tốt nhất trên mỗi bộ dữ liệu.*

### 3.9.3. Phân tích và nhận xét kết quả

#### Về kết quả phân cụm hai chiều (LE-DBSCAN):

1. **Vượt trội rõ rệt trên dữ liệu đa mật độ:** Compound (ACC: 0.952 vs 0.782 của 3W-DBSCAN), Pathbased (0.957 vs 0.707), Wine (0.764 vs 0.612), Dermatology (0.825 vs 0.658). Đây đều là bộ dữ liệu có mật độ không đồng đều — xác nhận hiệu quả của bán kính cục bộ.

2. **Ngoại lệ — Ecoli:** 3W-DBSCAN đạt ACC = 0.679 > LE-DBSCAN = 0.554 trên Ecoli. Nguyên nhân: Ecoli có các cụm kích thước rất không cân bằng, và phân phối mật độ phức tạp. Tuy nhiên, xét ARI, LE-DBSCAN đạt 0.639 > 3W-DBSCAN = 0.528 — cho thấy sự không nhất quán giữa ACC và ARI.

3. **Bán kính cục bộ xác nhận nguyên tắc mật độ giảm dần:** Như thể hiện trong Bảng 2, trên Compound, cụm $C_1$ (mật độ thấp nhất) được gán bán kính 0.1588 — lớn hơn nhiều so với các cụm khác (0.036–0.056). Nếu cụm này được phát hiện trước, bán kính lớn sẽ hút nhầm các điểm từ cụm khác.

> **[Chèn hình ảnh]:** Hình 3 trong bài báo — *"Experimental results of two-way clustering of Aggregation"* — so sánh DBSCAN, KR-DBSCAN, 3W-DBSCAN và LE-DBSCAN.

> **[Chèn hình ảnh]:** Hình 4 trong bài báo — *"Experimental results of two-way clustering of Compound"* — thể hiện rõ lợi thế của LE-DBSCAN với cụm $C_1$ mật độ thấp.

> **[Chèn hình ảnh]:** Hình 5 trong bài báo — *"Experimental results of two-way clustering of Pathbased"*.

> **[Chèn hình ảnh]:** Hình 6 trong bài báo — *"Experimental results of two-way clustering of Jain"*.

> **[Chèn hình ảnh]:** Hình 7 trong bài báo — *"Experimental results of two-way clustering of Flame"*.

#### Về kết quả phân cụm ba chiều (LE3W-DBSCAN):

1. **Cải thiện mạnh trên chỉ số mềm:** LE3W-DBSCAN cho kết quả tốt hơn 3W-DBSCAN trên **hầu hết** bộ dữ liệu ở cả ba chỉ số $\alpha$, $\alpha^*$, $\gamma$. Cải thiện đặc biệt lớn trên Dermatology ($\gamma$: 0.929 vs 0.314) và Iris ($\gamma$: 0.960 vs 0.487).

2. **Vùng biên chính xác hơn:** Cải tiến tái phân loại điểm biên (Công thức 7a) cho phép nhiều điểm biên được "nâng cấp" lên vùng dương khi không có sự chồng lấp thực sự → $\alpha$, $\alpha^*$, $\gamma$ cao hơn. LE3W-DBSCAN **chỉ giữ điểm thực sự mơ hồ trong vùng biên** — cân bằng tốt hơn giữa tính mờ và tính xác định.

3. **Ví dụ minh hoạ — Aggregation:** Với Aggregation, 3W-DBSCAN tạo ra vùng chồng lấp rộng giữa $C_4$–$C_5$ và $C_2$–$C_3$ ($\alpha = 0.889$), trong khi LE3W-DBSCAN chỉ đưa những điểm thực sự nằm ở vùng giao nhau vào BND, đạt $\alpha = 0.977$.

4. **Ngoại lệ — Seeds:** 3W-DBSCAN có $\alpha^* = 0.734 >$ LE3W-DBSCAN = 0.638. Nguyên nhân tương tự như phân tích trên Ecoli — phân phối Seeds phù hợp tự nhiên hơn với cách phân vùng của 3W-DBSCAN.

> **[Chèn hình ảnh]:** Hình 8 trong bài báo — *"Experimental results of three-way clustering of Aggregation"* — so sánh kết quả ba chiều của 3W-DBSCAN và LE3W-DBSCAN.

> **[Chèn hình ảnh]:** Hình 9 trong bài báo — *"Experimental results of three-way clustering of Compound"*.

> **[Chèn hình ảnh]:** Hình 10 trong bài báo — *"Experimental results of three-way clustering of Pathbased"*.

> **[Chèn hình ảnh]:** Hình 11 trong bài báo — *"Experimental results of three-way clustering of Flame"*.

---

## 3.10. So sánh tổng hợp 3W-DBSCAN và LE3W-DBSCAN

**Bảng 5: So sánh toàn diện hai thuật toán**

| Tiêu chí | 3W-DBSCAN (Yu 2019) | LE3W-DBSCAN (Shen 2023) |
|----------|---------------------|--------------------------|
| **Xử lý mật độ không đều** | DScale toàn cục | Bán kính cục bộ $\text{Eps}_j$ |
| **Số tham số** | $\epsilon$, MinPts, $\eta$ (3 tham số) | $k$, MinPts (2 tham số) |
| **Độ nhạy tham số** | Cao ($\eta$ khó chọn) | Thấp hơn ($k$ tự nhiên hơn) |
| **Thứ tự phân cụm** | Ngẫu nhiên | Mật độ giảm dần (có hệ thống) |
| **Điểm biên → POS** | Không | Có (khi toàn bộ NN cùng cụm) |
| **Kiểm tra chồng lấp** | $\epsilon$ toàn cục | $\text{Eps}_j$ cục bộ |
| **Vùng biên** | Có thể quá mờ | Cân bằng mờ-xác định |
| **Độ phức tạp** | $O(n^2)$ | $O(n^2)$ |
| **Chỉ số cứng (ACC)** | Kém hơn trên đa mật độ | Tốt hơn trên hầu hết bộ dữ liệu |
| **Chỉ số mềm ($\alpha$, $\gamma$)** | Thấp hơn | Cao hơn rõ rệt |

---

## 3.11. Tóm tắt chương

Chương này đã trình bày phương pháp cải tiến **LE3W-DBSCAN** (Shen et al., 2023) so với thuật toán gốc 3W-DBSCAN (Yu et al., 2019):

1. **Hai hạn chế của 3W-DBSCAN được xác định:** (1) DScale toàn cục với tham số $\eta$ nhạy cảm; (2) Vùng biên quá mờ do đưa toàn bộ điểm biên vào BND bất kể láng giềng.

2. **Bán kính cục bộ và nguyên tắc mật độ giảm dần:** Mỗi cụm có bán kính $\text{Eps}_j$ riêng được tính tự động từ khoảng cách $k$-lân cận, phân cụm theo thứ tự từ mật độ cao đến thấp → xử lý tốt dữ liệu đa mật độ.

3. **Tái phân loại dựa trên nhãn láng giềng:** Điểm biên được phân vào POS (nếu tất cả láng giềng đồng nhất cụm) hoặc BND (nếu có láng giềng từ cụm khác) → kết quả ba chiều chính xác hơn, không quá mờ.

4. **Kết quả thực nghiệm trên 11 bộ dữ liệu** xác nhận: LE-DBSCAN vượt trội trong phân cụm cứng (đặc biệt trên dữ liệu đa mật độ), và LE3W-DBSCAN đạt chỉ số mềm $\alpha$, $\alpha^*$, $\gamma$ cao hơn đáng kể so với 3W-DBSCAN trên hầu hết bộ dữ liệu.
