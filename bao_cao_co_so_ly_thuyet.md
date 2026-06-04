# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

> **Nguồn tham khảo chính:** Hui Yu, LuYuan Chen, JingTao Yao, XingNan Wang, *"A three-way clustering method based on an improved DBSCAN algorithm"*, Physica A 535 (2019) 122289, Elsevier.

---

## 2.1. Tổng quan về phân cụm

Phân cụm (clustering) là một kỹ thuật học máy không giám sát (unsupervised learning) với mục tiêu nhóm các đối tượng có độ tương đồng cao vào cùng một cụm, trong khi các đối tượng khác nhau thuộc về các cụm khác nhau. Không giống với phân loại (classification), phân cụm không sử dụng nhãn đã biết trước mà tự khám phá cấu trúc ẩn trong dữ liệu.

Hầu hết các phương pháp phân cụm truyền thống (hard clustering) yêu cầu mỗi đối tượng phải được gán vào đúng một cụm duy nhất. Tuy nhiên, trong thực tế, ranh giới giữa các cụm thường không rõ ràng — một đối tượng có thể đồng thời nằm gần nhiều cụm, hoặc có mức độ thuộc về mỗi cụm khác nhau. Điều này dẫn đến nguy cơ mắc lỗi cao khi áp dụng hard clustering trong các tình huống dữ liệu không đồng nhất.

Bài báo Yu et al. (2019) đề xuất phương pháp **3W-DBSCAN** — kết hợp **lý thuyết quyết định ba chiều** (three-way decision theory) với **thuật toán DBSCAN cải tiến** — để biểu diễn một cụm bằng cặp tập hợp lồng nhau, từ đó phân loại mỗi đối tượng thành một trong ba trạng thái: *thuộc về*, *không thuộc về*, hoặc *mơ hồ*. Cách biểu diễn này phù hợp hơn với nhận thức của con người và xử lý tốt hơn các đối tượng nằm ở vùng biên.

---

## 2.2. Lý thuyết quyết định ba chiều (Three-way Decision Theory)

### 2.2.1. Khái niệm cơ bản

Lý thuyết quyết định ba chiều (three-way decision) được đề xuất bởi Yao [19, 20] như một mở rộng của mô hình quyết định nhị phân thông thường. Ý tưởng cốt lõi là: khi thông tin hiện có không đủ hoặc không chắc chắn để đưa ra một quyết định xác định (có hoặc không), người ta có thể chọn phương án thứ ba là **hoãn quyết định** (deferment) cho đến khi có đủ thông tin.

Lý thuyết này chia tập vũ trụ $V$ thành **ba vùng phân biệt, không giao nhau**:

| Vùng | Ký hiệu | Ý nghĩa | Hành động |
|------|---------|---------|-----------|
| Vùng dương (Positive region) | POS | Đối tượng chắc chắn thuộc về | Chấp nhận (acceptance) |
| Vùng âm (Negative region) | NEG | Đối tượng chắc chắn không thuộc về | Từ chối (rejection) |
| Vùng biên (Boundary region) | BND | Quan hệ với cụm còn mơ hồ | Hoãn quyết định (non-commitment) |

> **[Chèn hình ảnh]:** Hình 1 trong bài báo — *"Trisecting-and-acting model"* — minh hoạ mô hình phân ba chiều: tập vũ trụ được chia thành POS (màu xanh), BND (màu vàng) và NEG (màu đỏ).

### 2.2.2. Đặc điểm nổi bật

Phương pháp quyết định ba chiều cung cấp cách tiếp cận linh hoạt trong xử lý thông tin phức tạp và không chắc chắn. Lý thuyết này đã được tích hợp vào nhiều nhánh nghiên cứu như lý thuyết bằng chứng (evidence theory), tập mờ (fuzzy set), và hỗ trợ quyết định (decision support). Trong bài báo này, lý thuyết ba chiều được áp dụng cho bài toán phân cụm để tạo ra biểu diễn cụm phong phú hơn so với phân cụm nhị phân truyền thống.

---

## 2.3. Biểu diễn cụm theo phương pháp ba chiều

### 2.3.1. Định nghĩa cụm ba chiều

Cho $C = \{C_1, C_2, \ldots, C_k\}$ là tập hợp các cụm và $V = \{x_1, x_2, \ldots, x_n\}$ là tập vũ trụ hữu hạn các đối tượng. Mỗi đối tượng $x_i$ có $h$ thuộc tính: $x_i = (x_{i1}, x_{i2}, \ldots, x_{ih})$, trong đó $x_{ij}$ là giá trị thuộc tính thứ $j$ của đối tượng $x_i$.

**Định nghĩa 1 (Cụm ba chiều):** Một cụm được biểu diễn bởi cặp tập hợp lồng nhau [33]:

$$C_i = [\underline{C_i},\ \overline{C_i}] \tag{1}$$

Trong đó:
- $\underline{C_i}$ là **cận dưới** (lower bound) của cụm $C_i$
- $\overline{C_i}$ là **cận trên** (upper bound) của cụm $C_i$
- Quan hệ lồng nhau: $\underline{C_i} \subseteq \overline{C_i} \subseteq V$

### 2.3.2. Ba vùng của một cụm

Từ cặp tập hợp cận dưới và cận trên, ta xác định ba vùng của cụm $C_i$ như sau:

$$\text{POS}(C_i) = \underline{C_i}$$

$$\text{BND}(C_i) = \overline{C_i} - \underline{C_i}$$

$$\text{NEG}(C_i) = V - \overline{C_i}$$

**(Công thức 2)**

**Giải thích từng ký hiệu:**

- $\text{POS}(C_i)$: **Vùng dương** — tập hợp tất cả đối tượng **chắc chắn thuộc** cụm $C_i$. Đây chính là cận dưới $\underline{C_i}$, gồm các điểm lõi (core points) mật độ cao.
- $\text{BND}(C_i)$: **Vùng biên** — phần chênh lệch giữa cận trên và cận dưới ($\overline{C_i} - \underline{C_i}$). Các đối tượng trong vùng này có **quan hệ mơ hồ** với cụm $C_i$: có thể thuộc $C_i$, hoặc đồng thời thuộc về nhiều cụm — cần thêm thông tin để xác định.
- $\text{NEG}(C_i)$: **Vùng âm** — phần bù của cận trên trong tập vũ trụ ($V - \overline{C_i}$). Các đối tượng ở đây **chắc chắn không thuộc** cụm $C_i$.

### 2.3.3. Các điều kiện ràng buộc

Các tập con của cụm phải thỏa mãn đồng thời năm điều kiện sau:

| Điều kiện | Ý nghĩa |
|-----------|---------|
| $\text{POS}(C_i) \neq \emptyset$ | Mỗi cụm phải có ít nhất một phần tử chắc chắn thuộc về |
| $\text{POS}(C_i) \cap \text{BND}(C_i) = \emptyset$ | Vùng dương và vùng biên không giao nhau |
| $\text{BND}(C_i) \cap \text{NEG}(C_i) = \emptyset$ | Vùng biên và vùng âm không giao nhau |
| $\text{POS}(C_i) \cap \text{POS}(C_j) = \emptyset,\ i \neq j$ | Các vùng dương của các cụm khác nhau không giao nhau |
| $\text{POS}(C_i) \cup \text{BND}(C_i) \cup \text{NEG}(C_i) = V$ | Ba vùng bao phủ toàn bộ tập vũ trụ |

Tập hợp tất cả các cụm ba chiều được biểu diễn bằng **tập hợp khoảng** (interval sets):

$$C = \{[\underline{C_1}, \overline{C_1}],\ [\underline{C_2}, \overline{C_2}],\ \ldots,\ [\underline{C_k}, \overline{C_k}]\} \tag{3}$$

### 2.3.4. So sánh phân cụm hai chiều và ba chiều

> **[Chèn hình ảnh]:** Hình 2 trong bài báo — *"Representation of two-way clustering and three-way clustering"* — cho thấy: (a) phân cụm hai chiều buộc phải gán 6 điểm rời rạc $p_1, \ldots, p_6$ vào đúng một cụm; (b) phân cụm ba chiều đưa các điểm này vào vùng biên BND, phản ánh đúng mức độ mơ hồ của chúng.

Xét ví dụ: một tập dữ liệu có hai vùng tập trung và sáu điểm tương đối rải rác $p_1, \ldots, p_6$. Khi áp dụng hard clustering (Hình 2a), mỗi điểm buộc phải được gán dứt khoát vào một cụm, gây ra sai số. Tuy nhiên, điểm $p_3$ và $p_4$ có khoảng cách tới $C_1$ và $C_2$ gần như bằng nhau — việc gán chúng vào bất kỳ cụm nào đều không hợp lý. Với phân cụm ba chiều (Hình 2b), các điểm này được xếp vào **vùng biên BND** — vùng chồng lấp giữa hai cụm — phù hợp hơn với nhận thức tự nhiên của con người.

---

## 2.4. Thuật toán DBSCAN

### 2.4.1. Giới thiệu

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) [40] là một trong những thuật toán phân cụm dựa trên mật độ (density-based) thành công nhất. Ưu điểm nổi bật của DBSCAN:

- Phát hiện cụm có hình dạng tùy ý (arbitrary shape)
- Tự động xác định số lượng cụm (không cần khai báo $k$ trước)
- Xử lý được điểm nhiễu (noise/outlier)

Tuy nhiên, **hạn chế chính của DBSCAN** là không thể tìm được các cụm có mật độ khác nhau khi chỉ sử dụng một ngưỡng mật độ toàn cục (global density threshold) duy nhất.

### 2.4.2. Các khái niệm cơ bản

DBSCAN sử dụng hai tham số:
- $\epsilon$ (epsilon): bán kính tối đa của vùng lân cận
- $\text{MinPts}$: số điểm tối thiểu để tạo thành vùng mật độ cao

**Định nghĩa 2 (Ba loại điểm):** Cho hai điểm bất kỳ $x$ và $y$, $d(x, y)$ là độ tương đồng (khoảng cách) giữa chúng. Vùng $\epsilon$-lân cận của $x$ được định nghĩa:

$$\Gamma_\epsilon(x) = \{y \in V \mid d(x, y) \leq \epsilon\}$$

Mật độ của $x$ được tính bằng:

$$\rho(x) = |\Gamma_\epsilon(x)|$$

Hàm loại điểm $S(x)$ phân loại mỗi điểm thành ba loại:

$$S(x) = \begin{cases}
1 & \text{điểm lõi (core point) khi } \rho(x) \geq \text{MinPts} \\
0 & \text{điểm biên (border point) khi } 1 < \rho(x) < \text{MinPts} \\
-1 & \text{điểm nhiễu (noise) khi } \rho(x) = 1
\end{cases} \tag{4}$$

**Giải thích từng ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $x, y$ | Hai điểm dữ liệu bất kỳ trong tập $V$ |
| $d(x, y)$ | Khoảng cách (độ tương đồng) giữa $x$ và $y$, thường dùng khoảng cách Euclidean |
| $\epsilon$ | Bán kính ngưỡng xác định vùng lân cận của một điểm |
| $\Gamma_\epsilon(x)$ | Tập hợp tất cả điểm $y$ nằm trong vòng bán kính $\epsilon$ của $x$ |
| $\rho(x) = |\Gamma_\epsilon(x)|$ | Số điểm trong vùng $\epsilon$-lân cận của $x$, tức là mật độ tại $x$ |
| $\text{MinPts}$ | Ngưỡng mật độ tối thiểu để một điểm được coi là điểm lõi |
| $S(x) = 1$ | $x$ là **điểm lõi**: vùng lân cận dày đặc, đại diện cho vùng mật độ cao |
| $S(x) = 0$ | $x$ là **điểm biên**: nằm trong vùng lân cận của điểm lõi nhưng bản thân không đủ mật độ |
| $S(x) = -1$ | $x$ là **điểm nhiễu**: không thuộc bất kỳ vùng lân cận nào đủ mật độ |

### 2.4.3. Quá trình hoạt động của DBSCAN

DBSCAN bắt đầu từ một điểm ngẫu nhiên chưa thăm. Nếu điểm đó là điểm lõi, tất cả các điểm trong $\epsilon$-lân cận của nó sẽ được đưa vào cùng cụm. Quá trình mở rộng cụm tiếp tục cho đến khi không còn điểm nào có thể tiếp cận theo mật độ (density-reachable). Các điểm không thuộc bất kỳ cụm nào được đánh dấu là nhiễu.

---

## 2.5. Cải tiến hàm tính độ tương đồng trong DBSCAN

### 2.5.1. Vấn đề của DBSCAN gốc với mật độ không đồng đều

Khi dữ liệu có **mật độ không đồng đều** (varied densities), DBSCAN với một ngưỡng $\epsilon$ duy nhất sẽ thất bại:
- Nếu $\epsilon$ quá nhỏ: các cụm mật độ thấp bị phá vỡ thành nhiều mảnh
- Nếu $\epsilon$ quá lớn: các cụm mật độ cao bị gộp lại sai

Để khắc phục, bài báo áp dụng kỹ thuật **co giãn khoảng cách đa chiều (DScale)** [49] như một bước tiền xử lý dữ liệu, biến đổi khoảng cách ban đầu sao cho dữ liệu trở nên gần đồng nhất về mật độ.

> **[Chèn hình ảnh]:** Hình 3 trong bài báo — *"Original and scaling data distribution"* — minh hoạ: (a) dữ liệu gốc gồm ba cụm Gaussian với mật độ khác nhau ($C_1$, $C_2$ dày đặc, $C_3$ thưa); (b) sau khi co giãn khoảng cách, ba cụm có mật độ tương đương hơn và DBSCAN có thể phát hiện cả ba bằng một ngưỡng duy nhất.

### 2.5.2. Hàm co giãn (Scaling Function)

**Định nghĩa 3 (Hàm co giãn):** Cho tập dữ liệu $V = \{x_1, x_2, \ldots, x_n\}$ gồm $n$ đối tượng, mỗi đối tượng có $h$ đặc trưng ($x_i \in \mathbb{R}^h$). Cho ma trận khoảng cách $D = [d(x, y)]$ và $d_{\max} = \max_{x,y \in V} d(x, y)$. Hàm co giãn được định nghĩa:

$$r(x) = \left(\frac{|\Gamma_\eta(x, d)|}{n}\right)^{\frac{1}{h}} \times \frac{d_{\max}}{\eta} \tag{5}$$

**Giải thích từng ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $r(x)$ | **Hệ số co giãn** của điểm $x$ — quyết định khoảng cách sẽ được phóng to hay thu nhỏ |
| $n$ | Tổng số đối tượng trong tập dữ liệu |
| $h$ | Số chiều (số đặc trưng) của dữ liệu |
| $\eta$ | Tham số bán kính lân cận dùng để tính hàm co giãn, $\eta \in (0, 1)$ |
| $\Gamma_\eta(x, d)$ | Vùng $\eta$-lân cận của $x$ theo khoảng cách gốc $d$: $\Gamma_\eta(x,d) = \{y \in V \mid d(x,y) \leq \eta\}$ |
| $|\Gamma_\eta(x, d)|$ | Số lượng điểm trong vùng $\eta$-lân cận của $x$ — phản ánh mật độ cục bộ tại $x$ |
| $d_{\max}$ | Khoảng cách lớn nhất trong toàn bộ tập dữ liệu: $d_{\max} = \max_{x,y \in V} d(x,y)$ |

**Ý nghĩa vật lý của $r(x)$:** Nếu điểm $x$ nằm trong vùng **mật độ cao** (nhiều điểm lân cận, $|\Gamma_\eta(x,d)|$ lớn), thì $r(x)$ lớn → khoảng cách sẽ được kéo dài (expand) → vùng dày đặc được giãn ra. Ngược lại, nếu $x$ ở vùng **mật độ thấp**, $r(x)$ nhỏ → khoảng cách được thu nhỏ → vùng thưa được co lại. Nhờ đó, sau biến đổi, toàn bộ dữ liệu trở nên xấp xỉ phân phối đều.

Số mũ $\frac{1}{h}$ là **hiệu chỉnh theo số chiều** để đảm bảo tính nhất quán khi dữ liệu có số chiều $h$ khác nhau.

### 2.5.3. Khoảng cách sau co giãn (Scaled Distance)

**Định nghĩa 4 (Khoảng cách co giãn):** Khoảng cách co giãn $d'(x, y)$ được tính từ khoảng cách gốc $d(x,y)$ và hàm co giãn $r(x)$ theo hai trường hợp:

$$d'(x, y) = \begin{cases}
d(x, y) \times r(x) & \text{nếu } y \in \Gamma_\eta(x, d) \\[8pt]
(d(x, y) - \eta) \times \dfrac{d_{\max} - \eta \cdot r(x)}{d_{\max} - \eta} + \eta \cdot r(x) & \text{nếu } y \in V \setminus \Gamma_\eta(x, d)
\end{cases} \tag{6}$$

**Giải thích từng ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $d'(x, y)$ | Khoảng cách **sau khi co giãn** giữa hai điểm $x$ và $y$ |
| $d(x, y)$ | Khoảng cách **gốc** (Euclidean) giữa $x$ và $y$ trước khi biến đổi |
| $r(x)$ | Hệ số co giãn của điểm $x$ (tính từ Công thức 5) |
| $\eta$ | Bán kính ngưỡng xác định vùng lân cận trong hàm co giãn |
| $d_{\max}$ | Khoảng cách lớn nhất trong tập dữ liệu |
| $V \setminus \Gamma_\eta(x, d)$ | Tập hợp các điểm nằm **ngoài** vùng $\eta$-lân cận của $x$ |

**Giải thích hai trường hợp:**

- **Trường hợp 1** ($y \in \Gamma_\eta(x, d)$, tức $y$ nằm **trong** vùng lân cận của $x$): công thức là **co giãn tuyến tính** $d'(x,y) = d(x,y) \times r(x)$. Điều này đảm bảo số lượng điểm trong $\Gamma_\eta(x,d)$ bằng đúng số điểm trong $\Gamma_{\eta \cdot r(x)}(x, d')$ — tức là bảo toàn số lượng điểm lân cận.

- **Trường hợp 2** ($y \in V \setminus \Gamma_\eta(x, d)$, tức $y$ nằm **ngoài** vùng lân cận): công thức là **chuẩn hóa min-max** ánh xạ khoảng cách $[\eta, d_{\max}]$ về khoảng mới $[\eta \cdot r(x), d_{\max}]$. Mục đích là **bảo toàn thứ hạng** (rank) giữa các điểm ngoài lân cận, không làm thay đổi thứ tự sắp xếp khoảng cách.

**Cơ chế tổng thể:** Sau biến đổi, các vùng dày đặc trong dữ liệu gốc được **giãn nở** trong dữ liệu mới, các vùng thưa thớt được **co lại**. Kết quả là khoảng cách giữa các cụm khác nhau tăng lên, ranh giới giữa các cụm rõ ràng hơn, cho phép DBSCAN dùng một ngưỡng $\epsilon$ duy nhất để phân tách các cụm có mật độ khác nhau.

### 2.5.4. Kết quả của DBSCAN cải tiến

Từ công thức (6), ta thu được **ma trận khoảng cách co giãn** $D' = [d'(x,y)]$. DBSCAN sau đó được thực thi trên $D'$ để tìm các cụm. Kết quả của DBSCAN cải tiến:

$$\mathcal{C} = \{C_1, C_2, \ldots, C_k\} \cup \text{No}(\mathcal{C})$$

Trong đó $k$ là số cụm tìm được và $\text{No}(\mathcal{C})$ là tập hợp tất cả điểm nhiễu.

---

## 2.6. Xử lý phân cụm ba chiều (Three-way Clustering Processing)

Sau khi có kết quả phân cụm từ DBSCAN cải tiến, bước tiếp theo là **xây dựng biểu diễn ba chiều** cho mỗi cụm. Quá trình này gồm **ba chiến lược** chính:

### 2.6.1. Chiến lược 1: Xác định vùng dương và vùng biên ban đầu

Ý tưởng: so sánh mật độ của từng đối tượng với ngưỡng $\text{MinPts}$ qua hàm $S(x)$ (Công thức 4):
- Nếu $S(x) = 1$ (điểm lõi, mật độ cao) → đối tượng **chắc chắn** thuộc cụm → đưa vào **vùng dương**
- Nếu $S(x) = 0$ (điểm biên, mật độ thấp) → đối tượng có thể thuộc nhiều cụm → đưa vào **vùng biên**

$$\text{POS}(C_i) = \{x \mid S(x) = 1,\ x \in C_i\}$$

$$\text{BND}(C_i) = \{x \mid S(x) = 0,\ x \in C_i\} \tag{7}$$

**Ý nghĩa:** $\text{POS}(C_i)$ là tập hợp **toàn bộ điểm lõi** của cụm $C_i$; $\text{BND}(C_i)$ là tập hợp **toàn bộ điểm biên** ban đầu của cụm $C_i$.

### 2.6.2. Chiến lược 2: Mở rộng vùng biên để xử lý đối tượng chồng lấp

Sau Chiến lược 1, một điểm biên chỉ được gán cho một cụm, nhưng thực tế nó có thể thuộc về nhiều cụm (overlapping). Chiến lược 2 xử lý trường hợp này:

**Quy tắc:** Với mỗi điểm $x$ có $S(x) = 0$ hoặc $S(x) = -1$ (điểm biên hoặc điểm nhiễu), nếu **một láng giềng** của $x$ trong vùng $\epsilon$-lân cận thuộc về cụm $C_i$ khác, thì $x$ được coi là đối tượng chồng lấp và được thêm vào $\text{BND}(C_i)$.

Vùng biên của $C_i$ được cập nhật:

$$\text{BND}(C_i) = \text{BND}(C_i) \cup \{x \mid C_i \cap \Gamma_\epsilon(x) \neq \emptyset\} \tag{8}$$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\text{BND}(C_i)$ (vế trái) | Vùng biên của $C_i$ **sau khi mở rộng** |
| $\text{BND}(C_i)$ (vế phải) | Vùng biên của $C_i$ **trước khi mở rộng** (từ Chiến lược 1) |
| $\Gamma_\epsilon(x)$ | Vùng $\epsilon$-lân cận của điểm $x$ |
| $C_i \cap \Gamma_\epsilon(x) \neq \emptyset$ | Điều kiện: có ít nhất một điểm lân cận của $x$ thuộc cụm $C_i$ |
| $\cup$ | Phép hợp — thêm các điểm chồng lấp vào vùng biên |

### 2.6.3. Chiến lược 3: Xử lý điểm nhiễu còn lại

Sau Chiến lược 2, vẫn còn một số điểm nhiễu chưa được gán vào bất kỳ cụm nào. Chiến lược 3 xử lý các điểm này bằng cách tìm **điểm lõi gần nhất** (Nearest Core Neighbor — NCN):

Đặt $\text{AllPOS} = \bigcup_{i=1}^{k} \text{POS}(C_i)$ là tập hợp tất cả điểm lõi. Với mỗi điểm nhiễu $x$ ($S(x) = -1$), điểm lõi gần nhất được xác định:

$$\text{NCN}(x) = y \leftarrow \arg\min_{y \in \text{AllPOS}} d(x, y),\quad S(x) = -1 \tag{9}$$

Sau đó, điểm nhiễu $x$ được thêm vào **vùng biên của cụm chứa** $\text{NCN}(x)$:

$$\text{if}\ \text{NCN}(x) \in \text{POS}(C_i),\ \text{then}\ \text{BND}(C_i) = \text{BND}(C_i) \cup \{x\} \tag{10}$$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\text{AllPOS}$ | Hợp tập của **tất cả điểm lõi** từ tất cả các cụm |
| $\text{NCN}(x)$ | **Điểm lõi gần nhất** với điểm nhiễu $x$ |
| $\arg\min_{y \in \text{AllPOS}} d(x, y)$ | Chọn điểm $y$ trong $\text{AllPOS}$ sao cho khoảng cách $d(x,y)$ là nhỏ nhất |
| $\text{NCN}(x) \in \text{POS}(C_i)$ | Điểm lõi gần nhất thuộc cụm $C_i$ → điểm nhiễu $x$ được gán vào vùng biên của $C_i$ |

**Lý do:** Không gán điểm nhiễu vào vùng dương vì điểm nhiễu không đủ điều kiện mật độ. Tuy nhiên, nó vẫn có thể nằm gần một cụm nào đó về không gian, nên hợp lý hơn khi đặt nó vào **vùng biên** — thể hiện quan hệ mơ hồ với cụm gần nhất.

---

## 2.7. Thuật toán 3W-DBSCAN

### 2.7.1. Mã giả của thuật toán

```
Algorithm 1: 3W-DBSCAN

Input : Tập dữ liệu V = {x₁, x₂, ..., xₙ}
        Tham số ε (bán kính lân cận)
        Tham số MinPts (ngưỡng mật độ tối thiểu)
        Tham số η (bán kính lân cận cho hàm co giãn)

Output: C = {[C̲₁, C̄₁], [C̲₂, C̄₂], ..., [C̲ₖ, C̄ₖ]}

─────────────────────────────────────────────────
BƯỚC 1: Tính ma trận khoảng cách gốc
  D ← ComputeDistanceMatrix(V)    // Sử dụng khoảng cách Euclidean
                                    // Độ phức tạp: O(n²)

BƯỚC 2: Tính ma trận khoảng cách co giãn
  FOR mỗi điểm xᵢ ∈ V:
    Tính Γ_η(xᵢ, d) = {y ∈ V | d(xᵢ, y) ≤ η}
    Tính r(xᵢ) = (|Γ_η(xᵢ, d)| / n)^(1/h) × (d_max / η)  // Công thức (5)
  FOR mỗi cặp (xᵢ, xⱼ):
    Tính d'(xᵢ, xⱼ) theo Công thức (6):
      IF xⱼ ∈ Γ_η(xᵢ, d):
        d'(xᵢ, xⱼ) ← d(xᵢ, xⱼ) × r(xᵢ)
      ELSE:
        d'(xᵢ, xⱼ) ← (d(xᵢ,xⱼ) - η) × (d_max - η·r(xᵢ))/(d_max - η) + η·r(xᵢ)
  D' ← [d'(x, y)]                // Độ phức tạp: O(n²)

BƯỚC 3: Chạy DBSCAN trên D' (DBSCAN cải tiến)
  {C₁, C₂, ..., Cₖ} ∪ No(C) ← DBSCAN(D', ε, MinPts)
                                    // Độ phức tạp: O(n²) trong trường hợp xấu nhất

  // Kết quả: k cụm và tập No(C) chứa toàn bộ điểm nhiễu

BƯỚC 4: [Chiến lược 1] Xác định vùng dương và vùng biên ban đầu
  FOR mỗi cụm Cᵢ (i = 1 đến k):
    POS(Cᵢ) ← {x | S(x) = 1, x ∈ Cᵢ}   // Tất cả điểm lõi
    BND(Cᵢ) ← {x | S(x) = 0, x ∈ Cᵢ}   // Tất cả điểm biên ban đầu
                                    // Công thức (7), độ phức tạp: O(n₁)
                                    // n₁: số điểm đã được phân cụm

BƯỚC 5: [Chiến lược 2] Mở rộng vùng biên xử lý điểm chồng lấp
  FOR mỗi điểm x có S(x) = 0 HOẶC S(x) = -1:
    FOR mỗi cụm Cᵢ:
      IF Cᵢ ∩ Γ_ε(x) ≠ ∅ VÀ x ∉ Cᵢ:    // x có láng giềng thuộc Cᵢ
        BND(Cᵢ) ← BND(Cᵢ) ∪ {x}         // Thêm x vào vùng biên của Cᵢ
                                    // Công thức (8), độ phức tạp: O(n)

BƯỚC 6: [Chiến lược 3] Gán điểm nhiễu còn lại vào vùng biên
  AllPOS ← ⋃ᵢ POS(Cᵢ)              // Hợp tất cả điểm lõi
  FOR mỗi điểm nhiễu x ∈ No(C) chưa được gán:
    NCN(x) ← argmin_{y ∈ AllPOS} d(x, y)   // Công thức (9)
    IF NCN(x) ∈ POS(Cᵢ):
      BND(Cᵢ) ← BND(Cᵢ) ∪ {x}            // Công thức (10)
                                    // Độ phức tạp: O(n₁·n₂), n₂: số điểm nhiễu

BƯỚC 7: Xây dựng cận dưới và cận trên
  FOR mỗi cụm Cᵢ (i = 1 đến k):
    C̲ᵢ ← POS(Cᵢ)                  // Cận dưới = vùng dương
    C̄ᵢ ← POS(Cᵢ) ∪ BND(Cᵢ)       // Cận trên = vùng dương ∪ vùng biên

RETURN C = {[C̲₁, C̄₁], [C̲₂, C̄₂], ..., [C̲ₖ, C̄ₖ]}
─────────────────────────────────────────────────
```

### 2.7.2. Lưu đồ thuật toán

> **[Chèn hình ảnh]:** Hình 4 trong bài báo — *"The flowchart of 3W-DBSCAN"* — thể hiện toàn bộ quy trình gồm: (1) tính ma trận khoảng cách gốc → (2) tính ma trận khoảng cách co giãn → (3) chạy DBSCAN cải tiến → (4) phân loại điểm lõi/biên/nhiễu → (5) mở rộng vùng biên → (6) gán điểm nhiễu → (7) xuất kết quả cụm ba chiều.

Dưới đây là mô tả lưu đồ theo dạng văn bản:

```
         ┌─────────────────────────────┐
         │  BẮT ĐẦU                    │
         │  Input: V, ε, MinPts, η     │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │  Tính ma trận khoảng cách D  │
         │  (khoảng cách Euclidean)     │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │  Tính hàm co giãn r(x) (Công thức 5)│
         │  Tính ma trận D' (Công thức 6)       │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │  Chạy DBSCAN trên D'        │
         │  → k cụm + No(C) (nhiễu)   │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────┐
         │  [Chiến lược 1]                          │
         │  Xác định POS(Cᵢ) = {x | S(x)=1, x∈Cᵢ} │
         │  Xác định BND(Cᵢ) = {x | S(x)=0, x∈Cᵢ} │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────┐
         │  [Chiến lược 2]                          │
         │  Duyệt điểm biên & nhiễu                 │
         │  Nếu có láng giềng ∈ Cᵢ khác            │
         │  → Thêm vào BND(Cᵢ) (Công thức 8)       │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────┐
         │  [Chiến lược 3]                          │
         │  Với điểm nhiễu còn lại:                 │
         │  Tìm NCN(x) (Công thức 9)               │
         │  Gán x → BND(Cᵢ) chứa NCN(x) (C.t. 10) │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────────────────┐
         │  Xây dựng kết quả ba chiều              │
         │  C̲ᵢ = POS(Cᵢ)                          │
         │  C̄ᵢ = POS(Cᵢ) ∪ BND(Cᵢ)               │
         └──────────────┬──────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────┐
         │  KẾT THÚC                                │
         │  Output: {[C̲₁,C̄₁], ..., [C̲ₖ,C̄ₖ]}      │
         └──────────────────────────────────────────┘
```

### 2.7.3. Phân tích độ phức tạp thời gian

| Bước | Thao tác | Độ phức tạp |
|------|----------|------------|
| Bước 1 | Tính ma trận khoảng cách $D$ | $O(n^2)$ |
| Bước 2 | Tính ma trận co giãn $D'$ | $O(n^2)$ |
| Bước 3 | Chạy DBSCAN trên $D'$ | $O(n^2)$ (worst case) |
| Bước 4 | Chiến lược 1 — phân loại $n_1$ điểm đã phân cụm | $O(n_1)$ |
| Bước 5 | Chiến lược 2 — xử lý chồng lấp | $O(n)$ |
| Bước 6 | Chiến lược 3 — gán $n_2$ điểm nhiễu | $O(n_1 \cdot n_2)$ |

Với $n_1 + n_2 = n$, ta có $O(n_1 \cdot n_2) < O(n^2)$, do đó **độ phức tạp tổng thể của 3W-DBSCAN là $O(n^2)$**.

---

## 2.8. Bộ dữ liệu thực nghiệm

Bài báo sử dụng **10 bộ dữ liệu** thuộc **3 loại** để đánh giá hiệu quả của 3W-DBSCAN. Chi tiết được trình bày trong Bảng 1.

### Bảng 1: Thông tin 10 bộ dữ liệu thực nghiệm

| Loại | Tên | Số mẫu | Số chiều | Số cụm | Nguồn |
|------|-----|--------|----------|--------|-------|
| **Dữ liệu tổng hợp** | 3L | 560 | 2 | 3 | [49] |
| | 4C | 1250 | 2 | 4 | [49] |
| | S1 | 5000 | 2 | 15 | [50] |
| **Dữ liệu UCI** | IRIS | 150 | 4 | 3 | [51] |
| | Glass | 214 | 9 | 6 | [51] |
| | Seeds | 210 | 7 | 3 | [51] |
| **Dữ liệu hình dạng** | Pathbased | 300 | 2 | 3 | [48] |
| | Aggregation | 788 | 2 | 7 | [52] |
| | Flame | 240 | 2 | 2 | [52] |
| | Compound | 399 | 2 | 6 | [48] |

### 2.8.1. Mô tả chi tiết từng bộ dữ liệu

#### Nhóm 1: Dữ liệu tổng hợp (Synthetic Datasets)

**Bộ dữ liệu 3L:**
- 560 điểm dữ liệu, 2 chiều, 3 cụm
- Đặc điểm: ba cụm kéo dài (elongated clusters) theo các hướng khác nhau
- Thách thức: các cụm không có hình dạng tròn, kiểm tra khả năng phát hiện cụm hình dạng tùy ý

**Bộ dữ liệu 4C:**
- 1250 điểm dữ liệu, 2 chiều, 4 cụm
- Đặc điểm: ba cụm Gaussian (hình tròn/elip) và một cụm kéo dài — kết hợp hai loại hình dạng
- Thách thức: mật độ không đồng đều giữa các cụm, kiểm tra khả năng xử lý hình dạng hỗn hợp

> **[Chèn hình ảnh]:** Hình 5 trong bài báo — *"Four different schematic diagram of 4C"* — so sánh kết quả phân cụm của CE3-kmeans và 3W-DBSCAN (cả cận dưới và cận trên) trên bộ dữ liệu 4C.

**Bộ dữ liệu S1:**
- 5000 điểm dữ liệu, 2 chiều, 15 cụm Gaussian
- Đặc điểm: số lượng cụm lớn (15), các cụm Gaussian có thể có mức độ chồng lấp
- Thách thức: kiểm tra khả năng mở rộng (scalability) và xử lý đồng thời nhiều cụm

#### Nhóm 2: Dữ liệu UCI (UCI Machine Learning Repository [51])

**Bộ dữ liệu IRIS:**
- 150 mẫu, 4 chiều (chiều dài và chiều rộng đài hoa, chiều dài và chiều rộng cánh hoa), 3 loài hoa Iris
- Đặc điểm: một cụm tách biệt hoàn toàn, hai cụm còn lại có sự chồng lấp
- Bài báo phân tích tại chiều thứ 3 và thứ 4 (hình 7)

> **[Chèn hình ảnh]:** Hình 7 trong bài báo — *"Four different schematic diagram of IRIS"* — so sánh phân phối thực tế, kết quả DScale-DBSCAN, CE3-kmeans và 3W-DBSCAN tại chiều 3-4.

**Bộ dữ liệu Glass:**
- 214 mẫu, 9 chiều (các oxyt thành phần như Na₂O, MgO, Al₂O₃, ...), 6 loại thủy tinh
- Đặc điểm: nhiều chiều, các cụm có kích thước không cân bằng
- Thách thức: bộ dữ liệu khó phân cụm do mật độ và phân phối phức tạp trong không gian 9 chiều

**Bộ dữ liệu Seeds:**
- 210 mẫu, 7 chiều (các thông số hình học của hạt giống lúa mì), 3 giống
- Đặc điểm: tương đối cân bằng (70 mẫu mỗi lớp), nhưng các cụm có sự chồng lấp trong không gian đặc trưng

#### Nhóm 3: Dữ liệu hình dạng (Shape Datasets)

**Bộ dữ liệu Pathbased:**
- 300 điểm, 2 chiều, 3 cụm
- Đặc điểm: hai cụm Gaussian và một cụm tròn có các kết nối (connections) giữa chúng
- Thách thức: các cụm được nối với nhau bằng các chuỗi điểm mảnh — khó để phân tách bằng các phương pháp dựa trên khoảng cách đơn thuần

**Bộ dữ liệu Aggregation:**
- 788 điểm, 2 chiều, 7 cụm
- Đặc điểm: bảy cụm tròn có mật độ tương đương, một số cụm nằm gần nhau tạo thành vùng kết nối
- Thách thức: các vùng kết nối giữa $C_4$-$C_5$ và $C_2$-$C_3$ dẫn đến đối tượng chồng lấp trong kết quả 3W-DBSCAN

> **[Chèn hình ảnh]:** Hình 8 trong bài báo — *"Four different schematic diagram of Aggregation"* — cho thấy phân phối gốc và kết quả phân cụm, bao gồm vùng chồng lấp.

> **[Chèn hình ảnh]:** Hình 9 trong bài báo — *"The overlap region of Aggregation"* — phóng to vùng chồng lấp giữa các cụm trong Aggregation.

**Bộ dữ liệu Flame:**
- 240 điểm, 2 chiều, 2 cụm
- Đặc điểm: phân phối kết nối giống ngọn lửa (flame-shaped), không phải hình tròn hay elip
- Thách thức: hình dạng đặc biệt đòi hỏi thuật toán có khả năng phát hiện cụm hình dạng tùy ý

**Bộ dữ liệu Compound:**
- 399 điểm, 2 chiều, 6 cụm
- Đặc điểm: hỗn hợp các hình dạng và mật độ khác nhau (một cụm lớn bao chứa các cụm nhỏ hơn, cụm hình tròn, cụm kéo dài...)
- Thách thức: được coi là bộ dữ liệu khó nhất trong nhóm, kiểm tra toàn diện nhất khả năng xử lý mật độ và hình dạng đa dạng

> **[Chèn hình ảnh]:** Hình 6 trong bài báo — *"Four different schematic diagram of Compound"* — so sánh kết quả phân cụm trên Compound với CE3-kmeans, DScale-DBSCAN và 3W-DBSCAN.

---

## 2.9. Các độ đo đánh giá

Bài báo sử dụng **ba chỉ số** truyền thống để đánh giá hiệu quả phân cụm:

### 2.9.1. Độ chính xác — Accuracy (Acc)

$$\text{Acc} = \frac{\sum_{i=1}^{k} n_i}{n} \tag{11}$$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $k$ | Tổng số cụm |
| $n_i$ | Số đối tượng **được dự đoán đúng** trong cụm $i$ khi so sánh với nhãn thực |
| $n$ | Tổng số đối tượng trong tập dữ liệu |
| $\text{Acc} \in [0, 1]$ | Giá trị càng cao → kết quả phân cụm càng tốt |

Accuracy đo tỷ lệ đối tượng được gán đúng cụm so với nhãn thực tế. Khi đánh giá 3W-DBSCAN, Acc được tính riêng cho **cận dưới** (chỉ dùng $\text{POS}$) và **cận trên** (dùng $\text{POS} \cup \text{BND}$).

### 2.9.2. F-measure (F1)

$$F_1 = \frac{1}{k} \times \sum_{i=1}^{k} \frac{2 \times P_i \times R_i}{P_i + R_i} \tag{12}$$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $k$ | Tổng số cụm thực tế |
| $P_i$ | **Precision** (độ chính xác) của cụm $i$: tỷ lệ đối tượng thuộc đúng cụm $i$ trong số tất cả đối tượng được gán vào cụm $i$ |
| $R_i$ | **Recall** (độ phủ) của cụm $i$: tỷ lệ đối tượng thuộc cụm $i$ thực sự được tìm thấy |
| $\frac{2P_iR_i}{P_i+R_i}$ | **F1 cục bộ** của cụm $i$ — trung bình điều hòa của Precision và Recall |
| $F_1 \in [0, 1]$ | Giá trị càng cao → hiệu năng phân loại càng tốt |

F1 là trung bình không trọng số (unweighted average) của F1 cục bộ qua tất cả các cụm, xét cả hai khía cạnh: độ chính xác và độ phủ.

### 2.9.3. Thông tin tương đồng chuẩn hóa — NMI (Normalized Mutual Information)

$$\text{NMI} = \frac{I(X : Y)}{\max(H(X), H(Y))} \tag{13}$$

**Giải thích ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $X$ | Kết quả phân cụm **dự đoán** |
| $Y$ | Kết quả phân cụm **thực tế** (ground truth) |
| $I(X : Y)$ | **Thông tin tương đồng** (mutual information) giữa $X$ và $Y$ — đo lường lượng thông tin chung |
| $H(X)$ | **Entropy** của biến ngẫu nhiên $X$ liên quan đến phân vùng $X$ |
| $H(Y)$ | **Entropy** của biến ngẫu nhiên $Y$ liên quan đến phân vùng $Y$ |
| $\max(H(X), H(Y))$ | Chuẩn hóa bằng entropy lớn hơn để NMI nằm trong $[0, 1]$ |
| $\text{NMI} \in [0, 1]$ | NMI = 1: hai phân vùng giống hệt nhau; NMI = 0: hoàn toàn độc lập |

NMI thường được dùng trong phát hiện cộng đồng (community detection) và phân cụm chồng lấp (overlapping clustering). Nó đo mức độ tương đồng giữa kết quả phân cụm dự đoán và phân vùng thực tế, không phụ thuộc vào cách đặt nhãn cho các cụm.

---

## 2.10. Kết quả thực nghiệm

### 2.10.1. Các phương pháp so sánh

3W-DBSCAN được so sánh với **hai phương pháp** tiên tiến:

1. **CE3-kmeans** [17]: phương pháp phân cụm ba chiều dựa trên k-means và toán học hình thái học (mathematical morphology). Vì kết quả CE3-kmeans thay đổi giữa các lần chạy (do tính ngẫu nhiên của k-means), hiệu năng được đo bằng trung bình cộng qua 100 lần chạy.

2. **DScale-DBSCAN** [49]: sử dụng DScale để tiền xử lý dữ liệu rồi chạy DBSCAN. Đây là phương pháp phân cụm hai chiều (hard clustering), nên kết quả của nó được dùng như **cận trên** khi so sánh.

### 2.10.2. Chiến lược đánh giá

Đối với 3W-DBSCAN, các chỉ số được tính riêng biệt cho:
- **Cận dưới $\underline{C}$**: chỉ dùng $\text{POS}(C_i)$ — phản ánh mức độ chắc chắn
- **Cận trên $\overline{C}$**: dùng $\text{POS}(C_i) \cup \text{BND}(C_i)$ — phản ánh phạm vi bao phủ đầy đủ

### 2.10.3. Kết quả chi tiết

**Bảng 2: Hiệu năng phân cụm trên 10 bộ dữ liệu**

| Bộ dữ liệu | Chỉ số | CE3-kmeans (Cận dưới) | **3W-DBSCAN (Cận dưới)** | CE3-kmeans (Cận trên) | **3W-DBSCAN (Cận trên)** | DScale-DBSCAN |
|------------|--------|----------------------|--------------------------|----------------------|--------------------------|---------------|
| **3L** | Acc | 0.5014 | **0.6982** | 0.5675 | **0.9411** | 0.8875 |
| | NMI | 0.1275 | **0.4691** | 0.1541 | **0.7163** | 0.5582 |
| | F1 | 0.4939 | **0.7628** | 0.5681 | **0.9189** | 0.8964 |
| **4C** | Acc | 0.5536 | **0.7592** | 0.6267 | **0.9232** | 0.8784 |
| | NMI | 0.2743 | **0.5642** | 0.3097 | **0.7050** | 0.5668 |
| | F1 | 0.6148 | **0.7896** | 0.6574 | **0.9072** | 0.8840 |
| **S1** | Acc | 0.7515 | **0.9450** | 0.9102 | **0.9958** | 0.9762 |
| | NMI | 0.6686 | **0.9019** | 0.8707 | **0.9800** | 0.9130 |
| | F1 | 0.8108 | **0.9711** | 0.8845 | **0.9950** | 0.9863 |
| **IRIS** | Acc | **0.7932** | 0.7200 | 0.8543 | **0.9800** | 0.8867 |
| | NMI | 0.5240 | **0.5578** | 0.6247 | **0.8124** | 0.5928 |
| | F1 | 0.8154 | **0.8273** | 0.8476 | **0.9555** | 0.9289 |
| **Glass** | Acc | 0.4311 | **0.4673** | **0.4954** | 0.4860 | 0.4673 |
| | NMI | 0.1123 | **0.1918** | 0.1035 | **0.1506** | 0.1722 |
| | F1 | 0.3766 | 0.3783 | 0.3786 | **0.3885** | 0.3675 |
| **Seeds** | Acc | **0.8451** | 0.4524 | 0.8928 | **0.9429** | 0.7524 |
| | NMI | **0.5462** | 0.2974 | 0.5818 | **0.7009** | 0.3393 |
| | F1 | **0.8657** | 0.6157 | 0.8809 | **0.9252** | 0.8426 |
| **Pathbased** | Acc | 0.7268 | **0.8767** | 0.7685 | **0.9933** | 0.9767 |
| | NMI | 0.4617 | **0.7548** | 0.4474 | **0.8861** | 0.8830 |
| | F1 | 0.7327 | **0.9324** | 0.7320 | 0.9775 | **0.9816** |
| **Aggregation** | Acc | 0.7856 | **0.9365** | 0.7913 | **1.0000** | **1.0000** |
| | NMI | 0.6677 | **0.9324** | 0.6688 | **0.9724** | 0.9114 |
| | F1 | 0.7389 | **0.9788** | 0.7372 | 0.9916 | **1.0000** |
| **Compound** | Acc | 0.6024 | **0.9098** | 0.6403 | **0.9424** | 0.9373 |
| | NMI | 0.6067 | **0.8300** | 0.5559 | **0.8456** | 0.8395 |
| | F1 | 0.5781 | 0.5835 | **0.8964** | 0.9163 | **0.9174** |
| **Flame** | Acc | 0.8400 | **0.8542** | 0.8484 | **1.0000** | 0.9708 |
| | NMI | 0.4505 | **0.6674** | 0.4505 | **0.9488** | 0.7863 |
| | F1 | 0.8385 | **0.9093** | 0.8432 | **0.9927** | 0.9804 |

*Chú thích: Giá trị in đậm là kết quả tốt nhất trong cận dưới và cận trên tương ứng.*

### 2.10.4. Phân tích và nhận xét kết quả

**Nhận xét tổng quan:**

1. **Cận dưới thấp hơn, cận trên cao hơn:** Cả 3W-DBSCAN và CE3-kmeans đều cho Acc thấp ở cận dưới nhưng cao ở cận trên. Điều này hoàn toàn tự nhiên: cận dưới chỉ gồm các điểm lõi (chắc chắn nhất), còn cận trên bổ sung cả điểm biên (mở rộng bao phủ).

2. **DScale-DBSCAN nằm giữa:** Phần lớn các giá trị Acc và NMI của DScale-DBSCAN nằm giữa giá trị cận dưới và cận trên của 3W-DBSCAN, vì 3W-DBSCAN co lại (tạo cận dưới) rồi mở rộng (tạo cận trên) từ kết quả của DBSCAN.

3. **3W-DBSCAN vượt trội trên dữ liệu tổng hợp và hình dạng:** Trên 3L, 4C, S1, Pathbased, Compound, Flame — 3W-DBSCAN cho kết quả tốt hơn đáng kể so với CE3-kmeans do không bị giới hạn bởi giả định hình dạng cầu của k-means.

4. **Trường hợp ngoại lệ — Seeds:** CE3-kmeans cho Acc và NMI cao hơn 3W-DBSCAN ở cận dưới trên Seeds. Nguyên nhân: phân phối dữ liệu Seeds phù hợp với k-means hơn DBSCAN (các cụm dạng elip khá đều đặn, không có mật độ đặc biệt).

5. **Trường hợp ngoại lệ — IRIS (cận dưới Acc):** CE3-kmeans đạt Acc = 0.7932 > 3W-DBSCAN = 0.7200 ở cận dưới, vì CE3-kmeans gán được hầu hết các điểm của cụm $C_2$ vào vùng dương, trong khi 3W-DBSCAN chỉ gán một phần. Tuy nhiên, ở cận trên, 3W-DBSCAN vượt trội hoàn toàn với Acc = 0.9800 so với 0.8543.

6. **Trường hợp F1 thấp hơn trên Aggregation:** Trên Aggregation, 3W-DBSCAN có F1 = 0.9916 < DScale-DBSCAN = 1.0000. Nguyên nhân: vùng kết nối dày đặc giữa $C_4$-$C_5$ và $C_2$-$C_3$ khiến 3W-DBSCAN đưa các điểm này vào vùng chồng lấp (thuộc đồng thời hai cụm), làm giảm recall. Mặc dù vậy, tác giả cho rằng việc coi các điểm này là "chồng lấp" là hợp lý về mặt nhận thức trực quan (xem Hình 9).

### 2.10.5. Phân tích ảnh hưởng của tham số η

> **[Chèn hình ảnh]:** Hình 10 trong bài báo — *"F1 on four datasets with different η values"* — cho thấy F1 tốt nhất (cận trên) trên 5 bộ dữ liệu khi $\eta$ thay đổi từ 0.1 đến 0.5.

Kết quả cho thấy:
- Hiệu ứng co giãn khác nhau tùy bộ dữ liệu
- **Khuyến nghị:** chọn $\eta \in [0.15, 0.3]$ để có hiệu năng tốt nhất trong hầu hết trường hợp
- $\eta$ là tham số quan trọng cần được nghiên cứu thêm về phương pháp chọn tự động (future work của bài báo)

---

## 2.11. Tóm tắt chương

Chương này đã trình bày toàn bộ cơ sở lý thuyết của thuật toán 3W-DBSCAN, bao gồm:

1. **Lý thuyết quyết định ba chiều:** Nền tảng lý thuyết phân chia tập vũ trụ thành ba vùng POS, BND, NEG, cho phép biểu diễn mối quan hệ mơ hồ giữa đối tượng và cụm.

2. **Biểu diễn cụm ba chiều:** Một cụm được mô tả bằng cặp tập lồng nhau $[\underline{C_i}, \overline{C_i}]$, với cận dưới là tập hợp điểm chắc chắn thuộc cụm và cận trên bao gồm cả vùng biên.

3. **DBSCAN và cải tiến:** DBSCAN cải tiến sử dụng hàm co giãn khoảng cách (Công thức 5-6) để giải quyết vấn đề mật độ không đồng đều, cho phép dùng một ngưỡng $\epsilon$ duy nhất cho toàn bộ dữ liệu.

4. **Ba chiến lược xử lý phân cụm ba chiều:** (1) Xác định POS và BND ban đầu dựa trên loại điểm; (2) Mở rộng BND để xử lý điểm chồng lấp; (3) Gán điểm nhiễu vào BND của cụm gần nhất.

5. **Thực nghiệm:** Trên 10 bộ dữ liệu đa dạng với 3 chỉ số Acc, F1, NMI, 3W-DBSCAN cho kết quả tốt hơn CE3-kmeans và DScale-DBSCAN trong hầu hết trường hợp, đặc biệt trên các bộ dữ liệu có hình dạng phức tạp và mật độ không đồng đều.
