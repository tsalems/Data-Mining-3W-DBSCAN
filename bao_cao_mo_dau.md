# CHƯƠNG 1: MỞ ĐẦU

---

## 1.1. Lý do chọn đề tài

### 1.1.1. Bài toán gom cụm dữ liệu và những hạn chế hiện tại

Trong kỷ nguyên dữ liệu lớn, phân cụm (clustering) là một trong những kỹ thuật học máy không giám sát được ứng dụng rộng rãi nhất, từ phân tích mạng xã hội, xử lý ảnh y tế, phát hiện gian lận tài chính cho đến tối ưu hóa chuỗi cung ứng và logistics. Mục tiêu cốt lõi của phân cụm là tự động nhóm các đối tượng tương đồng về cùng một cụm mà không cần nhãn dữ liệu định sẵn, từ đó khai phá các mẫu ẩn và cấu trúc tự nhiên trong dữ liệu.

Trong số các thuật toán phân cụm, **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) nổi bật nhờ khả năng phát hiện cụm có hình dạng tùy ý, tự động xác định số lượng cụm và xử lý điểm nhiễu — những ưu điểm mà các phương pháp phân vùng truyền thống như K-Means không có được. Tuy nhiên, các phương pháp phân cụm hiện hành, kể cả DBSCAN, đều thuộc nhóm **phân cụm cứng (hard clustering)**: mỗi điểm dữ liệu bắt buộc phải được gán vào đúng một cụm duy nhất, hoặc bị loại bỏ hoàn toàn như điểm nhiễu.

Giả định phân cụm cứng này bộc lộ rõ hạn chế trong nhiều tình huống thực tế. Khi một đơn hàng nằm ở vị trí giao thoa giữa hai khu vực giao hàng, hay một bệnh nhân có triệu chứng đồng thời của nhiều nhóm bệnh, hay một thành viên mạng xã hội tham gia nhiều cộng đồng có chủ đề khác nhau — việc cưỡng ép gán nhãn duy nhất sẽ làm mất đi thông tin quan trọng về sự mơ hồ, dẫn đến quyết định sai lầm hoặc tăng rủi ro vận hành.

### 1.1.2. Thách thức phân vùng ranh giới mờ và dữ liệu đa mật độ trong thực tế

Trong thực tế, dữ liệu không gian địa lý — đặc biệt là dữ liệu đơn hàng giao hàng tại các thành phố lớn — thường tồn tại đồng thời hai thách thức lớn mà các thuật toán phân cụm truyền thống chưa giải quyết thỏa đáng:

**Thứ nhất, vấn đề ranh giới mờ (ambiguous boundaries):** Các khu vực đô thị không có ranh giới hành chính rõ ràng trên bản đồ đơn hàng. Tại vùng giao thoa giữa hai quận, một điểm giao hàng có thể hợp lý khi được phục vụ bởi shipper từ bất kỳ khu vực nào. Phân cụm cứng buộc phải đưa ra quyết định nhị phân trong khi thực tế đòi hỏi sự linh hoạt. Hệ quả là tỷ lệ giao hàng thất bại tăng cao tại các vùng biên, chi phí vận hành đội xe tăng do thiếu linh hoạt trong điều phối.

**Thứ hai, vấn đề mật độ không đồng đều (varied densities):** Trong một thành phố như Hà Nội, mật độ đơn hàng tại khu vực trung tâm (Hoàn Kiếm) dày đặc gấp nhiều lần so với vùng ngoại ô (Đông Anh, Hà Đông). DBSCAN sử dụng một bán kính $\epsilon$ toàn cục duy nhất: nếu chọn $\epsilon$ nhỏ phù hợp với vùng trung tâm thì các đơn ngoại ô bị phân mảnh hoặc bị loại thành nhiễu; nếu chọn $\epsilon$ lớn phù hợp với vùng ngoại ô thì trung tâm bị gộp thành một siêu cụm khổng lồ không thể phân công shipper. Đây là bài toán nan giải trực tiếp ảnh hưởng đến hiệu quả vận hành logistics mỗi ngày.

Hai thách thức trên thúc đẩy sự ra đời của các phương pháp phân cụm tiên tiến hơn: **phân cụm ba chiều (three-way clustering)** kết hợp với kỹ thuật bán kính cục bộ. Phân cụm ba chiều, được truyền cảm hứng từ lý thuyết quyết định ba chiều (three-way decision theory), không ép buộc mọi điểm phải thuộc về hay không thuộc về một cụm — thay vào đó, nó cho phép tồn tại một **vùng biên (boundary region)** thể hiện sự không chắc chắn, phù hợp hơn với bản chất mờ của dữ liệu thực tế.

Chính từ những thách thức trên, nhóm nghiên cứu quyết định chọn đề tài: **"Nghiên cứu thuật toán phân cụm ba chiều dựa trên DBSCAN và đề xuất cải tiến ứng dụng trong bài toán logistics"**, với kỳ vọng không chỉ hiểu sâu nền tảng lý thuyết mà còn hiện thực hóa thành một hệ thống ứng dụng có giá trị thực tiễn.

---

## 1.2. Mục tiêu nghiên cứu

Đề tài được thực hiện với ba mục tiêu cụ thể, có tính kế thừa và bổ sung lẫn nhau:

### 1.2.1. Nghiên cứu thuật toán 3W-DBSCAN

Mục tiêu đầu tiên là nghiên cứu toàn diện thuật toán **3W-DBSCAN** (Three-Way DBSCAN) được đề xuất bởi Yu et al. (2019) — một trong những công trình tiên phong kết hợp lý thuyết quyết định ba chiều với thuật toán DBSCAN mật độ. Cụ thể, nhóm tập trung vào:

- Nắm vững lý thuyết quyết định ba chiều: cơ sở lý luận của việc phân chia tập vũ trụ thành ba vùng POS, BND, NEG và ý nghĩa của từng vùng trong bối cảnh phân cụm.
- Hiểu cơ chế hàm co giãn khoảng cách (DScale) — giải pháp của 3W-DBSCAN để xử lý mật độ không đồng đều — bao gồm hàm co giãn $r(x)$ và khoảng cách co giãn $d'(x, y)$.
- Phân tích ba chiến lược xây dựng vùng biên: xác định POS/BND ban đầu, mở rộng vùng biên cho điểm chồng lấp, và gán điểm nhiễu vào vùng biên dựa trên điểm lõi gần nhất.
- Đánh giá ưu điểm cũng như các hạn chế còn tồn tại của 3W-DBSCAN trên bộ dữ liệu thực nghiệm đa dạng.

### 1.2.2. Đề xuất thuật toán cải tiến LE3W-DBSCAN

Trên cơ sở phân tích hạn chế của 3W-DBSCAN — đặc biệt là tham số $\eta$ nhạy cảm và vùng biên không chính xác do toàn bộ điểm biên bị đưa vào BND — mục tiêu thứ hai là nghiên cứu và trình bày thuật toán cải tiến **LE3W-DBSCAN** (Three-Way DBSCAN Based on Local Eps) do Shen et al. (2023) đề xuất. Điểm đột phá của LE3W-DBSCAN nằm ở hai cơ chế mới:

- **Bán kính cục bộ (Local Eps):** mỗi cụm có bán kính $\text{Eps}_j$ riêng được tính tự động từ khoảng cách $k$-lân cận của điểm khởi đầu cụm, thay thế hoàn toàn hàm co giãn DScale phức tạp.
- **Nguyên tắc mật độ giảm dần và tái phân loại dựa trên nhãn láng giềng:** cụm được hình thành theo thứ tự từ mật độ cao xuống thấp; điểm biên được nâng lên vùng dương nếu toàn bộ láng giềng đồng nhất cụm, chỉ giữ trong vùng biên khi thực sự có sự chồng lấp.

Đây là cải tiến có hệ thống, giải quyết trực tiếp cả hai hạn chế cốt lõi của 3W-DBSCAN, và được kiểm chứng trên 11 bộ dữ liệu chuẩn với kết quả vượt trội ở cả chỉ số phân cụm cứng lẫn chỉ số phân cụm mềm.

### 1.2.3. Xây dựng ứng dụng minh họa thực tế

Mục tiêu thứ ba — và cũng là phần có giá trị ứng dụng trực tiếp nhất — là hiện thực hóa kiến thức lý thuyết thành một **hệ thống dashboard điều phối logistics thông minh** dựa trên nền tảng web. Ứng dụng mang tên **SMART-LOGISTICS: Live Dispatch Dashboard** được xây dựng trên nền Streamlit, với bộ dữ liệu mô phỏng 570 đơn hàng giao hàng tại bốn khu vực của Hà Nội (Hoàn Kiếm, Cầu Giấy, Hà Đông, Đông Anh), đặc trưng bởi mật độ không đồng đều phản ánh thực tế đô thị.

Ứng dụng cho phép người dùng so sánh trực tiếp ba phương pháp — DBSCAN cổ điển, 3W-DBSCAN, và LE3W-DBSCAN — qua bốn chức năng chính:

- **Khám phá dữ liệu (EDA):** trực quan hóa phân bố địa lý đơn hàng, thống kê mô tả theo khu vực bằng biểu đồ scatter, pie chart và histogram.
- **Phân cụm và hiển thị kết quả trên bản đồ:** ba màu phân biệt rõ vùng Lõi (POS — đỏ), vùng Lân cận (BND — cam), và vùng Nhiễu (NEG — xám) trên bản đồ Folium tương tác.
- **Điều phối shipper tự động:** sử dụng K-Means trên tâm cụm để gán đơn hàng cho shipper theo vùng địa lý, kết hợp thuật toán heuristic Nearest-Neighbor để tối ưu tuyến đường.
- **Đánh giá hiệu năng thời gian thực:** hiển thị các chỉ số $\alpha$, $\alpha^*$, $\gamma$ (soft clustering metrics) và tỷ lệ cân bằng tải giữa các shipper.

---

## 1.3. Đối tượng và phạm vi nghiên cứu

### 1.3.1. Đối tượng nghiên cứu

Đề tài tập trung nghiên cứu các đối tượng sau:

- **Lý thuyết quyết định ba chiều (Three-way Decision Theory):** nền tảng lý luận cho phân cụm ba chiều, bao gồm mô hình phân vùng POS-BND-NEG và biểu diễn cụm bằng cặp tập hợp lồng nhau $[\underline{C_i}, \overline{C_i}]$.
- **Thuật toán 3W-DBSCAN** (Yu et al., Physica A, 2019): thuật toán phân cụm ba chiều kết hợp DScale và ba chiến lược xây dựng vùng biên.
- **Thuật toán LE3W-DBSCAN** (Shen et al., Computer Science, 2023): cải tiến của 3W-DBSCAN với bán kính cục bộ và nguyên tắc mật độ giảm dần.
- **Bài toán điều phối logistics dựa trên phân cụm địa lý:** ứng dụng kết quả phân cụm ba chiều vào việc phân vùng giao hàng và tối ưu hóa phân công shipper.

### 1.3.2. Phạm vi nghiên cứu

**Về lý thuyết:** Đề tài giới hạn trong phạm vi các thuật toán phân cụm dựa trên mật độ (density-based clustering), cụ thể là dòng DBSCAN và các biến thể phân cụm ba chiều. Các phương pháp phân cụm phân cấp (hierarchical), mô hình hỗn hợp Gaussian (GMM) hay mạng nơ-ron không thuộc phạm vi nghiên cứu chính.

**Về dữ liệu thực nghiệm:**
- *Dữ liệu chuẩn quốc tế:* 10 bộ dữ liệu từ bài báo Yu et al. (2019) gồm 3 bộ tổng hợp, 3 bộ UCI, 4 bộ hình dạng; và 11 bộ dữ liệu từ bài báo Shen et al. (2023) gồm 5 bộ nhân tạo và 6 bộ UCI thực tế.
- *Dữ liệu ứng dụng:* 570 điểm tọa độ địa lý (kinh độ, vĩ độ) mô phỏng đơn hàng giao hàng tại bốn khu vực Hà Nội với mật độ phân bố khác nhau theo mô hình Gaussian (Hoàn Kiếm: 200 đơn; Cầu Giấy: 120 đơn; Hà Đông: 120 đơn; Đông Anh: 80 đơn; nhiễu phân bố đều: 50 điểm).

**Về ứng dụng:** Hệ thống dashboard được xây dựng trên môi trường Python với Streamlit, phục vụ mục đích minh họa và so sánh thuật toán. Đề tài không đặt mục tiêu triển khai hệ thống production-level có kết nối dữ liệu thực thời gian thực từ các nền tảng thương mại điện tử.

---

## 1.4. Bố cục của quyển báo cáo

Báo cáo được tổ chức thành bốn chương chính với logic kế thừa từ nền tảng lý thuyết đến ứng dụng thực tế:

**Chương 1 — Mở đầu**
Trình bày lý do chọn đề tài xuất phát từ những hạn chế của phân cụm cứng và bài toán dữ liệu đa mật độ trong thực tế. Xác định ba mục tiêu nghiên cứu cụ thể, đối tượng và phạm vi nghiên cứu, cùng bố cục tổng thể của báo cáo.

**Chương 2 — Cơ sở lý thuyết**
Trình bày toàn bộ nền tảng lý thuyết của thuật toán 3W-DBSCAN (Yu et al., 2019). Chương này đi từ lý thuyết quyết định ba chiều, định nghĩa biểu diễn cụm ba chiều bằng cặp cận dưới — cận trên, đến thuật toán DBSCAN gốc và cơ chế cải tiến bằng hàm co giãn khoảng cách DScale. Tiếp theo trình bày ba chiến lược xây dựng vùng biên, mã giả và lưu đồ thuật toán 3W-DBSCAN đầy đủ. Chương kết thúc bằng mô tả chi tiết 10 bộ dữ liệu thực nghiệm, ba độ đo đánh giá (Acc, F1, NMI) và phân tích kết quả so sánh với CE3-kmeans và DScale-DBSCAN.

**Chương 3 — Đề xuất cải tiến**
Phân tích hai hạn chế cốt lõi của 3W-DBSCAN, từ đó trình bày thuật toán cải tiến LE3W-DBSCAN (Shen et al., 2023). Chương này đi sâu vào bốn khái niệm mới: hàm mật độ, khoảng cách $k$-lân cận, bán kính cục bộ và nguyên tắc mật độ giảm dần; trình bày chi tiết LE-DBSCAN (giai đoạn phân cụm hai chiều) và cơ chế tái phân loại ba chiều dựa trên nhãn láng giềng. Mã giả, lưu đồ và so sánh độ phức tạp với 3W-DBSCAN được trình bày đầy đủ. Chương kết thúc bằng kết quả thực nghiệm trên 11 bộ dữ liệu với năm độ đo (ACC, ARI, NMI, $\alpha$, $\alpha^*$, $\gamma$) và phân tích ưu nhược điểm.

**Chương 4 — Xây dựng ứng dụng**
Trình bày kiến trúc và chi tiết triển khai hệ thống **SMART-LOGISTICS: Live Dispatch Dashboard**. Chương mô tả bài toán thực tế (phân vùng giao hàng và điều phối shipper tại Hà Nội), kiến trúc hệ thống gồm tầng dữ liệu, tầng thuật toán (ba module: `ThreeWayDBSCAN`, `LE3W_DBSCAN`, `metrics`) và tầng giao diện Streamlit. Bốn chức năng chính của dashboard được minh họa qua ảnh chụp màn hình: khám phá dữ liệu, phân cụm trên bản đồ Folium, phân công shipper tự động và đánh giá hiệu năng. Chương kết luận bằng so sánh trực quan kết quả ba thuật toán trên cùng bộ dữ liệu logistics Hà Nội.
