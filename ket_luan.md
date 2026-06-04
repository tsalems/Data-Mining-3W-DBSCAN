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
