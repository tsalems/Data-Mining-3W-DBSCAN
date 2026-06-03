import numpy as np
from sklearn.metrics import pairwise_distances

def apply_dscale(X, eta):
    """
    Hàm tính toán ma trận khoảng cách D' theo DScale (Phương trình 5, 6).
    Mục đích: Nội suy khoảng cách để làm lộ rõ các cụm có mật độ khác nhau, 
    kéo giãn vùng dày đặc và thu hẹp vùng thưa thớt.
    """
    # n: số lượng mẫu (points), h: số chiều dữ liệu (features)
    n, h = X.shape 
    
    # Tính ma trận khoảng cách Euclidean nguyên bản D
    D = pairwise_distances(X, metric='euclidean')
    # Tìm khoảng cách lớn nhất trong toàn bộ ma trận (d_max)
    d_max = np.max(D) 
    
    r = np.zeros(n)
    # Tính hàm tỷ lệ r(x) cho từng điểm dữ liệu - Eq. (5)
    for i in range(n):
        # Đếm số lượng điểm nằm trong bán kính lân cận eta
        gamma_eta = np.sum(D[i] <= eta)
        # Tính tỷ lệ r(x) dựa trên số điểm lân cận và số chiều dữ liệu
        r[i] = ((gamma_eta / n) ** (1 / h)) * (d_max / eta)
        
    D_prime = np.zeros_like(D)
    # Tính ma trận khoảng cách mới D' sau khi scale - Eq. (6)
    for i in range(n):
        for j in range(n):
            if D[i, j] <= eta:
                # Nếu điểm y nằm trong lân cận eta của x -> Scale tuyến tính
                D_prime[i, j] = D[i, j] * r[i]
            else:
                # Nếu điểm y nằm ngoài lân cận -> Dùng Min-max normalization để giữ thứ hạng
                numerator = d_max - eta * r[i]
                denominator = d_max - eta
                # Tránh lỗi chia cho 0 trong lập trình
                if denominator == 0: 
                    denominator = 1e-10 
                D_prime[i, j] = (D[i, j] - eta) * (numerator / denominator) + eta * r[i]
                
    # Ép ma trận D' thành ma trận đối xứng hoàn hảo để đưa vào thư viện DBSCAN
    D_prime = (D_prime + D_prime.T) / 2
    return D, D_prime