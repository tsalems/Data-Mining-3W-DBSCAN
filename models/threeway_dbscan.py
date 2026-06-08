import numpy as np
from sklearn.cluster import DBSCAN
from models.dscale import apply_dscale

class DScaleDBSCAN:
    """
    Thuật toán DScale-DBSCAN được bài báo sử dụng làm đối chứng.
    Đây là thuật toán phân cụm 2 chiều (two-way clustering) thông thường, 
    chỉ sử dụng ma trận D' rồi chạy DBSCAN.
    """
    def __init__(self, eps, min_samples, eta):
        self.eps = eps
        self.min_samples = min_samples
        self.eta = eta
        self.labels_ = None
        
    def fit(self, X):
        _, D_prime = apply_dscale(X, self.eta)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        self.labels_ = db.fit_predict(D_prime)
        return self

class ThreeWayDBSCAN:
    """
    Thuật toán 3W-DBSCAN theo Algorithm 1.
    Biểu diễn cụm bằng 2 tập hợp: 
    - Lower bound: Vùng chắc chắn (Positive Region - POS)
    - Upper bound: Bao gồm cả vùng chắc chắn và vùng biên (POS U BND)
    """
    def __init__(self, eps, min_samples, eta):
        self.eps = eps                 # Bán kính lân cận cho DBSCAN
        self.min_samples = min_samples # Ngưỡng số lượng điểm tối thiểu (MinPts)
        self.eta = eta                 # Tham số bán kính cho DScale
        self.POS = {}                  # Lưu index các điểm vùng POS
        self.BND = {}                  # Lưu index các điểm vùng biên BND
        self.labels_ = None
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # Line 1 & 2: Tính ma trận khoảng cách gốc D và ma trận đã scale D'
        D, D_prime = apply_dscale(X, self.eta)
        
        # Line 3: Chạy DBSCAN truyền thống trên ma trận D'
        # Tham số metric='precomputed' bắt buộc để sklearn nhận ma trận tự tính
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        self.labels_ = db.fit_predict(D_prime)
        
        # Lấy mảng boolean đánh dấu các điểm lõi (core points)
        core_samples_mask = np.zeros_like(self.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        
        # Loại bỏ nhãn -1 vì đó là điểm nhiễu chung
        unique_clusters = set(self.labels_)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            
        for k in unique_clusters:
            self.POS[k] = set()
            self.BND[k] = set()
            
        noises = []
        
        # Line 4: Strategy 1 - Xác định POS và BND ban đầu (Eq. 7)
        for i in range(n_samples):
            cluster_id = self.labels_[i]
            if cluster_id != -1:
                if core_samples_mask[i]:
                    self.POS[cluster_id].add(i)  # Core point -> POS
                else:
                    self.BND[cluster_id].add(i)  # Border point -> BND
            else:
                noises.append(i)  # Noise -> Xử lý sau
                
        # Line 5: Strategy 2 - Mở rộng BND cho điểm biên VÀ điểm nhiễu chồng lấp (Eq. 8)
        # Paper: "FOR mỗi điểm x có S(x)=0 HOẶC S(x)=-1" — cả border lẫn noise
        assigned_noises = set()
        for k in unique_clusters:
            bnd_items = list(self.BND[k])
            for idx in bnd_items:
                # Tìm lân cận eps dựa trên khoảng cách GỐC (D)
                neighbors = np.where(D[idx] <= self.eps)[0]
                for n_idx in neighbors:
                    n_cluster = self.labels_[n_idx]
                    if n_cluster != -1 and n_cluster != k:
                        self.BND[n_cluster].add(idx)

        # Xử lý noise points trong Strategy 2: nếu có eps-neighbor thuộc cụm nào thì add vào BND đó
        for noise_idx in noises:
            neighbors = np.where(D[noise_idx] <= self.eps)[0]
            for n_idx in neighbors:
                n_cluster = self.labels_[n_idx]
                if n_cluster != -1:
                    self.BND[n_cluster].add(noise_idx)
                    assigned_noises.add(noise_idx)

        # Line 6: Strategy 3 - Gán điểm nhiễu CÒN LẠI (chưa được gán ở Strategy 2) vào BND (Eq. 9, 10)
        remaining_noises = [n for n in noises if n not in assigned_noises]
        all_pos_indices = []
        pos_cluster_map = {}
        for k in unique_clusters:
            for idx in self.POS[k]:
                all_pos_indices.append(idx)
                pos_cluster_map[idx] = k

        if len(all_pos_indices) > 0:
            for noise_idx in remaining_noises:
                # Eq. 9: Tìm điểm lõi gần nhất (NCN)
                distances_to_cores = D[noise_idx, all_pos_indices]
                nearest_core_relative_idx = np.argmin(distances_to_cores)
                nearest_core_real_idx = all_pos_indices[nearest_core_relative_idx]

                # Eq. 10: Gán noise vào BND của cụm tương ứng
                target_cluster = pos_cluster_map[nearest_core_real_idx]
                self.BND[target_cluster].add(noise_idx)
                
        return self