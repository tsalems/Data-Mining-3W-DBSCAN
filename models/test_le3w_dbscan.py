import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LE3W_DBSCAN:
    def __init__(self, min_pts, k):
        self.min_pts = min_pts
        self.k = k
        self.pos_regions = {}
        self.bnd_regions = {}
        self.labels_two_way = None
        self.local_eps_dict = {}

    def fit(self, X):
        n = X.shape[0]
        # Bước 1: Tính ma trận khoảng cách
        dist_matrix = cdist(X, X)

        # Bước 2: Tính mật độ cho tất cả các đối tượng
        # Mật độ = Trung bình khoảng cách tới MinPts láng giềng gần nhất. 
        # Khoảng cách trung bình càng nhỏ -> Mật độ càng cao.
        sorted_dist = np.sort(dist_matrix, axis=1)
        # Lấy từ 0 đến min_pts (bao gồm cả chính nó để đúng khái niệm k-láng giềng)
        mean_dist = np.mean(sorted_dist[:, 0:self.min_pts], axis=1)
        # Đảo dấu để hàm argmax tìm điểm có khoảng cách nhỏ nhất (mật độ lớn nhất)
        density = -mean_dist 

        # Khởi tạo mảng lưu trữ
        labels = np.full(n, -1) # -1: Chưa phân loại, 0: Nhiễu (Noise), >0: Cluster ID
        point_type = np.full(n, 'U') # U: Unclassified, C: Core, B: Border, N: Noise
        
        cluster_id = 0

        # === BƯỚC 1 ĐẾN 6: THUẬT TOÁN LE-DBSCAN (PHÂN CỤM 2 NHÁNH) ===
        while np.any(labels == -1):
            unclassified_idx = np.where(labels == -1)[0]
            
            # Bước 3 (Mật độ giảm dần): Chọn điểm có mật độ cao nhất làm điểm bắt đầu
            p = unclassified_idx[np.argmax(density[unclassified_idx])]
            
            cluster_id += 1
            
            # Bước 4: Tính Local Eps dựa trên khoảng cách k-láng giềng
            eps_j = sorted_dist[p, self.k]
            self.local_eps_dict[cluster_id] = eps_j
            
            # Bước 5: Tìm điểm Lõi, Biên, Nhiễu và Bành trướng cụm (Expand Cluster)
            neighbors_p = np.where(dist_matrix[p] <= eps_j)[0]
            
            if len(neighbors_p) >= self.min_pts:
                point_type[p] = 'C'
                labels[p] = cluster_id
                seed_set = list(neighbors_p)
                seed_set.remove(p)
                
                while len(seed_set) > 0:
                    q = seed_set.pop(0)
                    
                    if labels[q] == 0: # Điểm nhiễu cũ bị thu nạp thành điểm biên
                        labels[q] = cluster_id
                        if point_type[q] != 'C': 
                            point_type[q] = 'B'
                            
                    if labels[q] == -1: # Điểm chưa phân loại
                        labels[q] = cluster_id
                        neighbors_q = np.where(dist_matrix[q] <= eps_j)[0]
                        if len(neighbors_q) >= self.min_pts:
                            point_type[q] = 'C' # Nó là điểm lõi
                            for n_q in neighbors_q:
                                if labels[n_q] == -1 or labels[n_q] == 0:
                                    if n_q not in seed_set:
                                        seed_set.append(n_q)
                        else:
                            point_type[q] = 'B' # Nó là điểm biên
            else:
                # Nếu không đủ điều kiện làm cụm, coi như nhiễu
                labels[p] = 0
                point_type[p] = 'N'
                cluster_id -= 1 # Trả lại ID cho lượt tìm cụm tiếp theo

        self.labels_two_way = labels.copy()

        # === BƯỚC 7: XEM XÉT NHÃN LÁNG GIỀNG ĐỂ TẠO 3 NHÁNH (LE3W-DBSCAN) ===
        self.pos_regions = {c: [] for c in range(1, cluster_id + 1)}
        self.bnd_regions = {c: [] for c in range(1, cluster_id + 1)}
        
        core_indices = np.where(point_type == 'C')[0]

        for i in range(n):
            c_id = labels[i]
            
            # Quy tắc 1: Điểm Lõi -> Đưa vào Vùng Chính (Positive)
            if point_type[i] == 'C':
                self.pos_regions[c_id].append(i)
                
            # Quy tắc 2: Điểm Biên -> Xét láng giềng
            elif point_type[i] == 'B':
                eps_j = self.local_eps_dict[c_id]
                neighbors_i = np.where(dist_matrix[i] <= eps_j)[0]
                neighbor_labels = labels[neighbors_i]
                
                # Tìm xem có nhãn nào khác với cụm hiện tại không (bỏ qua nhiễu 0)
                different_clusters = set(neighbor_labels) - {c_id, 0, -1}
                
                if len(different_clusters) == 0:
                    # Nếu xung quanh toàn "người nhà" -> Đưa vào Vùng Chính
                    self.pos_regions[c_id].append(i)
                else:
                    # Nếu có "người lạ" -> Đưa vào Vùng Biên của cả 2 cụm giao tranh
                    self.bnd_regions[c_id].append(i)
                    for m in different_clusters:
                        self.bnd_regions[m].append(i)
                        
            # Quy tắc 3: Điểm Nhiễu -> Níu kéo vào Vùng Biên của điểm Lõi gần nhất
            elif point_type[i] == 'N':
                if len(core_indices) > 0:
                    distances_to_cores = dist_matrix[i, core_indices]
                    nearest_core_idx = core_indices[np.argmin(distances_to_cores)]
                    nearest_core_cluster = labels[nearest_core_idx]
                    self.bnd_regions[nearest_core_cluster].append(i)

        return self.pos_regions, self.bnd_regions

# ==========================================
# TEST VÀ TRỰC QUAN HÓA GIỐNG BÀI BÁO
# ==========================================
if __name__ == "__main__":
    # Tạo dữ liệu giả lập đa mật độ (2 cụm đặc, 1 cụm thưa)
    X_dense1, _ = make_blobs(n_samples=200, centers=[[2, 2]], cluster_std=0.4, random_state=42)
    X_dense2, _ = make_blobs(n_samples=200, centers=[[4, 4]], cluster_std=0.3, random_state=42)
    X_sparse, _ = make_blobs(n_samples=100, centers=[[3, 3]], cluster_std=1.5, random_state=42)
    X = np.vstack((X_dense1, X_dense2, X_sparse))

    # Chạy thuật toán
    min_pts = 2
    k = 30
    model = LE3W_DBSCAN(min_pts=min_pts, k=k)
    pos_regions, bnd_regions = model.fit(X)

    # Trực quan hóa
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for c_id in pos_regions.keys():
        color = colors[(c_id - 1) % len(colors)]
        
        # Vẽ Vùng Chính (Positive) - Dùng chấm đặc giống bài báo
        pos_points = pos_regions[c_id]
        if pos_points:
            plt.scatter(X[pos_points, 0], X[pos_points, 1], c=color, label=f'POS($C_{c_id}$)', marker='o', s=30)
            
        # Vẽ Vùng Biên (Boundary) - Dùng chấm rỗng giống bài báo
        bnd_points = bnd_regions[c_id]
        if bnd_points:
            plt.scatter(X[bnd_points, 0], X[bnd_points, 1], edgecolors=color, facecolors='none', label=f'BND($C_{c_id}$)', marker='o', s=50, linewidths=1.5)

    plt.title(f"LE3W-DBSCAN Clustering Result (MinPts={min_pts}, k={k})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()