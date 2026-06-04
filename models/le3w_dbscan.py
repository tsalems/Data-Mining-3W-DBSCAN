import numpy as np

class LE3W_DBSCAN:
    def __init__(self, min_samples, k_neighbors):
        self.min_samples = min_samples
        self.k_neighbors = k_neighbors
        self.POS = {}
        self.BND = {}
        self.local_eps_dict = {}
        
    def fit(self, X):
        n = X.shape[0]
        # Tính ma trận khoảng cách
        D = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        
        # 1. Tính mật độ (Trung bình khoảng cách tới MinPts láng giềng gần nhất - Khoảng cách CÀNG NHỎ thì mật độ CÀNG CAO)
        density = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(D[i])
            # Bỏ qua chính nó (index 0), lấy từ 1 đến min_samples
            density[i] = np.mean(sorted_dists[1:self.min_samples+1])
            
        labels = np.full(n, -1)
        core_flags = np.zeros(n, dtype=bool)
        unclassified = set(range(n))
        cluster_id = 0
        
        # --- Giai đoạn 1: LE-DBSCAN (Hai chiều) ---
        while unclassified:
            # Chọn điểm có mật độ cao nhất (khoảng cách trung bình nhỏ nhất)
            p = min(unclassified, key=lambda idx: density[idx])
            
            # Tính Bán kính cục bộ (Local Eps) dựa trên k-láng giềng
            eps_j = np.sort(D[p])[self.k_neighbors]
            
            # Tìm láng giềng của p
            neighbors_p = np.where(D[p] <= eps_j)[0]
            
            if len(neighbors_p) >= self.min_samples:
                cluster_id += 1
                self.local_eps_dict[cluster_id] = eps_j
                labels[p] = cluster_id
                core_flags[p] = True
                unclassified.remove(p)
                
                # Bắt đầu loang cụm
                queue = list(neighbors_p)
                while queue:
                    q = queue.pop(0)
                    if q in unclassified:
                        unclassified.remove(q)
                        labels[q] = cluster_id
                    elif labels[q] == -1:
                        labels[q] = cluster_id
                        
                    # Chỉ mở rộng từ điểm thuộc cụm hiện tại, tránh set core_flags sai cho điểm cụm khác
                    if labels[q] == cluster_id:
                        neighbors_q = np.where(D[q] <= eps_j)[0]
                        if len(neighbors_q) >= self.min_samples:
                            core_flags[q] = True
                            for n_q in neighbors_q:
                                if n_q in unclassified or labels[n_q] == -1:
                                    if n_q not in queue:
                                        queue.append(n_q)
            else:
                unclassified.remove(p)
                labels[p] = -1
                
        # Gộp cụm quá nhỏ vào noise — ngưỡng tối thiểu là max(min_samples, 5)
        # để tránh cụm chỉ 2-3 điểm khi min_samples nhỏ
        min_cluster_size = max(self.min_samples, 5)
        for cid in set(labels) - {-1}:
            members = np.where(labels == cid)[0]
            if len(members) < min_cluster_size:
                labels[members] = -1
                core_flags[members] = False
                if cid in self.local_eps_dict:
                    del self.local_eps_dict[cid]

        # --- Giai đoạn 2: LE3W-DBSCAN (Ba chiều) ---
        unique_clusters = set(labels) - {-1}
        for c in unique_clusters:
            self.POS[c] = set()
            self.BND[c] = set()
            
        # Phân bổ điểm lõi vào POS
        for i in range(n):
            c = labels[i]
            if c != -1 and core_flags[i]:
                self.POS[c].add(i)
                
        # Phân bổ điểm biên (xét nhãn của các láng giềng)
        for i in range(n):
            c = labels[i]
            if c != -1 and not core_flags[i]:
                eps_c = self.local_eps_dict[c]
                neighbors = np.where(D[i] <= eps_c)[0]
                neighbor_labels = set(labels[neighbors]) - {-1}
                
                if len(neighbor_labels) == 1 and list(neighbor_labels)[0] == c:
                    self.POS[c].add(i)
                else:
                    for m in neighbor_labels:
                        self.BND[m].add(i)
                    
        # Phân bổ điểm nhiễu (Gán vào BND của cụm chứa điểm lõi gần nhất)
        core_indices = np.where(core_flags)[0]
        if len(core_indices) > 0:
            for i in range(n):
                if labels[i] == -1:
                    dists_to_cores = D[i, core_indices]
                    nearest_core = core_indices[np.argmin(dists_to_cores)]
                    c = labels[nearest_core]
                    self.BND[c].add(i)
                    
        return self