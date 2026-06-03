import numpy as np
from sklearn.cluster import KMeans

class CE3KMeans:
    """
    Thuật toán đối chứng CE3-kmeans (Three-way K-Means).
    """
    def __init__(self, n_clusters, threshold=0.8, random_state=42):
        self.n_clusters = n_clusters
        self.threshold = threshold # Ngưỡng quyết định vùng POS hay BND
        self.random_state = random_state
        self.POS = {}
        self.BND = {}
        
    def fit(self, X):
        # 1. Chạy K-means truyền thống
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        
        for k in range(self.n_clusters):
            self.POS[k] = set()
            self.BND[k] = set()
            
        # 2. Phân bổ vào 3 vùng (Three-way decision)
        for i, x in enumerate(X):
            # Tính khoảng cách từ điểm hiện tại tới tất cả các tâm cụm
            dists = np.linalg.norm(centers - x, axis=1)
            sorted_idx = np.argsort(dists)
            
            c1, c2 = sorted_idx[0], sorted_idx[1] # 2 cụm gần nhất
            d1, d2 = dists[c1], dists[c2]
            
            if d2 == 0:
                self.POS[c1].add(i)
            elif d1 / d2 <= self.threshold:
                # Gần tâm c1 rõ rệt -> Thuộc vùng chắc chắn (POS)
                self.POS[c1].add(i)
            else:
                # Lấp lửng giữa c1 và c2 -> Thuộc vùng biên (BND)
                self.BND[c1].add(i)
                self.BND[c2].add(i)
                
        return self