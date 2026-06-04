import numpy as np


class LE3W_DBSCAN:
    def __init__(self, min_samples, k_neighbors, min_merge_size=15):
        self.min_samples     = min_samples
        self.k_neighbors     = k_neighbors
        self.min_merge_size  = min_merge_size  # Mức 2: ngưỡng gộp cụm nhỏ
        self.POS             = {}
        self.BND             = {}
        self.local_eps_dict  = {}

    def fit(self, X):
        n = X.shape[0]
        D = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

        # Tính mật độ cục bộ
        density = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(D[i])
            density[i] = np.mean(sorted_dists[1:self.min_samples + 1])

        labels      = np.full(n, -1)
        core_flags  = np.zeros(n, dtype=bool)
        unclassified = set(range(n))
        cluster_id  = 0

        # ── Giai đoạn 1: LE-DBSCAN ─────────────────────────────────────
        while unclassified:
            p = min(unclassified, key=lambda idx: density[idx])

            # Bán kính cục bộ: k-th nearest neighbor (giữ nguyên độ chính xác)
            eps_j = np.sort(D[p])[self.k_neighbors]

            neighbors_p = np.where(D[p] <= eps_j)[0]

            if len(neighbors_p) >= self.min_samples:
                cluster_id += 1
                self.local_eps_dict[cluster_id] = eps_j
                labels[p]     = cluster_id
                core_flags[p] = True
                unclassified.remove(p)

                queue = list(neighbors_p)
                while queue:
                    q = queue.pop(0)
                    if q in unclassified:
                        unclassified.remove(q)
                        labels[q] = cluster_id
                    elif labels[q] == -1:
                        labels[q] = cluster_id

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

        # ── Giai đoạn 2: LE3W (Ba chiều) ────────────────────────────────
        unique_clusters = set(labels) - {-1}
        for c in unique_clusters:
            self.POS[c] = set()
            self.BND[c] = set()

        for i in range(n):
            c = labels[i]
            if c != -1 and core_flags[i]:
                self.POS[c].add(i)

        for i in range(n):
            c = labels[i]
            if c != -1 and not core_flags[i]:
                eps_c     = self.local_eps_dict[c]
                neighbors = np.where(D[i] <= eps_c)[0]
                neighbor_labels = set(labels[neighbors]) - {-1}

                if len(neighbor_labels) == 1 and list(neighbor_labels)[0] == c:
                    self.POS[c].add(i)
                else:
                    for m in neighbor_labels:
                        self.BND[m].add(i)
                    self.BND[c].add(i)

        core_indices = np.where(core_flags)[0]
        if len(core_indices) > 0:
            for i in range(n):
                if labels[i] == -1:
                    dists_to_cores = D[i, core_indices]
                    nearest_core   = core_indices[np.argmin(dists_to_cores)]
                    c              = labels[nearest_core]
                    self.BND[c].add(i)

        return self

    def _merge_small_clusters(self, labels, X):
        """Gộp cụm nhỏ vào cụm lớn gần nhất — chỉ khi khoảng cách tâm < max_merge_dist."""
        # Ngưỡng khoảng cách tối đa: 30% dải biến động của dữ liệu
        data_range    = np.max(X, axis=0) - np.min(X, axis=0)
        max_merge_dist = 0.30 * np.linalg.norm(data_range)

        changed = True
        while changed:
            changed = False
            cluster_ids   = list(set(labels) - {-1})
            cluster_sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}

            small = [c for c, sz in cluster_sizes.items() if sz < self.min_merge_size]
            large = [c for c, sz in cluster_sizes.items() if sz >= self.min_merge_size]

            if not small or not large:
                break

            large_centroids = {c: np.mean(X[labels == c], axis=0) for c in large}

            for sc in small:
                sc_centroid = np.mean(X[labels == sc], axis=0)
                nearest     = min(large, key=lambda lc:
                                  np.linalg.norm(sc_centroid - large_centroids[lc]))
                dist        = np.linalg.norm(sc_centroid - large_centroids[nearest])

                if dist < max_merge_dist:   # chỉ gộp nếu đủ gần
                    labels[labels == sc] = nearest
                    changed = True
                    break   # restart sau mỗi lần gộp

        return labels
