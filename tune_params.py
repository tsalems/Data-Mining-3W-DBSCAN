"""
Script tìm tham số (MinPts, k) cho LE3W-DBSCAN bằng cách so sánh
local eps thu được với Table 2 của bài báo Shen (2023).

Cách dùng:
  python tune_params.py

Thay đổi SEARCH_GRID và PAPER_TABLE2 bên dưới để điều chỉnh.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.le3w_dbscan import LE3W_DBSCAN
from utils.dataloader import load_dataset
import os

# ─── Giá trị local eps trong Table 2 của bài báo Shen (2023) ───────────────
PAPER_TABLE2 = {
    # Aggregation và Pathbased đã tìm được tham số tốt → bỏ khỏi search
    # 'Aggregation': [0.0513, 0.0573, 0.0633, 0.0690, 0.0735, 0.0735, 0.0635],
    # 'Compound':    [0.0495, 0.0556, 0.0391, 0.0363, 0.0401, 0.1588],
    # 'Pathbased':   [0.0561, 0.0733, 0.1998],
    'Flame':       [0.0628, 0.0680],
    'Iris':        [0.3498, 0.2927, 0.6089],
    'Seeds':       [0.3092, 0.3993, 0.5265],
}

# ─── Không gian tìm kiếm tham số ────────────────────────────────────────────
SEARCH_GRID = {
    'Aggregation': {'minPts_range': range(5, 20),  'k_range': range(10, 40)},
    'Compound':    {'minPts_range': range(3, 15),  'k_range': range(5,  30)},
    'Pathbased':   {'minPts_range': range(3, 15),  'k_range': range(5,  25)},
    # Flame: paper eps=[0.0628,0.0680] nhỏ hơn kết quả cũ → thử k nhỏ hơn (3..15)
    'Flame':       {'minPts_range': range(3, 20),  'k_range': range(3,  20)},
    # Iris: paper eps=[0.35,0.29,0.61] lớn hơn rất nhiều → thử k lớn hơn (lên 80)
    'Iris':        {'minPts_range': range(3, 20),  'k_range': range(5,  80)},
    # Seeds: paper dùng 197 mẫu, ta có 210 → thử dải rộng hơn
    'Seeds':       {'minPts_range': range(3, 30),  'k_range': range(5,  60)},
}

DATA_DIR = "data"


def eps_error(found: list, target: list) -> float:
    """
    Tính sai số giữa local eps tìm được và Table 2.
    Chỉ so sánh nếu số cụm khớp nhau.
    """
    if len(found) != len(target):
        return float('inf')
    # Sắp xếp cả hai để so sánh độc lập với thứ tự phát hiện
    found_s  = sorted(found)
    target_s = sorted(target)
    return np.mean([abs(f - t) for f, t in zip(found_s, target_s)])


def tune_dataset(ds_name: str, file_path: str, paper_eps: list,
                 minPts_range, k_range) -> pd.DataFrame:
    X, _ = load_dataset(file_path)
    X = MinMaxScaler().fit_transform(X)
    n_clusters_target = len(paper_eps)

    results = []
    for minPts in minPts_range:
        for k in k_range:
            if k <= minPts:          # k phải > minPts theo thiết kế thuật toán
                continue
            if k >= X.shape[0]:
                continue
            try:
                model = LE3W_DBSCAN(min_samples=minPts, k_neighbors=k).fit(X)
                found_eps = sorted(model.local_eps_dict.values())
                n_found   = len(found_eps)
                error     = eps_error(found_eps, paper_eps)
                results.append({
                    'minPts': minPts,
                    'k':      k,
                    'n_clusters_found': n_found,
                    'n_clusters_target': n_clusters_target,
                    'mean_eps_error': round(error, 6),
                    'found_eps': [round(e, 4) for e in found_eps],
                })
            except Exception:
                pass

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Ưu tiên: đúng số cluster trước, rồi sai số eps nhỏ nhất
    df_correct = df[df['n_clusters_found'] == n_clusters_target]
    if not df_correct.empty:
        df_correct = df_correct.sort_values('mean_eps_error')
        return df_correct
    else:
        # Nếu không tìm được đúng số cluster, hiển thị các kết quả gần nhất
        df = df.sort_values(['mean_eps_error'])
        return df.head(10)


def main():
    print("=" * 90)
    print(f"{'TÌM THAM SỐ CHO LE3W-DBSCAN — SO SÁNH VỚI TABLE 2 BÀI BÁO':^90}")
    print("=" * 90)

    for ds_name, paper_eps in PAPER_TABLE2.items():
        # Thử cả .csv và .txt
        path = None
        for ext in ('.csv', '.txt'):
            candidate = os.path.join(DATA_DIR, ds_name + ext)
            if os.path.exists(candidate):
                path = candidate
                break
        # Một số dataset tên file viết hoa/thường khác nhau
        if path is None:
            for ext in ('.csv', '.txt'):
                candidate = os.path.join(DATA_DIR, ds_name.lower() + ext)
                if os.path.exists(candidate):
                    path = candidate
                    break

        if path is None:
            print(f"\n[SKIP] {ds_name}: không tìm thấy file dữ liệu")
            continue

        grid = SEARCH_GRID.get(ds_name, {
            'minPts_range': range(3, 25),
            'k_range': range(5, 40),
        })

        print(f"\n{'─'*90}")
        print(f"Dataset: {ds_name}  |  Mục tiêu: {len(paper_eps)} cụm  |  "
              f"Paper eps: {paper_eps}")
        print(f"{'─'*90}")

        df = tune_dataset(
            ds_name, path, paper_eps,
            grid['minPts_range'], grid['k_range']
        )

        if df.empty:
            print("  Không có kết quả nào.")
            continue

        top5 = df.head(5)
        for _, row in top5.iterrows():
            marker = "✓" if row['n_clusters_found'] == row['n_clusters_target'] else "✗"
            print(f"  {marker} minPts={int(row['minPts']):3d}  k={int(row['k']):3d}  "
                  f"clusters={int(row['n_clusters_found'])}/{int(row['n_clusters_target'])}  "
                  f"eps_error={row['mean_eps_error']:.5f}  "
                  f"eps={row['found_eps']}")

    print(f"\n{'='*90}")
    print("Gợi ý: Cập nhật tham số tốt nhất vào main_up.py (cột minPts_le và k_le).")
    print("='*90")


if __name__ == "__main__":
    main()
