"""
Grid search tìm tham số tối ưu (minPts, k) cho LE3W-DBSCAN trên từng dataset.
Tiêu chí: NMI cao nhất, ưu tiên số cụm đúng với ground truth.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings, os
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.le3w_dbscan import LE3W_DBSCAN
from utils.dataloader import load_dataset
from utils.metrics import calculate_accuracy, match_labels

# -----------------------------------------------------------------------
# Cấu hình dataset: tên file, số cụm đúng, phạm vi tìm kiếm
# -----------------------------------------------------------------------
DATASETS = {
    'Aggregation': {'n_clusters': 7,  'minpts_range': (5, 20),  'k_range': (5, 40)},
    'Compound':    {'n_clusters': 6,  'minpts_range': (3, 20),  'k_range': (5, 50)},
    'Pathbased':   {'n_clusters': 3,  'minpts_range': (3, 20),  'k_range': (5, 50)},
    'Flame':       {'n_clusters': 2,  'minpts_range': (3, 15),  'k_range': (5, 40)},
    'IRIS':        {'n_clusters': 3,  'minpts_range': (3, 15),  'k_range': (5, 40)},
    'Seeds':       {'n_clusters': 3,  'minpts_range': (3, 15),  'k_range': (5, 40)},
}

def get_upper_labels(model, n):
    """Tạo mảng nhãn từ POS+BND (upper bound) cho đánh giá hard clustering."""
    labels = np.full(n, -1)
    for c in model.POS:
        for i in model.POS[c]:
            labels[i] = c
        for i in model.BND[c]:
            if labels[i] == -1:   # ưu tiên POS nếu trùng
                labels[i] = c
    return labels

def search_dataset(ds_name, config):
    file_csv = f'data/{ds_name}.csv'
    file_txt = f'data/{ds_name}.txt'
    file_path = file_csv if os.path.exists(file_csv) else file_txt
    if not os.path.exists(file_path):
        print(f'  [SKIP] Khong tim thay file {file_path}')
        return None

    X, y = load_dataset(file_path)
    X = MinMaxScaler().fit_transform(X)
    n = X.shape[0]
    n_true = config['n_clusters']
    mp_min, mp_max = config['minpts_range']
    k_min,  k_max  = config['k_range']

    best_exact   = None   # kết quả tốt nhất khi đúng số cụm
    best_any     = None   # kết quả tốt nhất bất kể số cụm

    results = []

    for minpts in range(mp_min, mp_max + 1):
        for k in range(k_min, min(k_max, n - 1) + 1):
            if k <= minpts:
                continue
            try:
                model = LE3W_DBSCAN(min_samples=minpts, k_neighbors=k).fit(X)
            except Exception:
                continue

            n_found = len(model.POS)
            if n_found == 0:
                continue

            pred = get_upper_labels(model, n)
            # Lọc noise khỏi đánh giá
            mask = pred != -1
            if mask.sum() < 10:
                continue

            nmi  = normalized_mutual_info_score(y[mask], pred[mask])
            ari  = adjusted_rand_score(y[mask], pred[mask])
            acc  = calculate_accuracy(y[mask], pred[mask])
            score = nmi   # tiêu chí chính

            row = dict(minPts=minpts, k=k, n_clusters=n_found,
                       NMI=round(nmi, 4), ARI=round(ari, 4), ACC=round(acc, 4))
            results.append(row)

            if n_found == n_true:
                if best_exact is None or score > best_exact['NMI']:
                    best_exact = row
            if best_any is None or score > best_any['NMI']:
                best_any = row

    return results, best_exact, best_any


def main():
    summary = []
    for ds_name, config in DATASETS.items():
        print(f'\n{"="*60}')
        print(f'Dataset: {ds_name}  (n_true_clusters={config["n_clusters"]})')
        print(f'{"="*60}')

        out = search_dataset(ds_name, config)
        if out is None:
            continue
        results, best_exact, best_any = out

        if best_exact:
            print(f'[DUNG SO CUM] minPts={best_exact["minPts"]}, k={best_exact["k"]} '
                  f'-> NMI={best_exact["NMI"]}, ARI={best_exact["ARI"]}, ACC={best_exact["ACC"]}')
        else:
            print(f'[KHONG TIM DUOC voi {config["n_clusters"]} cum]')

        if best_any:
            print(f'[TOT NHAT BAT KE] minPts={best_any["minPts"]}, k={best_any["k"]} '
                  f'-> n_clusters={best_any["n_clusters"]}, '
                  f'NMI={best_any["NMI"]}, ARI={best_any["ARI"]}, ACC={best_any["ACC"]}')

        # Top 5 kết quả đúng số cụm
        exact = [r for r in results if r['n_clusters'] == config['n_clusters']]
        exact_sorted = sorted(exact, key=lambda x: x['NMI'], reverse=True)[:5]
        if exact_sorted:
            print(f'\n  Top 5 (dung {config["n_clusters"]} cum, sap xep theo NMI):')
            for r in exact_sorted:
                print(f'    minPts={r["minPts"]}, k={r["k"]} -> '
                      f'NMI={r["NMI"]}, ARI={r["ARI"]}, ACC={r["ACC"]}')

        best = best_exact if best_exact else best_any
        if best:
            summary.append({
                'Dataset': ds_name,
                'n_true': config['n_clusters'],
                'n_found': best['n_clusters'],
                'minPts': best['minPts'],
                'k': best['k'],
                'NMI': best['NMI'],
                'ARI': best['ARI'],
                'ACC': best['ACC'],
            })

    print(f'\n\n{"="*70}')
    print('TONG KET - THAM SO TOI UU CHO TUNG DATASET')
    print(f'{"="*70}')
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    print('\nCopy vao main_up.py:')
    for row in summary:
        print(f'  \'{row["Dataset"]}\': '
              f'{{"minPts": {row["minPts"]}, "k_le": {row["k"]}, ...}},')


if __name__ == '__main__':
    main()
