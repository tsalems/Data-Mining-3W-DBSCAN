import numpy as np, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
sys.path.insert(0, 'd:/STUDY/KPDL/Data-Mining-3W-DBSCAN')

from models.le3w_dbscan import LE3W_DBSCAN
from utils.dataloader import load_dataset
from utils.metrics import calculate_accuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def get_upper_labels(model, n):
    labels = np.full(n, -1)
    for c in model.POS:
        for i in model.POS[c]: labels[i] = c
        for i in model.BND[c]:
            if labels[i] == -1: labels[i] = c
    return labels

def search(ds_name, n_true, mp_range, k_range):
    X, y = load_dataset(f'data/{ds_name}.csv')
    X = MinMaxScaler().fit_transform(X)
    n = X.shape[0]
    best_exact, best_any, top5 = None, None, []
    for minpts in range(mp_range[0], mp_range[1] + 1):
        for k in range(k_range[0], min(k_range[1], n - 1) + 1):
            if k <= minpts:
                continue
            try:
                model = LE3W_DBSCAN(min_samples=minpts, k_neighbors=k).fit(X)
            except Exception:
                continue
            nf = len(model.POS)
            if nf == 0:
                continue
            pred = get_upper_labels(model, n)
            mask = pred != -1
            if mask.sum() < 10:
                continue
            nmi = normalized_mutual_info_score(y[mask], pred[mask])
            ari = adjusted_rand_score(y[mask], pred[mask])
            acc = float(calculate_accuracy(y[mask], pred[mask]))
            row = dict(minPts=minpts, k=k, n_found=nf,
                       NMI=round(nmi, 4), ARI=round(ari, 4), ACC=round(acc, 4))
            if nf == n_true:
                top5.append(row)
                if best_exact is None or nmi > best_exact['NMI']:
                    best_exact = row
            if best_any is None or nmi > best_any['NMI']:
                best_any = row
    return best_exact, best_any, sorted(top5, key=lambda x: -x['NMI'])[:5]

DATASETS = {
    '3L':        (3,  (2, 20), (5, 50)),
    '4C':        (4,  (2, 20), (5, 50)),
    'Compound':  (6,  (2, 15), (5, 40)),
    'S1':        (15, (5, 25), (10, 60)),
    'Pathbased': (3,  (2, 20), (5, 50)),
}

print('Bat dau grid search cho 5 datasets...\n')
for ds, (n_true, mp, kr) in DATASETS.items():
    t0 = time.time()
    print(f'Dang xu ly {ds} (n_true={n_true})...', flush=True)
    be, ba, top5 = search(ds, n_true, mp, kr)
    elapsed = round(time.time() - t0, 1)

    print(f'\n{"="*60}')
    print(f'Dataset: {ds}  (n_true={n_true})  [{elapsed}s]')
    print(f'{"="*60}')

    if be:
        print(f'  DUNG SO CUM : minPts={be["minPts"]}, k={be["k"]} '
              f'-> n_found={be["n_found"]}, NMI={be["NMI"]}, ARI={be["ARI"]}, ACC={be["ACC"]}')
    else:
        print(f'  DUNG SO CUM : KHONG TIM DUOC')

    if ba and (be is None or ba['NMI'] > be['NMI'] + 0.01):
        print(f'  TOT NHAT    : minPts={ba["minPts"]}, k={ba["k"]} '
              f'-> n_found={ba["n_found"]}, NMI={ba["NMI"]}, ARI={ba["ARI"]}, ACC={ba["ACC"]}')

    if top5:
        print(f'  Top 5 voi dung {n_true} cum (theo NMI):')
        for r in top5:
            print(f'    minPts={r["minPts"]}, k={r["k"]} '
                  f'-> NMI={r["NMI"]}, ARI={r["ARI"]}, ACC={r["ACC"]}')
    print()

print('HOAN THANH!')
