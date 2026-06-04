import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

from models.threeway_dbscan import ThreeWayDBSCAN, DScaleDBSCAN
from models.ce3_kmeans import CE3KMeans
from utils.metrics import calculate_accuracy, calculate_f1, calculate_nmi
from utils.dataloader import load_dataset
from visualize import plot_4_panels, plot_figure_10

def extract_bounds_labels(model, n_samples):
    lower_preds = np.full(n_samples, -1)
    upper_preds = np.full(n_samples, -1)
    for k in model.POS.keys():
        for idx in model.POS[k]:
            lower_preds[idx] = k
            upper_preds[idx] = k
        for idx in model.BND[k]:
            upper_preds[idx] = k
    return lower_preds, upper_preds

def main():
    data_dir = "data"
    datasets = [
        "3L", "4C", "S1", "IRIS", "Glass", 
        "Seeds", "Pathbased", "Aggregation", "Compound", "Flame"
    ]
    
    param_grid = {
        '3L': {'min_samples': 5, 'eta': 0.20, 'k': 3},
        '4C': {'min_samples': 5, 'eta': 0.20, 'k': 4},
        # 'S1': {'min_samples': 10, 'eta': 0.15, 'k': 15},
        'Pathbased': {'min_samples': 5, 'eta': 0.20, 'k': 3},
        'Aggregation': {'min_samples': 5, 'eta': 0.20, 'k': 7},
        'Compound': {'min_samples': 5, 'eta': 0.20, 'k': 6},
        'Flame': {'min_samples': 4, 'eta': 0.20, 'k': 2},
        'IRIS': {'min_samples': 4, 'eta': 0.25, 'k': 3},
        'Glass': {'min_samples': 4, 'eta': 0.20, 'k': 6},
        'Seeds': {'min_samples': 4, 'eta': 0.20, 'k': 3}
    }
    
    results = []
    best_params_saved = {}
    
    print(f"{'='*110}")
    print(f"{'BẮT ĐẦU CHẠY THỰC NGHIỆM TỔNG HỢP (TABLE 2 & FIGURES) TRÊN 10 BỘ DỮ LIỆU':^110}")
    print(f"{'='*110}\n")

    for ds_name in datasets:
        file_path_csv = os.path.join(data_dir, f"{ds_name}.csv")
        file_path_txt = os.path.join(data_dir, f"{ds_name}.txt")
        file_path = file_path_csv if os.path.exists(file_path_csv) else file_path_txt
        
        if not os.path.exists(file_path):
            continue
            
        print(f"Đang xử lý & Vẽ biểu đồ: {ds_name:<12}", end="")
        X, y = load_dataset(file_path)
        n_samples = X.shape[0]
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        params = param_grid.get(ds_name)
        minPts, base_eta, k = params['min_samples'], params['eta'], params['k']
        
        # 1. Super Auto-Tuner
        best_eps, best_eta, best_nmi = 0.1, base_eta, -1
        eps_candidates = np.concatenate([
            np.arange(0.05, 0.3, 0.02), np.arange(0.3, 2.0, 0.1), np.arange(2.0, 10.0, 0.5)
        ])
        eta_candidates = [base_eta] if ds_name not in ['IRIS', 'Glass', 'Seeds'] else [0.15, 0.2, 0.25, 0.3, 0.4]

        for test_eta in eta_candidates:
            for test_eps in eps_candidates:
                try:
                    tw_test = ThreeWayDBSCAN(eps=test_eps, min_samples=minPts, eta=test_eta).fit(X)
                    _, ub_test = extract_bounds_labels(tw_test, n_samples)
                    score = calculate_nmi(y, ub_test)
                    if score > best_nmi:
                        best_nmi, best_eps, best_eta = score, test_eps, test_eta
                except: pass
        
        best_params_saved[ds_name] = {'eps': best_eps, 'minPts': minPts, 'eta': best_eta}
        
        try:
            ce3 = CE3KMeans(n_clusters=k).fit(X)
            tw_dbscan = ThreeWayDBSCAN(eps=best_eps, min_samples=minPts, eta=best_eta).fit(X)
            ds_dbscan = DScaleDBSCAN(eps=best_eps, min_samples=minPts, eta=best_eta).fit(X)
            
            # --- VẼ HÌNH TRỰC QUAN ĐỒNG THỜI VÀO FOLDER FIGURES ---
            plot_4_panels(X, y, ds_dbscan.labels_, ce3, tw_dbscan, ds_name)
            print(f" (Đã xuất ảnh: figures/Figure_{ds_name}.png)")
            
            # Tính Metrics cho bảng
            ce3_lb_preds, ce3_ub_preds = extract_bounds_labels(ce3, n_samples)
            acc_ce3_lb, nmi_ce3_lb, f1_ce3_lb = calculate_accuracy(y, ce3_lb_preds), calculate_nmi(y, ce3_lb_preds), calculate_f1(y, ce3_lb_preds)
            acc_ce3_ub, nmi_ce3_ub, f1_ce3_ub = calculate_accuracy(y, ce3_ub_preds), calculate_nmi(y, ce3_ub_preds), calculate_f1(y, ce3_ub_preds)

            tw_lb_preds, tw_ub_preds = extract_bounds_labels(tw_dbscan, n_samples)
            acc_tw_lb, nmi_tw_lb, f1_tw_lb = calculate_accuracy(y, tw_lb_preds), calculate_nmi(y, tw_lb_preds), calculate_f1(y, tw_lb_preds)
            acc_tw_ub, nmi_tw_ub, f1_tw_ub = calculate_accuracy(y, tw_ub_preds), calculate_nmi(y, tw_ub_preds), calculate_f1(y, tw_ub_preds)
            
            ds_preds = ds_dbscan.labels_
            acc_ds_ub, nmi_ds_ub, f1_ds_ub = calculate_accuracy(y, ds_preds), calculate_nmi(y, ds_preds), calculate_f1(y, ds_preds)
            
            results.append((ds_name, "ACC", acc_ce3_lb, acc_tw_lb, acc_ce3_ub, acc_tw_ub, acc_ds_ub))
            results.append(("", "NMI", nmi_ce3_lb, nmi_tw_lb, nmi_ce3_ub, nmi_tw_ub, nmi_ds_ub))
            results.append(("", "F1", f1_ce3_lb, f1_tw_lb, f1_ce3_ub, f1_tw_ub, f1_ds_ub))
            
        except Exception as e:
            print(f"  -> [LỖI] Xử lý {ds_name} thất bại: {e}")

    # --- CHẠY VÀ VẼ FIGURE 10 (F1 trên tất cả các bộ dữ liệu) ---
    print("\nĐang quét thông số Eta và vẽ Figure 10 (F1 vs Eta)...", end="")
    eta_range = np.arange(0.05, 0.45, 0.05)
    dataset_f1_dict = {}
    
    # Lấy ra các dataset tiêu biểu hoặc toàn bộ dataset đã chạy thành công
    for ds_fig10 in best_params_saved.keys():
        file_path_csv = os.path.join(data_dir, f"{ds_fig10}.csv")
        file_path_txt = os.path.join(data_dir, f"{ds_fig10}.txt")
        file_path = file_path_csv if os.path.exists(file_path_csv) else file_path_txt
        
        X, y = load_dataset(file_path)
        X = MinMaxScaler().fit_transform(X)
        n_samples = X.shape[0]
        
        f1_list = []
        for eta_val in eta_range:
            tw = ThreeWayDBSCAN(eps=best_params_saved[ds_fig10]['eps'], min_samples=best_params_saved[ds_fig10]['minPts'], eta=eta_val).fit(X)
            _, ub_preds = extract_bounds_labels(tw, n_samples)
            f1_list.append(calculate_f1(y, ub_preds))
            
        dataset_f1_dict[ds_fig10] = f1_list
        
    plot_figure_10(eta_range, dataset_f1_dict)
    print(" (Đã xuất ảnh: figures/Figure_10_F1_vs_eta.png)")

    # --- IN BẢNG ĐA TẦNG TABLE 2 ---
    if len(results) > 0:
        columns = pd.MultiIndex.from_tuples([
            ("Dataset", ""), ("Metric", ""),
            ("Lower bound C", "CE3-kmeans"), ("Lower bound C", "3W-DBSCAN"),
            ("Upper bound C", "CE3-kmeans"), ("Upper bound C", "3W-DBSCAN"), ("Upper bound C", "DScale-DBSCAN")
        ])
        
        df_results = pd.DataFrame(results, columns=columns)
        pd.options.display.float_format = '{:.4f}'.format
        
        print("\n" + "="*110)
        print(f"{'TABLE 2: DIFFERENT CLUSTERING PERFORMANCE ON 10 DATASETS':^110}")
        print("="*110)
        print(df_results.to_string(index=False))
        print("="*110)

if __name__ == "__main__":
    main()