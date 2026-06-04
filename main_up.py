import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from models.threeway_dbscan import ThreeWayDBSCAN
from models.le3w_dbscan import LE3W_DBSCAN
from utils.metrics import calculate_soft_metrics
from utils.dataloader import load_dataset
from visualize_up import plot_paper_comparison

def main():
    data_dir = "data"
    os.makedirs("results", exist_ok=True)
    
    # 4 Dataset được dùng trong paper cho Figure 8, 9, 10, 11
    target_datasets = {
        'Aggregation': {'eps_3w': 0.09, 'minPts': 5, 'eta': 0.20, 'k_le': 10, 'fig': 8},
        'Compound': {'eps_3w': 0.13, 'minPts': 5, 'eta': 0.20, 'k_le': 16, 'fig': 9},
        'Pathbased': {'eps_3w': 0.13, 'minPts': 5, 'eta': 0.20, 'k_le': 17, 'fig': 10},
        'Flame': {'eps_3w': 0.15, 'minPts': 4, 'eta': 0.20, 'k_le': 11, 'fig': 11}
    }
    
    table2_data = []
    table4_data = []

    print(f"{'='*120}")
    print(f"{'BẮT ĐẦU CHẠY THỰC NGHIỆM ĐỐI CHỨNG LE3W-DBSCAN VÀ 3W-DBSCAN':^120}")
    print(f"{'='*120}\n")

    for ds_name, params in target_datasets.items():
        file_path_csv = os.path.join(data_dir, f"{ds_name}.csv")
        file_path_txt = os.path.join(data_dir, f"{ds_name}.txt")
        file_path = file_path_csv if os.path.exists(file_path_csv) else file_path_txt
        
        if not os.path.exists(file_path):
            continue
            
        print(f"Đang xử lý Dataset: {ds_name}...")
        X, y = load_dataset(file_path)
        n_samples = X.shape[0]
        X = MinMaxScaler().fit_transform(X)
        
        # 1. Chạy 3W-DBSCAN (Thuật toán cũ)
        tw_dbscan = ThreeWayDBSCAN(eps=params['eps_3w'], min_samples=params['minPts'], eta=params['eta']).fit(X)
        gamma_3w, alpha_3w, a_star_3w = calculate_soft_metrics(tw_dbscan, n_samples)
        
        # 2. Chạy LE3W-DBSCAN (Thuật toán cải tiến)
        le3w_dbscan = LE3W_DBSCAN(min_samples=params['minPts'], k_neighbors=params['k_le']).fit(X)
        gamma_le, alpha_le, a_star_le = calculate_soft_metrics(le3w_dbscan, n_samples)
        
        # Lấy Bán kính cục bộ cho Table 2
        eps_dict = le3w_dbscan.local_eps_dict
        row_t2 = {'Dataset': ds_name, 'Classes': len(eps_dict)}
        for c_id, eps_val in eps_dict.items():
            row_t2[f'eps_C{c_id}'] = f"{eps_val:.4f}"
        table2_data.append(row_t2)
        
        # Ghi nhận kết quả Soft Metrics cho Table 4
        table4_data.append((ds_name, "alpha", alpha_3w, alpha_le))
        table4_data.append(("", "gamma", gamma_3w, gamma_le))
        table4_data.append(("", "alpha_star", a_star_3w, a_star_le))
        
        # Xuất biểu đồ so sánh giống Figure 8, 9, 10, 11
        plot_paper_comparison(X, y, tw_dbscan, le3w_dbscan, ds_name, params['fig'])
        print(f" -> Đã xuất ảnh: figures/Figure_{params['fig']}_{ds_name}.png")

    # --- IN BẢNG TABLE 2 ---
    print("\n" + "="*120)
    print(f"{'TABLE 2: LOCAL EPS OBTAINED BY LE-DBSCAN':^120}")
    print("="*120)
    df_t2 = pd.DataFrame(table2_data)
    df_t2 = df_t2.fillna("")
    print(df_t2.to_string(index=False))

    # --- IN BẢNG TABLE 4 VÀ XUẤT EXCEL ---
    if len(table4_data) > 0:
        columns = pd.MultiIndex.from_tuples([
            ("Dataset", ""), ("Metric", ""),
            ("3W-DBSCAN", ""), ("LE3W-DBSCAN", "")
        ])
        
        df_t4 = pd.DataFrame(table4_data, columns=columns)
        pd.options.display.float_format = '{:.3f}'.format
        
        print("\n" + "="*120)
        print(f"{'TABLE 4: EXPERIMENTAL RESULTS OF THREE-WAY CLUSTERING (SOFT METRICS)':^120}")
        print("="*120)
        print(df_t4.to_string(index=False))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("results", f"Table4_LE3W_{timestamp}.xlsx")
        
        df_export = df_t4.set_index([("Dataset", ""), ("Metric", "")])
        df_export.to_excel(filepath)
        print(f"\n[THÀNH CÔNG] Bảng Table 4 đã được lưu ra file Excel tại: {filepath}")

if __name__ == "__main__":
    main()