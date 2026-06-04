import numpy as np
import pandas as pd
import os

def generate_hanoi_deliveries():
    np.random.seed(42) # Cố định seed để ra hình giống nhau mỗi lần chạy
    
    # 1. Khu vực 1: Lõi Hoàn Kiếm (Mật độ CỰC ĐẶC)
    # Rất nhiều đơn hàng, diện tích hẹp
    hoan_kiem = np.random.normal(loc=[105.852, 21.028], scale=[0.003, 0.003], size=(300, 2))
    label_hk = np.full(300, 0) # Nhãn cụm 0

    # 2. Khu vực 2: Trục Cầu Giấy - Mỹ Đình (Mật độ VỪA)
    # Đơn hàng phân bố rải rác hơn
    cau_giay = np.random.normal(loc=[105.785, 21.037], scale=[0.008, 0.008], size=(150, 2))
    label_cg = np.full(150, 1) # Nhãn cụm 1

    # 3. Khu vực 3: Ngoại thành Hà Đông (Mật độ THƯA THỚT)
    # Đơn hàng nằm cách xa nhau, trải trên diện tích rộng
    ha_dong = np.random.normal(loc=[105.772, 20.973], scale=[0.015, 0.015], size=(60, 2))
    label_hd = np.full(60, 2) # Nhãn cụm 2

    # 4. Đơn rác (Nhiễu / Noise)
    # Các đơn hàng lẻ tẻ nằm rải rác khắp bản đồ Hà Nội, không bõ công Shipper chạy
    noise = np.random.uniform(low=[105.740, 20.950], high=[105.880, 21.060], size=(40, 2))
    label_noise = np.full(40, -1) # Nhãn nhiễu là -1

    # Gộp tất cả lại thành 1 bộ dữ liệu
    X = np.vstack([hoan_kiem, cau_giay, ha_dong, noise])
    y = np.concatenate([label_hk, label_cg, label_hd, label_noise])

    # Lưu thành file CSV (Format: x, y, label - giống hệt các file Benchmark cũ)
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame({'lon': X[:, 0], 'lat': X[:, 1], 'label': y})
    df.to_csv("data/Hanoi_Deliveries.csv", index=False, header=False)
    
    print(f"✅ Đã tạo thành công bộ dữ liệu bản đồ Hà Nội với {len(X)} đơn hàng!")
    print("File lưu tại: data/Hanoi_Deliveries.csv")

if __name__ == "__main__":
    generate_hanoi_deliveries()