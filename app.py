import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

# Import các module đã xây dựng trong đồ án
from models.threeway_dbscan import ThreeWayDBSCAN
from models.le3w_dbscan import LE3W_DBSCAN
from utils.metrics import calculate_soft_metrics
from utils.dataloader import load_dataset
from visualize_up import plot_original, plot_threeway

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Hệ thống Điều phối Giao hàng", page_icon="📦", layout="wide")

# Áp dụng một chút CSS để giao diện nhìn hiện đại hơn
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📦 SMART-LOGISTICS: HỆ THỐNG GOM CỤM ĐƠN GIAO HÀNG")
st.markdown("### Ứng dụng thuật toán LE3W-DBSCAN để tối ưu hóa tuyến đường cho Shipper đa mật độ")
st.markdown("---")

# ==========================================
# THANH BÊN (SIDEBAR) - ĐIỀU KHIỂN
# ==========================================
with st.sidebar:
    st.header("⚙️ TRUNG TÂM ĐIỀU PHỐI")
    
    # Chọn dữ liệu
    st.subheader("1. Dữ liệu Đơn hàng")
    dataset_name = st.selectbox(
        "Chọn khu vực mô phỏng (Bản đồ)", 
        ["Hanoi_Deliveries", "Aggregation", "Compound", "Flame"] # Đã thêm Hanoi_Deliveries
    )
    
    st.markdown("---")
    st.subheader("2. Cấu hình Thuật toán")
    
    min_pts = st.slider("Chỉ tiêu đơn/chuyến (MinPts)", min_value=2, max_value=20, value=5, 
                        help="Số lượng đơn hàng tối thiểu để tạo thành một cụm giao hàng.")
    
    eps_3w = st.slider("Bán kính gom đơn chung (Epsilon)", 0.01, 1.0, 0.13, 0.01,
                       help="Dùng cho DBSCAN và 3W-DBSCAN gốc.")
    
    eta_3w = st.slider("Hệ số vùng lân cận (Eta)", 0.05, 0.5, 0.20, 0.01,
                       help="Dùng cho 3W-DBSCAN gốc.")
    
    k_neighbors = st.slider("Độ nhạy bù trừ khoảng cách (K-Neighbors)", 2, 30, 15,
                            help="Tham số cốt lõi của LE3W-DBSCAN để tự động điều chỉnh bán kính khu vực thưa/đặc.")
    
    st.markdown("---")
    run_btn = st.button("🚀 Phân Cụm Đơn Hàng", type="primary")

# ==========================================
# KHU VỰC HIỂN THỊ CHÍNH
# ==========================================
if run_btn:
    with st.spinner('Đang tính toán tuyến đường và chia cụm đơn...'):
        # 1. Tải và chuẩn hóa dữ liệu
        try:
            X, y = load_dataset(f"data/{dataset_name}.csv")
        except:
            X, y = load_dataset(f"data/{dataset_name}.txt")
            
        X = MinMaxScaler().fit_transform(X)
        n_samples = X.shape[0]
        
        # Tạo 3 cột hiển thị
        col1, col2, col3 = st.columns(3)
        
        # --- CỘT 1: DBSCAN GỐC ---
        with col1:
            st.markdown("<h4 style='text-align: center; color: #e74c3c;'>1. Điều phối Cũ (DBSCAN)</h4>", unsafe_allow_html=True)
            dbscan = DBSCAN(eps=eps_3w, min_samples=min_pts).fit(X)
            
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            plot_original(ax1, X, dbscan.labels_, "Chia cụm cứng nhắc")
            st.pyplot(fig1)
            
            st.error("**Nhược điểm:** Kém hiệu quả. Khách ngoại thành bị hủy đơn (coi là nhiễu). Cụm trung tâm quá tải.")
            
        # --- CỘT 2: 3W-DBSCAN GỐC ---
        with col2:
            st.markdown("<h4 style='text-align: center; color: #f39c12;'>2. Điều phối 3 Vùng (3W-DBSCAN)</h4>", unsafe_allow_html=True)
            tw_model = ThreeWayDBSCAN(eps=eps_3w, min_samples=min_pts, eta=eta_3w).fit(X)
            _, _, a_star_tw = calculate_soft_metrics(tw_model, n_samples)
            
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            plot_threeway(ax2, X, tw_model, "Chiến lược Điểm tập kết")
            st.pyplot(fig2)
            
            st.warning(f"**Cải thiện:** Đã phân tách được Vùng lõi và Vùng lân cận. Tuy nhiên vẫn bỏ sót cụm ngoại thành.\n\n**Độ tin cậy (α*):** {a_star_tw:.3f}")
            
        # --- CỘT 3: LE3W-DBSCAN (ĐỀ XUẤT) ---
        with col3:
            st.markdown("<h4 style='text-align: center; color: #27ae60;'>3. Điều phối Thông minh (LE3W-DBSCAN)</h4>", unsafe_allow_html=True)
            le3w_model = LE3W_DBSCAN(min_samples=min_pts, k_neighbors=k_neighbors).fit(X)
            _, _, a_star_le = calculate_soft_metrics(le3w_model, n_samples)
            
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            plot_threeway(ax3, X, le3w_model, "Bù trừ mật độ (Local Eps)")
            st.pyplot(fig3)
            
            st.success(f"**Tối ưu hoàn hảo:** Nhận diện và chia cụm vừa sức cho Shipper ở mọi khu vực mật độ.\n\n**Độ tin cậy (α*):** {a_star_le:.3f}")

        # ==========================================
        # HƯỚNG DẪN NGHIỆP VỤ (STORYTELLING)
        # ==========================================
        st.markdown("---")
        st.subheader("📋 KẾ HOẠCH GIAO HÀNG TỰ ĐỘNG (DỰA TRÊN 3 CHIỀU)")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.info("🔴 **CỤM LÕI (VÙNG POS)**\n\n"
                    "- **Mật độ đơn:** Rất dày (Tòa nhà chung cư, Khu văn phòng).\n"
                    "- **Hành động:** Thiết lập điểm tập kết. Shipper đỗ xe 1 chỗ, gọi đồng loạt khách hàng xuống sảnh nhận để tiết kiệm thời gian.")
            
        with info_col2:
            st.warning("🟡 **CỤM LÂN CẬN (VÙNG BND)**\n\n"
                       "- **Mật độ đơn:** Rải rác quanh khu vực trung tâm.\n"
                       "- **Hành động:** Lên lộ trình chạy xe máy luồn lách vào từng ngõ ngách để giao tận cửa nhà cho khách hàng.")
            
        with info_col3:
            st.error("⚪ **ĐƠN NẰM NGOÀI CỤM (VÙNG NEG)**\n\n"
                     "- **Mật độ đơn:** Quá xa, đơn lẻ tẻ, không bõ công chạy.\n"
                     "- **Hành động:** Tạm thời không gắn vào cuốc xe hiện tại. Chờ hệ thống gom thêm đơn hoặc tính phụ phí giao xa.")
else:
    st.info("👈 Hãy tùy chỉnh các tham số bên thanh điều khiển và bấm **Phân Cụm Đơn Hàng** để bắt đầu.")