import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

from models.threeway_dbscan import ThreeWayDBSCAN
from models.le3w_dbscan import LE3W_DBSCAN
from utils.metrics import calculate_soft_metrics

# ==========================================
# 1. TẠO DỮ LIỆU ĐƠN HÀNG (CACHE ĐỂ TRÁNH RESET)
# ==========================================
@st.cache_data
def load_live_deliveries():
    np.random.seed(42)
    hoan_kiem = np.random.normal(loc=[21.0285, 105.8542], scale=[0.002, 0.002], size=(200, 2))
    cau_giay = np.random.normal(loc=[21.0378, 105.7816], scale=[0.006, 0.006], size=(120, 2))
    ha_dong = np.random.normal(loc=[20.9760, 105.7700], scale=[0.012, 0.012], size=(50, 2))
    noise = np.random.uniform(low=[20.9500, 105.7400], high=[21.0600, 105.8800], size=(30, 2))
    
    data = np.vstack([hoan_kiem, cau_giay, ha_dong, noise])
    df = pd.DataFrame(data, columns=['Lat', 'Lon'])
    df['OrderID'] = [f"ORD-{str(i).zfill(4)}" for i in range(len(df))]
    return df

# ==========================================
# 2. HÀM VẼ BẢN ĐỒ VỚI HIỆU ỨNG NHẤP NHÁY
# ==========================================
def draw_dashboard_map(df, mode="Live", model=None):
    m = folium.Map(location=[21.0200, 105.8000], zoom_start=12, tiles='CartoDB dark_matter')
    
    # Kỹ thuật tiêm CSS để tạo hiệu ứng nhấp nháy như Radar
    blink_css = """
    {% macro html(this, kwargs) %}
    <style>
        .blink-dot {
            width: 14px; height: 14px;
            background-color: #f1c40f;
            border-radius: 50%;
            box-shadow: 0 0 12px #f1c40f;
            animation: pulsate 1.5s ease-out infinite;
        }
        @keyframes pulsate {
            0% { transform: scale(0.5); opacity: 1.0; }
            100% { transform: scale(1.5); opacity: 0.0; }
        }
    </style>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(blink_css)
    m.get_root().add_child(macro)

    if mode == "Live":
        # Vẽ các điểm nhấp nháy chờ xử lý
        for i, row in df.iterrows():
            icon_html = folium.DivIcon(html="<div class='blink-dot'></div>")
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                icon=icon_html,
                tooltip=f"<b>{row['OrderID']}</b><br>Trạng thái: Đang chờ Shipper"
            ).add_to(m)
            
    elif mode == "DBSCAN":
        labels = model.labels_
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c', '#1abc9c']
        for i, row in df.iterrows():
            cluster_id = labels[i]
            if cluster_id == -1:
                folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=3, color='gray', tooltip="Hủy").add_to(m)
            else:
                c_color = colors[cluster_id % len(colors)]
                folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=5, color=c_color, fill=True, tooltip=f"Cụm {cluster_id}").add_to(m)
                
    elif mode in ["3W", "LE3W"]:
        pos_points, bnd_points = set(), set()
        for v in model.POS.values(): pos_points.update(v)
        for v in model.BND.values(): bnd_points.update(v)
            
        for i, row in df.iterrows():
            if i in pos_points:
                folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=6, color='#e74c3c', fill=True, fill_opacity=0.8, tooltip="Điểm tập kết lõi").add_to(m)
            elif i in bnd_points:
                folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=5, color='#f39c12', fill=True, fill_opacity=0.5, tooltip="Giao tận nhà").add_to(m)
            else:
                folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=3, color='gray', fill=True, fill_opacity=0.3, tooltip="Rác / Hủy").add_to(m)
    return m

# ==========================================
# 3. GIAO DIỆN WEB & THỐNG KÊ DASHBOARD
# ==========================================
st.set_page_config(page_title="Live Logistics Dashboard", page_icon="🌐", layout="wide")
st.title("🌐 SMART-LOGISTICS: LIVE DISPATCH DASHBOARD")

df_orders = load_live_deliveries()
X = df_orders[['Lat', 'Lon']].values
X_scaled = MinMaxScaler().fit_transform(X)

# --- THANH SIDEBAR ---
with st.sidebar:
    st.header("🎛️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    
    # Selectbox chọn thuật toán
    mode = st.radio(
        "LỰA CHỌN THUẬT TOÁN ĐIỀU PHỐI:",
        ("🔴 Live (Chờ xử lý)", "🔵 1. DBSCAN (Cũ)", "🟡 2. 3W-DBSCAN", "🟢 3. LE3W-DBSCAN (Đề xuất)")
    )
    
    st.markdown("---")
    st.subheader("CẤU HÌNH THAM SỐ")
    min_pts = st.slider("Chỉ tiêu (MinPts)", 2, 20, 5)
    eps_val = st.slider("Bán kính quyét (Eps)", 0.01, 1.0, 0.13, 0.01)
    k_val = st.slider("Bù trừ mật độ (K-Neighbors)", 2, 30, 15)

# --- KHỐI THỐNG KÊ DASHBOARD (KPIs) ---
st.markdown("### 📊 THỐNG KÊ TRẠNG THÁI (REAL-TIME)")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

if mode == "🔴 Live (Chờ xử lý)":
    kpi1.metric("📦 Tổng đơn chờ", len(df_orders))
    kpi2.metric("🔴 Đơn lõi (POS)", "0", "-100%")
    kpi3.metric("🟡 Đơn lân cận (BND)", "0")
    kpi4.metric("⚪ Bỏ lọt / Hủy", len(df_orders))
    kpi5.metric("⭐ Độ tin cậy (α*)", "0.000")
    
    st.info("💡 **Trạng thái:** Các điểm vàng đang nhấp nháy báo hiệu hệ thống đang liên tục nhận đơn hàng mới. Hãy chọn thuật toán bên tay trái để bắt đầu chia đơn cho Shipper.")
    map_html = draw_dashboard_map(df_orders, mode="Live")
    st_folium(map_html, width=1300, height=550)

elif mode == "🔵 1. DBSCAN (Cũ)":
    model = DBSCAN(eps=eps_val, min_samples=min_pts).fit(X_scaled)
    n_noise = list(model.labels_).count(-1)
    
    kpi1.metric("📦 Tổng đơn", len(df_orders))
    kpi2.metric("🔴 Đơn phân cụm", f"{len(df_orders) - n_noise}")
    kpi3.metric("🟡 Đơn BND", "Không hỗ trợ")
    kpi4.metric("⚪ Bỏ lọt / Hủy", f"{n_noise}")
    kpi5.metric("⭐ Độ tin cậy (α*)", "N/A")
    
    st.error("❌ **Đánh giá DBSCAN:** Lãng phí! Gom quá nhiều đơn thành cụm khổng lồ, Shipper không thể chạy nổi, trong khi ngoại thành thì bị hủy sạch.")
    map_html = draw_dashboard_map(df_orders, mode="DBSCAN", model=model)
    st_folium(map_html, width=1300, height=550)

elif mode == "🟡 2. 3W-DBSCAN":
    model = ThreeWayDBSCAN(eps=eps_val, min_samples=min_pts, eta=0.2).fit(X_scaled)
    total_pos = sum([len(v) for v in model.POS.values()])
    total_bnd = sum([len(v) for v in model.BND.values()])
    _, _, a_star = calculate_soft_metrics(model, len(df_orders))
    
    kpi1.metric("📦 Tổng đơn", len(df_orders))
    kpi2.metric("🔴 Đơn Lõi (POS)", total_pos)
    kpi3.metric("🟡 Lân cận (BND)", total_bnd)
    kpi4.metric("⚪ Bỏ lọt / Hủy", len(df_orders) - total_pos - total_bnd)
    kpi5.metric("⭐ Độ tin cậy (α*)", f"{a_star:.3f}")
    
    st.warning("⚠️ **Đánh giá 3W-DBSCAN:** Tư duy 3 vùng rất tốt, nhưng do bán kính cố định, đơn hàng khu vực ngoại vi (Hà Đông) vẫn bị phân loại thành rác.")
    map_html = draw_dashboard_map(df_orders, mode="3W", model=model)
    st_folium(map_html, width=1300, height=550)

elif mode == "🟢 3. LE3W-DBSCAN (Đề xuất)":
    model = LE3W_DBSCAN(min_samples=min_pts, k_neighbors=k_val).fit(X_scaled)
    total_pos = sum([len(v) for v in model.POS.values()])
    total_bnd = sum([len(v) for v in model.BND.values()])
    _, _, a_star = calculate_soft_metrics(model, len(df_orders))
    
    kpi1.metric("📦 Tổng đơn", len(df_orders))
    kpi2.metric("🔴 Đơn Lõi (POS)", total_pos, "Tối ưu trạm tập kết")
    kpi3.metric("🟡 Lân cận (BND)", total_bnd, "Giao tận nhà")
    kpi4.metric("⚪ Bỏ lọt / Hủy", len(df_orders) - total_pos - total_bnd)
    kpi5.metric("⭐ Độ tin cậy (α*)", f"{a_star:.3f}", "+ Cải thiện nhất")
    
    st.success("✅ **Đánh giá LE3W-DBSCAN:** Hoàn hảo! Nhờ **Local Eps**, thuật toán khoanh vùng Lõi rất gọn ở trung tâm nhưng vẫn bắt trọn được cụm điểm ở Hà Đông. Không một Shipper nào bị quá tải!")
    map_html = draw_dashboard_map(df_orders, mode="LE3W", model=model)
    st_folium(map_html, width=1300, height=550)