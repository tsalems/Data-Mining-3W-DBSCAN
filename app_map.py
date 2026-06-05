import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

from models.threeway_dbscan import ThreeWayDBSCAN
from models.le3w_dbscan import LE3W_DBSCAN
from utils.metrics import calculate_soft_metrics

# ==========================================
# 1. QUẢN LÝ TRẠNG THÁI (STATE MANAGEMENT)
# ==========================================
# Khởi tạo bộ lọc bản đồ mặc định là hiển thị "Tất cả"
if 'map_filter' not in st.session_state:
    st.session_state.map_filter = "All"

# Hàm reset bộ lọc khi đổi thuật toán
def reset_filter():
    st.session_state.map_filter = "All"

# ==========================================
# 2. TẠO DỮ LIỆU ĐƠN HÀNG (CÓ BỘ LỌC ĐỊA HÌNH HỒ NƯỚC)
# ==========================================
@st.cache_data
def load_live_deliveries():
    np.random.seed(42)
    
    # Kỹ thuật Rejection Sampling: Lọc các đơn hàng rơi xuống nước
    def is_in_water(lat, lon):
        # 1. Chặn dải Sông Hồng (Phía Đông)
        if lon > 105.860 + (lat - 21.000) * 0.4: return True 
        # 2. Vùng Hồ Tây & Trúc Bạch (Khu vực mặt nước cực lớn)
        if 21.040 < lat < 21.075 and 105.810 < lon < 105.845: return True
        # 3. Hồ Hoàn Kiếm (Gươm)
        if 21.025 < lat < 21.031 and 105.850 < lon < 105.854: return True
        # 4. Hồ Bảy Mẫu & Thiền Quang (Công viên Thống Nhất)
        if 21.011 < lat < 21.019 and 105.840 < lon < 105.847: return True
        # 5. Hồ Đống Đa (Hoàng Cầu)
        if 21.013 < lat < 21.019 and 105.818 < lon < 105.823: return True
        # 6. Hồ Thành Công & Giảng Võ
        if 21.020 < lat < 21.032 and 105.812 < lon < 105.818: return True
        return False

    # Hàm sinh điểm an toàn (Nếu rơi xuống nước -> Xóa sinh lại)
    def get_safe_normal(loc, scale, size):
        pts = []
        while len(pts) < size:
            p = np.random.normal(loc=loc, scale=scale)
            if not is_in_water(p[0], p[1]): pts.append(p)
        return np.array(pts)

    def get_safe_uniform(low, high, size):
        pts = []
        while len(pts) < size:
            p = np.random.uniform(low=low, high=high)
            if not is_in_water(p[0], p[1]): pts.append(p)
        return np.array(pts)

    # ================= CHỌN TỌA ĐỘ ĐIỂM NÓNG THỰC TẾ =================
    
    # 1. Lõi Hoàn Kiếm: Tòa nhà Pacific Place (83B Lý Thường Kiệt) - Rất đặc
    hk_core = get_safe_normal(loc=[21.0245, 105.8465], scale=[0.00015, 0.00015], size=120)
    hk_around = get_safe_normal(loc=[21.0245, 105.8465], scale=[0.0015, 0.0015], size=80)
    
    # 2. Lõi Cầu Giấy/Nam Từ Liêm: Keangnam Landmark 72 - Mật độ vừa
    cg_core = get_safe_normal(loc=[21.0168, 105.7838], scale=[0.0002, 0.0002], size=90)
    cg_around = get_safe_normal(loc=[21.0168, 105.7838], scale=[0.0025, 0.0025], size=60)
    
    # 3. Lõi Hà Đông: Khu đô thị Mulberry Lane (Mỗ Lao) - Mật độ thưa
    hd_core = get_safe_normal(loc=[20.9785, 105.7865], scale=[0.0008, 0.0008], size=30)
    hd_around = get_safe_normal(loc=[20.9785, 105.7865], scale=[0.0035, 0.0035], size=20)
    
    # 4. Đơn nhiễu (Rải rác khắp Hà Nội nhưng tuyệt đối né các hồ nước)
    noise = get_safe_uniform(low=[20.9500, 105.7400], high=[21.0450, 105.8550], size=30)
    
    # Tổng hợp dữ liệu
    data = np.vstack([hk_core, hk_around, cg_core, cg_around, hd_core, hd_around, noise])
    df = pd.DataFrame(data, columns=['Lat', 'Lon'])
    df['OrderID'] = [f"ORD-{str(i).zfill(4)}" for i in range(len(df))]
    return df

# ==========================================
# 3. HÀM VẼ BẢN ĐỒ VỚI BỘ LỌC TƯƠNG TÁC (PHÂN CHIA SHIPPER)
# ==========================================
def draw_dashboard_map(df, mode="Live", model=None, current_filter="All"):
    m = folium.Map(location=[21.0150, 105.8100], zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles='http://mt0.google.com/vt/lyrs=m&hl=vi&x={x}&y={y}&z={z}',
        attr='Google Maps', name='Google Maps Standard', max_zoom=20
    ).add_to(m)
    
    # Bảng màu Neon rực rỡ đại diện cho từng Shipper / Tuyến giao hàng
    shipper_colors = ['#00e5ff', '#d500f9', '#39ff14', '#ff9100', '#ff1744', '#ffff00', '#18ffff']

    if mode == "Live":
        for i, row in df.iterrows():
            # Màn hình Live: Đỏ rực, chưa phân tuyến
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']], radius=5, 
                color='#8b0000', fill=True, fill_color='#ff0000', fill_opacity=1.0, 
                tooltip=f"<b>{row['OrderID']}</b><br>Trạng thái: Đang chờ điều phối"
            ).add_to(m)
            
    elif mode == "DBSCAN":
        labels = model.labels_
        for i, row in df.iterrows():
            cluster_id = labels[i]
            is_noise = (cluster_id == -1)
            
            if current_filter == "Clustered" and is_noise: continue
            if current_filter == "Noise" and not is_noise: continue
            
            if is_noise:
                folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=4, color='#616161', fill=True, fill_color='#9e9e9e', fill_opacity=0.6, tooltip="Hủy").add_to(m)
            else:
                c_color = shipper_colors[cluster_id % len(shipper_colors)]
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=6, 
                    color=c_color, fill=True, fill_color=c_color, fill_opacity=0.9, 
                    tooltip=f"<b>{row['OrderID']}</b><br>Phụ trách: <b>Shipper tuyến {cluster_id}</b>"
                ).add_to(m)
                
    elif mode in ["3W", "LE3W"]:
        # Xây dựng từ điển (Dictionary) gán nhãn Shipper cho từng đơn hàng
        point_info = {}
        for cid, pts in model.POS.items():
            for p in pts:
                point_info[p] = {"shipper_id": cid, "zone": "POS"}
                
        for cid, pts in model.BND.items():
            for p in pts:
                # Nếu điểm chưa được gán, hoặc ưu tiên gán vào vùng biên của Shipper mới quét qua
                if p not in point_info:
                    point_info[p] = {"shipper_id": cid, "zone": "BND"}
                    
        for i, row in df.iterrows():
            info = point_info.get(i, {"shipper_id": -1, "zone": "NEG"})
            
            # CƠ CHẾ LỌC THEO NÚT BẤM
            if current_filter == "POS" and info["zone"] != "POS": continue
            if current_filter == "BND" and info["zone"] != "BND": continue
            if current_filter == "NEG" and info["zone"] != "NEG": continue
            
            if info["zone"] == "POS":
                # ĐƠN LÕI: Chấm to (radius=7), màu đặc (100%), có viền trắng sắc nét
                c_color = shipper_colors[info["shipper_id"] % len(shipper_colors)]
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=7, 
                    color='#ffffff', weight=1.5, # Viền trắng
                    fill=True, fill_color=c_color, fill_opacity=1.0, 
                    tooltip=f"<b>{row['OrderID']}</b><br>Phụ trách: <b>Shipper tuyến {info['shipper_id']}</b><br>Nhiệm vụ: 🔴 Giao tập kết"
                ).add_to(m)
                
            elif info["zone"] == "BND":
                # ĐƠN LÂN CẬN: Chấm nhỏ (radius=4), màu mờ hơn (45%), viền chìm
                c_color = shipper_colors[info["shipper_id"] % len(shipper_colors)]
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=4, 
                    color=c_color, weight=1, 
                    fill=True, fill_color=c_color, fill_opacity=0.45, 
                    tooltip=f"<b>{row['OrderID']}</b><br>Phụ trách: <b>Shipper tuyến {info['shipper_id']}</b><br>Nhiệm vụ: 🟠 Giao tận nhà"
                ).add_to(m)
                
            else:
                # ĐƠN HỦY/RÁC
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=3, 
                    color='#424242', weight=1, 
                    fill=True, fill_color='#757575', fill_opacity=0.5, 
                    tooltip=f"<b>{row['OrderID']}</b><br>Nhiệm vụ: ⚪ Hủy / Đợi ghép chuyến"
                ).add_to(m)
    return m

# ==========================================
# 4. GIAO DIỆN WEB (UI)
# ==========================================
st.set_page_config(page_title="Live Logistics Dashboard", page_icon="🌐", layout="wide")
st.title("🌐 SMART-LOGISTICS: LIVE DISPATCH DASHBOARD")

df_orders = load_live_deliveries()
X = df_orders[['Lat', 'Lon']].values
X_scaled = MinMaxScaler().fit_transform(X)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ BẢNG ĐIỀU KHIỂN LOGISTICS")
    
    # ... (Phần st.radio chọn thuật toán giữ nguyên) ...
    mode = st.radio(
        "LỰA CHỌN THUẬT TOÁN ĐIỀU PHỐI:",
        ("🔴 Live (Chờ xử lý)", "🔵 1. DBSCAN (Cũ)", "🟡 2. 3W-DBSCAN", "🟢 3. LE3W-DBSCAN (Đề xuất)"),
        on_change=reset_filter
    )
    
    st.markdown("---")
    st.subheader("⚙️ CẤU HÌNH NGHIỆP VỤ")
    
    # 1. Tham số MinPts (Đơn vị: Đơn hàng)
    min_pts = st.slider(
        "1. Chỉ tiêu ghép chuyến (MinPts)", 
        min_value=2, max_value=20, value=5,
        format="%d Đơn", # Hiển thị chữ "Đơn" trên thanh trượt
        help="Đơn vị: Đơn hàng. Số đơn tối thiểu cần thiết để tạo thành một 'Trạm tập kết' (Vùng Đỏ)."
    )
    
    # 2. Tham số Eps (Giao diện hiển thị Mét thực tế)
    eps_meter = st.slider(
        "2. Bán kính gom đơn cứng (Epsilon)", 
        min_value=50, max_value=2000, value=500, step=50,
        format="%d mét", # Tự động nối chữ "mét" vào đuôi con số khi kéo thả
        help="Đơn vị: Mét. Bán kính vật lý tối đa Shipper di chuyển từ tâm cụm để gom đơn (Dùng cho DBSCAN và 3W-DBSCAN gốc)."
    )
    
    # --- BƯỚC MAP DỮ LIỆU BÍ MẬT ---
    # Chuyển đổi số mét ngoài đời thực về tỷ lệ chuẩn hóa [0, 1] của thuật toán.
    # Khung bản đồ ta sinh dữ liệu rộng khoảng 14km (14.000 mét) tương đương 1 đơn vị scale.
    eps_val = eps_meter / 14000.0
    
    # 3. Tham số k (Đơn vị: Đơn hàng)
    k_val = st.slider(
        "3. Tầm nhìn bù trừ mật độ (k-Neighbors)", 
        min_value=2, max_value=30, value=15,
        format="%d Đơn", # Hiển thị chữ "Đơn" trên thanh trượt
        help="Đơn vị: Đơn hàng tham chiếu. Thuật toán LE3W sẽ đo khoảng cách tới 'đơn hàng thứ k' để tự động nới rộng/thu hẹp bán kính khu vực."
    )
    
    st.markdown("---")
    st.info("💡 **Mẹo:** Thuật toán LE3W-DBSCAN chỉ dùng tham số số (1) và (3) để tự động hóa bán kính, loại bỏ hoàn toàn sự cứng nhắc của bán kính mét (2).")

# --- KHỐI THỐNG KÊ (KPIs CÓ NÚT BẤM) ---
st.markdown(f"### 📊 BÁO CÁO ĐIỀU PHỐI: TÙY CHỈNH LỌC (Đang xem: **{st.session_state.map_filter}**)")
st.info("💡 **Hướng dẫn:** Bấm vào các nút bên dưới mỗi chỉ số để xem riêng lẻ từng cụm đơn hàng trên bản đồ.")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

if mode == "🔴 Live (Chờ xử lý)":
    # Giả lập số liệu vận hành thực tế tại thời điểm giữa ngày (Mid-day)
    # Tự động tính toán theo tỷ lệ để đảm bảo tính logic tuyệt đối
    current_pending = len(df_orders)
    sim_delivered = int(current_pending * 1.5)    # Số đơn đã giao thành công (Nhiều nhất)
    sim_dispatched = int(current_pending * 0.8)   # Số đơn Shipper đang cầm đi trên đường
    sim_canceled = int(current_pending * 0.12)    # Số đơn khách hủy hoặc bom hàng
    
    # Tổng đơn trong ngày = Đang chờ + Đã phân tuyến + Đã giao + Đã hủy
    total_orders_day = current_pending + sim_delivered + sim_dispatched + sim_canceled
    
    # Hiển thị lên Dashboard
    kpi1.metric("📦 Tổng đơn hôm nay", f"{total_orders_day:,}")
    kpi2.metric("⏳ Đang chờ (Pending)", f"{current_pending:,}")
    kpi3.metric("🔄 Đang giao (Dispatched)", f"{sim_dispatched:,}")
    kpi4.metric("✅ Đã giao (Delivered)", f"{sim_delivered:,}")
    kpi5.metric("❌ Hủy/Bom hàng", f"{sim_canceled:,}")
    
    st.info("💡 **Trạng thái:** Hệ thống đang liên tục nhận đơn hàng mới. Hãy chọn thuật toán bên tay trái để bắt đầu phân tuyến và điều phối Shipper cho các đơn đang chờ.")
    map_html = draw_dashboard_map(df_orders, mode="Live")
    folium_static(map_html, width=1300, height=550)

elif mode == "🔵 1. DBSCAN (Cũ)":
    model = DBSCAN(eps=eps_val, min_samples=min_pts).fit(X_scaled)
    n_noise = list(model.labels_).count(-1)
    n_clustered = len(df_orders) - n_noise
    
    with kpi1:
        st.metric("📦 Tổng đơn", len(df_orders))
        if st.button("👁️ Xem Tất cả", use_container_width=True, key="db_all"): st.session_state.map_filter = "All"
    with kpi2:
        # Tên gọi chính xác cho Hard Clustering
        st.metric("🔴 Đơn phải giao", n_clustered, help="Các đơn được gộp cứng nhắc thành cụm lớn, Shipper phải tự tìm lộ trình giao toàn bộ.")
        if st.button("👁️ Xem Đơn giao", use_container_width=True, key="db_cluster"): st.session_state.map_filter = "Clustered"
    with kpi3:
        st.metric("⚪ Đơn quá xa/Hủy", n_noise, help="Hành động: Bị hệ thống tự động loại bỏ do bán kính gom đơn không chạm tới.")
        if st.button("👁️ Xem Đơn hủy", use_container_width=True, key="db_noise"): st.session_state.map_filter = "Noise"
        
    map_html = draw_dashboard_map(df_orders, mode="DBSCAN", model=model, current_filter=st.session_state.map_filter)
    folium_static(map_html, width=1300, height=550)

elif mode in ["🟡 2. 3W-DBSCAN", "🟢 3. LE3W-DBSCAN (Đề xuất)"]:
    if "3W-DBSCAN" in mode and "LE3W" not in mode:
        model = ThreeWayDBSCAN(eps=eps_val, min_samples=min_pts, eta=0.2).fit(X_scaled)
        m_type = "3W"
    else:
        model = LE3W_DBSCAN(min_samples=min_pts, k_neighbors=k_val).fit(X_scaled)
        m_type = "LE3W"
        
    unique_pos, unique_bnd = set(), set()
    for v in model.POS.values(): unique_pos.update(v)
    for v in model.BND.values(): unique_bnd.update(v)
    
    total_pos = len(unique_pos)
    total_bnd = len(unique_bnd)
    total_neg = len(df_orders) - total_pos - total_bnd
    _, _, a_star = calculate_soft_metrics(model, len(df_orders))
    
    with kpi1:
        st.metric("📦 Tổng đơn", len(df_orders))
        if st.button("👁️ Xem Tất cả", use_container_width=True, key=f"{m_type}_all"): st.session_state.map_filter = "All"
    with kpi2:
        st.metric("🔴 Đơn lõi POS (giao tập kết)", total_pos, help="Mật độ cực đặc. Shipper đỗ xe 1 chỗ dưới sảnh và phát hàng.")
        if st.button("👁️ Chỉ xem POS", use_container_width=True, key=f"{m_type}_pos"): st.session_state.map_filter = "POS"
    with kpi3:
        st.metric("🟠 Đơn lân cận BND (giao tận cửa)", total_bnd, help="Mật độ vừa. Lộ trình xe máy chạy vòng quanh khu vực.")
        if st.button("👁️ Chỉ xem BND", use_container_width=True, key=f"{m_type}_bnd"): st.session_state.map_filter = "BND"
    with kpi4:
        st.metric("⚪ Đơn quá xa/rác NEG", total_neg, help="Hành động: Đợi gom thêm đơn / Tăng phụ phí / Hủy chuyến do khoảng cách xa.")
        if st.button("👁️ Chỉ xem NEG", use_container_width=True, key=f"{m_type}_neg"): st.session_state.map_filter = "NEG"
    with kpi5:
        st.metric("⭐ Độ tin cậy (α*)", f"{a_star:.3f}")

    map_html = draw_dashboard_map(df_orders, mode=m_type, model=model, current_filter=st.session_state.map_filter)
    folium_static(map_html, width=1300, height=550)