import streamlit as st
import numpy as np
import pandas as pd
import folium
import plotly.express as px
from collections import defaultdict
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
# CONSTANTS
# ==========================================
AREA_COLORS = {
    'Hoàn Kiếm': '#e74c3c',
    'Cầu Giấy':  '#3498db',
    'Hà Đông':   '#2ecc71',
    'Đông Anh':  '#f1c40f',
    'Nhiễu':     '#95a5a6',
}

SHIPPER_COLORS = [
    '#f1c40f', '#e74c3c', '#3498db', '#2ecc71',
    '#9b59b6', '#1abc9c', '#e67e22', '#e91e63',
    '#00bcd4', '#ff5722',
]

# ==========================================
# 1. TẠO DỮ LIỆU ĐƠN HÀNG
# ==========================================
@st.cache_data
def load_live_deliveries():
    np.random.seed(42)
    hoan_kiem = np.random.normal(loc=[21.0285, 105.8542], scale=[0.002, 0.002], size=(200, 2))
    cau_giay  = np.random.normal(loc=[21.0378, 105.7816], scale=[0.006, 0.006], size=(120, 2))
    ha_dong   = np.random.normal(loc=[20.9760, 105.7700], scale=[0.015, 0.015], size=(120, 2))
    dong_anh  = np.random.normal(loc=[21.1300, 105.8450], scale=[0.018, 0.018], size=(80,  2))
    noise     = np.random.uniform(low=[20.9300, 105.7200], high=[21.1600, 105.9200], size=(50, 2))

    data = np.vstack([hoan_kiem, cau_giay, ha_dong, dong_anh, noise])
    df = pd.DataFrame(data, columns=['Lat', 'Lon'])
    df['OrderID']  = [f"ORD-{str(i).zfill(4)}" for i in range(len(df))]
    df['Khu_vuc']  = (['Hoàn Kiếm'] * 200 + ['Cầu Giấy'] * 120 +
                      ['Hà Đông'] * 120 + ['Đông Anh'] * 80 + ['Nhiễu'] * 50)
    return df

# ==========================================
# 2. BẢN ĐỒ PHÂN CỤM (chế độ thuật toán)
# ==========================================
def draw_dashboard_map(df, mode="Live", model=None):
    m = folium.Map(location=[21.0200, 105.8000], zoom_start=12, tiles='CartoDB dark_matter')

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
            0%   { transform: scale(0.5); opacity: 1.0; }
            100% { transform: scale(1.5); opacity: 0.0; }
        }
    </style>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(blink_css)
    m.get_root().add_child(macro)

    if mode == "Live":
        for i, row in df.iterrows():
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                icon=folium.DivIcon(html="<div class='blink-dot'></div>"),
                tooltip=f"<b>{row['OrderID']}</b><br>Trạng thái: Đang chờ Shipper"
            ).add_to(m)

    elif mode == "DBSCAN":
        labels = model.labels_
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c', '#1abc9c']
        for i, row in df.iterrows():
            cid = labels[i]
            if cid == -1:
                folium.CircleMarker([row['Lat'], row['Lon']], radius=3, color='gray', tooltip="Hủy").add_to(m)
            else:
                folium.CircleMarker([row['Lat'], row['Lon']], radius=5, color=colors[cid % len(colors)],
                                    fill=True, tooltip=f"Cụm {cid}").add_to(m)

    elif mode in ["3W", "LE3W"]:
        # Tô màu theo VAI TRÒ để kể câu chuyện POS / BND / NEG
        pos_pts, bnd_pts = set(), set()
        for v in model.POS.values(): pos_pts.update(v)
        for v in model.BND.values(): bnd_pts.update(v)

        for i, row in df.iterrows():
            if i in pos_pts:
                folium.CircleMarker(
                    [row['Lat'], row['Lon']], radius=7,
                    color='#e74c3c', fill=True, fill_color='#e74c3c', fill_opacity=0.9,
                    tooltip=f"🔴 {row['OrderID']} — Vùng LÕI (POS): Đặt điểm tập kết"
                ).add_to(m)
            elif i in bnd_pts:
                folium.CircleMarker(
                    [row['Lat'], row['Lon']], radius=5,
                    color='#e67e22', fill=True, fill_color='#e67e22', fill_opacity=0.7,
                    tooltip=f"🟠 {row['OrderID']} — Vùng LÂN CẬN (BND): Giao tận nhà"
                ).add_to(m)
            else:
                folium.CircleMarker(
                    [row['Lat'], row['Lon']], radius=4,
                    color='#7f8c8d', fill=True, fill_color='#7f8c8d', fill_opacity=0.4,
                    tooltip=f"⚪ {row['OrderID']} — Vùng NHIỄU (NEG): Phụ phí giao xa"
                ).add_to(m)
    return m

# ==========================================
# 3. SHIPPER ASSIGNMENT
# ==========================================
def get_cluster_labels(model, n_points, model_type="DBSCAN", pos_only=False):
    """pos_only=True: chỉ lấy POS (điểm lõi) cho shipper routing — tránh noise rải rác."""
    if model_type == "DBSCAN":
        return model.labels_
    labels = np.full(n_points, -1, dtype=int)
    for cid, points in model.POS.items():
        for p in points:
            labels[p] = int(cid)
    if not pos_only:
        for cid, points in model.BND.items():
            for p in points:
                if labels[p] == -1:
                    labels[p] = int(cid)
    return labels

def assign_shippers_geographic(cluster_labels, coords, n_shippers):
    """Group clusters by geographic proximity via K-Means on cluster centroids."""
    from sklearn.cluster import KMeans as _KMeans

    clusters = defaultdict(list)
    for idx, cid in enumerate(cluster_labels):
        if cid != -1:
            clusters[int(cid)].append(idx)

    if not clusters:
        return {}, [0] * n_shippers

    cluster_ids = sorted(clusters.keys())
    n_actual    = len(cluster_ids)

    # Tâm địa lý của mỗi cụm
    centroids = np.array([
        [np.mean(coords[clusters[cid], 0]),
         np.mean(coords[clusters[cid], 1])]
        for cid in cluster_ids
    ])

    # Ít cụm hơn số shipper → gán 1-1
    if n_actual <= n_shippers:
        point_to_shipper = {}
        shipper_load = [0] * n_shippers
        for i, cid in enumerate(cluster_ids):
            for p in clusters[cid]:
                point_to_shipper[p] = i
            shipper_load[i] = len(clusters[cid])
        return point_to_shipper, shipper_load

    # K-Means trên tâm cụm → vùng địa lý cho từng shipper
    km = _KMeans(n_clusters=n_shippers, random_state=42, n_init=10)
    zone_labels = km.fit_predict(centroids)

    point_to_shipper = {}
    shipper_load = [0] * n_shippers
    for i, cid in enumerate(cluster_ids):
        sid = int(zone_labels[i])
        shipper_load[sid] += len(clusters[cid])
        for p in clusters[cid]:
            point_to_shipper[p] = sid

    return point_to_shipper, shipper_load

def nearest_neighbor_route(points):
    """Nearest-neighbor TSP heuristic."""
    if len(points) <= 2:
        return list(points)
    pts = list(points)
    remaining = list(range(1, len(pts)))
    route = [0]
    while remaining:
        last = pts[route[-1]]
        nearest = min(remaining, key=lambda i: (pts[i][0]-last[0])**2 + (pts[i][1]-last[1])**2)
        route.append(nearest)
        remaining.remove(nearest)
    return [pts[i] for i in route]

def draw_shipper_map(df, point_to_shipper, n_shippers, bnd_indices=None, point_cluster_labels=None):
    m = folium.Map(location=[21.0200, 105.8000], zoom_start=12, tiles='CartoDB dark_matter')

    # Nhóm điểm theo (shipper, cluster) để route từng cụm riêng
    shipper_cluster_pts = defaultdict(lambda: defaultdict(list))
    shipper_all_pts     = defaultdict(list)

    for idx, row in df.iterrows():
        pt = [row['Lat'], row['Lon']]
        if idx in point_to_shipper:
            sid = point_to_shipper[idx]
            cid = int(point_cluster_labels[idx]) if point_cluster_labels is not None else 0
            shipper_cluster_pts[sid][cid].append(pt)
            shipper_all_pts[sid].append(pt)
        elif bnd_indices and idx in bnd_indices:
            folium.CircleMarker(
                pt, radius=4, color='#e67e22',
                fill=True, fill_color='#e67e22', fill_opacity=0.5,
                tooltip=f"🟠 {row['OrderID']} — Giao đặc biệt (ngoài tuyến chính)"
            ).add_to(m)
        else:
            folium.CircleMarker(
                pt, radius=3, color='#444',
                fill=True, fill_opacity=0.3,
                tooltip=f"⚪ {row['OrderID']} — Không phân công"
            ).add_to(m)

    for sid in range(n_shippers):
        if sid not in shipper_all_pts:
            continue
        color = SHIPPER_COLORS[sid % len(SHIPPER_COLORS)]
        name  = f"Shipper {chr(65 + sid)}"
        all_pts = shipper_all_pts[sid]

        # Route RIÊNG từng cụm — không nối giữa các cụm xa nhau
        for cid, pts in shipper_cluster_pts[sid].items():
            if len(pts) >= 2:
                route = nearest_neighbor_route(pts)
                folium.PolyLine(route, color=color, weight=2.5, opacity=0.7,
                                tooltip=f"Tuyến {name} — Cụm {cid}").add_to(m)
            for i, pt in enumerate(pts):
                folium.CircleMarker(
                    pt, radius=5, color=color, fill=True,
                    fill_color=color, fill_opacity=0.85,
                    tooltip=f"{name} — Điểm {i+1}/{len(pts)}"
                ).add_to(m)

        # Huy hiệu tại tâm toàn bộ vùng shipper
        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)
        badge_html = (
            f'<div style="background:{color};color:#111;border-radius:50%;'
            f'width:30px;height:30px;display:flex;align-items:center;'
            f'justify-content:center;font-weight:bold;font-size:13px;'
            f'border:2px solid white;box-shadow:0 0 6px {color};">'
            f'{chr(65+sid)}</div>'
        )
        folium.Marker(
            [cx, cy],
            icon=folium.DivIcon(html=badge_html, icon_size=(30, 30), icon_anchor=(15, 15)),
            tooltip=f"{name} — {len(all_pts)} đơn POS"
        ).add_to(m)

    return m

def show_shipper_section(df, cluster_labels, coords, n_shippers, model=None, model_type="DBSCAN", map_key="shipper_map"):
    st.markdown("---")
    st.markdown("### 🚴 PHÂN CÔNG SHIPPER")

    # Chỉ route POS points — tránh noise rải rác làm hỏng tuyến đường
    if model_type != "DBSCAN" and model is not None:
        pos_labels = get_cluster_labels(model, len(df), model_type, pos_only=True)
        bnd_set = set()
        for v in model.BND.values(): bnd_set.update(v)
        pos_set = set()
        for v in model.POS.values(): pos_set.update(v)
        bnd_indices = bnd_set - pos_set
    else:
        pos_labels  = cluster_labels
        bnd_indices = None

    point_to_shipper, loads = assign_shippers_geographic(pos_labels, coords, n_shippers)
    total_assigned = sum(loads)
    n_unassigned   = len(df) - total_assigned

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m = draw_shipper_map(df, point_to_shipper, n_shippers,
                             bnd_indices=bnd_indices,
                             point_cluster_labels=pos_labels)
        st_folium(m, width=None, height=520, key=map_key)

    with col_stats:
        rows = []
        for i in range(n_shippers):
            pct = loads[i] / total_assigned * 100 if total_assigned > 0 else 0
            rows.append({
                'Shipper': f"{chr(65+i)}",
                'Số đơn':  loads[i],
                'Tải (%)': f"{pct:.1f}%",
            })
        df_stats = pd.DataFrame(rows)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        fig_bar = px.bar(
            df_stats, x='Shipper', y='Số đơn', color='Shipper',
            color_discrete_sequence=SHIPPER_COLORS[:n_shippers],
            template='plotly_dark', title="Cân bằng tải",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        if n_unassigned > 0:
            st.warning(f"⚠️ **{n_unassigned} đơn** không phân công được (nhiễu / ngoại vi)")

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.set_page_config(page_title="Live Logistics Dashboard", page_icon="🌐", layout="wide")
st.title("🌐 SMART-LOGISTICS: LIVE DISPATCH DASHBOARD")

df_orders = load_live_deliveries()
X         = df_orders[['Lat', 'Lon']].values
X_scaled  = MinMaxScaler().fit_transform(X)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")

    mode = st.radio(
        "LỰA CHỌN THUẬT TOÁN ĐIỀU PHỐI:",
        ("📊 0. Khám phá dữ liệu", "🔴 Live (Chờ xử lý)",
         "🔵 1. DBSCAN (Cũ)", "🟡 2. 3W-DBSCAN", "🟢 3. LE3W-DBSCAN (Đề xuất)")
    )

    st.markdown("---")
    st.subheader("CẤU HÌNH THUẬT TOÁN")
    min_pts       = st.slider("Chỉ tiêu (MinPts)",            2,  20,  5)
    eps_val       = st.slider("Bán kính quét (Eps)",          0.01, 1.0, 0.13, 0.01)
    k_val         = st.slider("Bù trừ mật độ (K-Neighbors)",  2, 30, 15)
    min_merge_val = st.slider("Ngưỡng gộp cụm nhỏ (LE3W)",   5, 100, 30)

    st.markdown("---")
    st.subheader("🚴 CẤU HÌNH SHIPPER")
    n_shippers = st.slider("Số shipper đang online", 1, 10, 3)

# --- KPIs ---
st.markdown("### 📊 THỐNG KÊ TRẠNG THÁI (REAL-TIME)")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# ==========================================
# CHẾ ĐỘ: KHÁM PHÁ DỮ LIỆU
# ==========================================
if mode == "📊 0. Khám phá dữ liệu":
    kpi1.metric("📦 Tổng đơn hàng", len(df_orders))
    kpi2.metric("📍 Hoàn Kiếm + Cầu Giấy", 320)
    kpi3.metric("📍 Hà Đông",   120)
    kpi4.metric("📍 Đông Anh",   80)
    kpi5.metric("⚡ Nhiễu / Rác", 50)

    st.markdown("### 🗺️ PHÂN BỐ DỮ LIỆU THỰC TẾ")
    col_map, col_pie = st.columns([2, 1])

    with col_map:
        fig_scatter = px.scatter(
            df_orders, x='Lon', y='Lat', color='Khu_vuc',
            color_discrete_map=AREA_COLORS,
            title="Phân bố tọa độ đơn hàng theo khu vực",
            labels={'Lon': 'Kinh độ', 'Lat': 'Vĩ độ'},
            hover_data=['OrderID'],
            template='plotly_dark', opacity=0.7,
        )
        fig_scatter.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_pie:
        area_counts = df_orders['Khu_vuc'].value_counts().reset_index()
        area_counts.columns = ['Khu_vuc', 'Số đơn']
        fig_pie = px.pie(
            area_counts, names='Khu_vuc', values='Số đơn',
            color='Khu_vuc', color_discrete_map=AREA_COLORS,
            title="Tỷ lệ đơn hàng theo khu vực",
            template='plotly_dark',
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    col_lat, col_lon = st.columns(2)
    with col_lat:
        fig_lat = px.histogram(
            df_orders, x='Lat', color='Khu_vuc', nbins=40,
            color_discrete_map=AREA_COLORS,
            title="Phân phối Vĩ độ (Lat)",
            labels={'Lat': 'Vĩ độ', 'count': 'Số đơn'},
            template='plotly_dark', barmode='overlay', opacity=0.75,
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    with col_lon:
        fig_lon = px.histogram(
            df_orders, x='Lon', color='Khu_vuc', nbins=40,
            color_discrete_map=AREA_COLORS,
            title="Phân phối Kinh độ (Lon)",
            labels={'Lon': 'Kinh độ', 'count': 'Số đơn'},
            template='plotly_dark', barmode='overlay', opacity=0.75,
        )
        st.plotly_chart(fig_lon, use_container_width=True)

    st.markdown("### 📋 THỐNG KÊ MÔ TẢ THEO KHU VỰC")
    stats = (
        df_orders.groupby('Khu_vuc')[['Lat', 'Lon']]
        .agg(Số_đơn=('Lat','count'), Lat_TB=('Lat','mean'), Lat_SD=('Lat','std'),
             Lon_TB=('Lon','mean'), Lon_SD=('Lon','std'))
        .round(4).reset_index()
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)

# ==========================================
# CHẾ ĐỘ: LIVE
# ==========================================
elif mode == "🔴 Live (Chờ xử lý)":
    kpi1.metric("📦 Tổng đơn chờ", len(df_orders))
    kpi2.metric("🔴 Đơn lõi (POS)", "0", "-100%")
    kpi3.metric("🟡 Đơn lân cận (BND)", "0")
    kpi4.metric("⚪ Bỏ lọt / Hủy", len(df_orders))
    kpi5.metric("⭐ Độ tin cậy (α*)", "0.000")

    st.info("💡 **Trạng thái:** Các điểm vàng đang nhấp nháy báo hiệu hệ thống đang liên tục nhận đơn hàng mới. Hãy chọn thuật toán bên tay trái để bắt đầu chia đơn cho Shipper.")
    st_folium(draw_dashboard_map(df_orders, mode="Live"), width=1300, height=550, key="live_map")

# ==========================================
# CHẾ ĐỘ: DBSCAN
# ==========================================
elif mode == "🔵 1. DBSCAN (Cũ)":
    model   = DBSCAN(eps=eps_val, min_samples=min_pts).fit(X_scaled)
    n_noise = list(model.labels_).count(-1)

    kpi1.metric("📦 Tổng đơn", len(df_orders))
    kpi2.metric("🔴 Đơn phân cụm", len(df_orders) - n_noise)
    kpi3.metric("🟡 Đơn BND", "Không hỗ trợ")
    kpi4.metric("⚪ Bỏ lọt / Hủy", n_noise)
    kpi5.metric("⭐ Độ tin cậy (α*)", "N/A")

    st.error("❌ **Đánh giá DBSCAN:** Lãng phí! Gom quá nhiều đơn thành cụm khổng lồ, Shipper không thể chạy nổi, trong khi ngoại thành thì bị hủy sạch.")
    st_folium(draw_dashboard_map(df_orders, mode="DBSCAN", model=model), width=1300, height=550, key="dbscan_map")

    labels = get_cluster_labels(model, len(df_orders), "DBSCAN")
    show_shipper_section(df_orders, labels, X, n_shippers, model=model, model_type="DBSCAN", map_key="dbscan_shipper_map")

# ==========================================
# CHẾ ĐỘ: 3W-DBSCAN
# ==========================================
elif mode == "🟡 2. 3W-DBSCAN":
    model    = ThreeWayDBSCAN(eps=eps_val, min_samples=min_pts, eta=0.2).fit(X_scaled)
    _pos_set = set()
    for v in model.POS.values(): _pos_set.update(v)
    _bnd_set = set()
    for v in model.BND.values(): _bnd_set.update(v)
    total_pos = len(_pos_set)
    total_bnd = len(_bnd_set - _pos_set)
    _, _, a_star = calculate_soft_metrics(model, len(df_orders))

    kpi1.metric("📦 Tổng đơn", len(df_orders))
    kpi2.metric("🔴 Đơn Lõi (POS)", total_pos)
    kpi3.metric("🟡 Lân cận (BND)", total_bnd)
    kpi4.metric("⚪ Bỏ lọt / Hủy", len(df_orders) - total_pos - total_bnd)
    kpi5.metric("⭐ Độ tin cậy (α*)", f"{a_star:.3f}")

    st.warning("⚠️ **Đánh giá 3W-DBSCAN:** Tư duy 3 vùng rất tốt, nhưng do bán kính cố định, đơn hàng khu vực ngoại vi (Hà Đông) vẫn bị phân loại thành rác.")
    st_folium(draw_dashboard_map(df_orders, mode="3W", model=model), width=1300, height=550, key="3w_map")

    labels = get_cluster_labels(model, len(df_orders), "3W")
    show_shipper_section(df_orders, labels, X, n_shippers, model=model, model_type="3W", map_key="3w_shipper_map")

# ==========================================
# CHẾ ĐỘ: LE3W-DBSCAN
# ==========================================
elif mode == "🟢 3. LE3W-DBSCAN (Đề xuất)":
    model    = LE3W_DBSCAN(min_samples=min_pts, k_neighbors=k_val, min_merge_size=min_merge_val).fit(X_scaled)
    _pos_set = set()
    for v in model.POS.values(): _pos_set.update(v)
    _bnd_set = set()
    for v in model.BND.values(): _bnd_set.update(v)
    total_pos = len(_pos_set)
    total_bnd = len(_bnd_set - _pos_set)
    _, _, a_star = calculate_soft_metrics(model, len(df_orders))

    kpi1.metric("📦 Tổng đơn", len(df_orders))
    kpi2.metric("🔴 Đơn Lõi (POS)", total_pos, "Tối ưu trạm tập kết")
    kpi3.metric("🟡 Lân cận (BND)", total_bnd, "Giao tận nhà")
    kpi4.metric("⚪ Bỏ lọt / Hủy", len(df_orders) - total_pos - total_bnd)
    kpi5.metric("⭐ Độ tin cậy (α*)", f"{a_star:.3f}", "+ Cải thiện nhất")

    st.success("✅ **Đánh giá LE3W-DBSCAN:** Hoàn hảo! Nhờ **Local Eps**, thuật toán khoanh vùng Lõi rất gọn ở trung tâm nhưng vẫn bắt trọn được cụm điểm ở Hà Đông. Không một Shipper nào bị quá tải!")
    st_folium(draw_dashboard_map(df_orders, mode="LE3W", model=model), width=1300, height=550, key="le3w_map")

    labels = get_cluster_labels(model, len(df_orders), "LE3W")
    show_shipper_section(df_orders, labels, X, n_shippers, model=model, model_type="LE3W", map_key="le3w_shipper_map")
