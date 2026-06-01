# app.py - 완전히 새로운 디자인 (다크 테마, 어두운 지구 지도)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 페이지 설정 (wide + 다크 테마 CSS)
# -------------------------------
st.set_page_config(page_title="🌍 지진 위험 탐지기", layout="wide")

# 커스텀 CSS: 다크 테마, 카드 스타일, 폰트
st.markdown("""
<style>
    /* 메인 배경 다크 */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    /* 카드 스타일 */
    .metric-card {
        background-color: #1e1e2f;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
        transition: 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-label {
        font-size: 1rem;
        color: #bbbbcc;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ff9f4a;
    }
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #1a1c23;
    }
    /* 버튼 */
    .stButton > button {
        background-color: #ff6b4a;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff8c6a;
        color: black;
    }
    /* expander 스타일 */
    .streamlit-expanderHeader {
        background-color: #262730;
        border-radius: 10px;
    }
    /* 데이터프레임 테이블 */
    .dataframe {
        background-color: #1e1e2f;
        color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. 데이터 수집 (USGS API)
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_earthquake_data(days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minmagnitude": 0,
        "orderby": "time"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        earthquakes = []
        for feature in data["features"]:
            coords = feature["geometry"]["coordinates"]
            props = feature["properties"]
            earthquakes.append({
                "위도": coords[1],
                "경도": coords[0],
                "깊이_km": coords[2],
                "규모": props["mag"],
                "장소": props["place"],
                "시간": datetime.fromtimestamp(props["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(earthquakes)
        df = df.dropna(subset=["규모", "위도", "경도"])
        return df
    except Exception as e:
        st.error(f"⚠️ 데이터 수집 실패: {e}")
        return pd.DataFrame()

# -------------------------------
# 2. 클러스터링 (위험도 그룹화)
# -------------------------------
def assign_risk_clusters(df):
    if df.empty:
        return df
    
    features = df[["위도", "경도", "규모"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)
    
    cluster_mag = df.groupby("cluster")["규모"].mean().sort_values()
    high_risk_cluster = cluster_mag.idxmax()
    low_risk_cluster = cluster_mag.idxmin()
    medium_risk_cluster = cluster_mag.index[~cluster_mag.index.isin([high_risk_cluster, low_risk_cluster])][0]
    
    risk_map = {
        high_risk_cluster: "높음",
        low_risk_cluster: "낮음",
        medium_risk_cluster: "중간"
    }
    color_map = {
        "높음": "#ff4d4d",   # 빨강
        "낮음": "#4dabff",   # 파랑
        "중간": "#ffaa4d"    # 주황
    }
    
    df["위험도"] = df["cluster"].map(risk_map)
    df["색상"] = df["위험도"].map(color_map)
    return df

# -------------------------------
# 3. Folium 지도 (다크 어스 스타일)
# -------------------------------
def create_earthquake_map(df, user_lat=None, user_lon=None):
    if not df.empty and user_lat is not None and user_lon is not None:
        map_center = [user_lat, user_lon]
        zoom_start = 6
    elif not df.empty:
        map_center = [df["위도"].mean(), df["경도"].mean()]
        zoom_start = 2
    else:
        map_center = [0, 0]
        zoom_start = 2
    
    # 어두운 지구 느낌의 타일 (CartoDB dark_matter)
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB dark_matter")
    
    # 지진 표시
    sample_df = df if len(df) <= 5000 else df.sample(5000, random_state=42)
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=4,
            color=row["색상"],
            fill=True,
            fill_color=row["색상"],
            fill_opacity=0.8,
            popup=f"<b>규모 {row['규모']}</b><br>{row['장소']}<br>{row['시간']}"
        ).add_to(m)
    
    if user_lat is not None and user_lon is not None:
        folium.Marker(
            location=[user_lat, user_lon],
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
            popup="<b>📍 선택한 위치</b>"
        ).add_to(m)
    
    return m

# -------------------------------
# 4. 위치 위험도 분석
# -------------------------------
def analyze_location_risk(df, lat, lon, radius_deg=2):
    if df.empty:
        return "데이터 없음", 0, "데이터 없음"
    nearby = df[
        (df["위도"] >= lat - radius_deg) & (df["위도"] <= lat + radius_deg) &
        (df["경도"] >= lon - radius_deg) & (df["경도"] <= lon + radius_deg)
    ]
    if nearby.empty:
        return "정보 부족", 0, f"반경 {radius_deg}° 내 지진 없음"
    risk_counts = nearby["위험도"].value_counts()
    main_risk = risk_counts.idxmax()
    cnt = len(nearby)
    details = f"반경 {radius_deg}° 내 지진 {cnt}개\n위험도: {risk_counts.to_dict()}"
    return main_risk, cnt, details

# -------------------------------
# 5. 메인 앱
# -------------------------------
st.title("🌍 지진 위험 탐지기")
st.markdown("*USGS 실시간 데이터(최근 30일) × 클러스터링 기반 위험도*")

# 데이터 로드
with st.spinner("🌀 지진 데이터 수집 중..."):
    df_raw = fetch_earthquake_data()
if df_raw.empty:
    st.stop()

df = assign_risk_clusters(df_raw)

# ========== 통계 카드 (커스텀 HTML) ==========
total_eq = len(df)
max_mag = df["규모"].max()
avg_mag = df["규모"].mean()
strong_eq = df[df["규모"] >= 5.0].shape[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 총 지진 횟수</div>
        <div class="metric-value">{total_eq:,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 최대 규모</div>
        <div class="metric-value">{max_mag:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📉 평균 규모</div>
        <div class="metric-value">{avg_mag:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚠️ 강진 (M≥5)</div>
        <div class="metric-value">{strong_eq}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ========== 지도 및 분석 영역 ==========
left, right = st.columns([3, 1])

with left:
    st.subheader("🗺️ 실시간 지진 지도 (위험도별 색상)")
    # 사용자 위치 마커 체크박스
    show_marker = st.checkbox("📍 지도에 내 위치 표시", value=True, key="map_marker")
    if show_marker:
        # 기본 위치 (한반도 중심)
        user_lat = 36.5
        user_lon = 127.5
        map_obj = create_earthquake_map(df, user_lat, user_lon)
    else:
        map_obj = create_earthquake_map(df, None, None)
    folium_static(map_obj, width=1000, height=700)

with right:
    st.subheader("🔍 위치 위험도 분석")
    with st.expander("📍 좌표 입력하기", expanded=True):
        lat_input = st.number_input("위도", value=36.5, format="%.4f", key="lat")
        lon_input = st.number_input("경도", value=127.5, format="%.4f", key="lon")
        radius = st.slider("분석 반경 (도)", 0.5, 5.0, 2.0, step=0.5)
        if st.button("🔥 위험도 분석 실행", use_container_width=True):
            risk, cnt, details = analyze_location_risk(df, lat_input, lon_input, radius)
            st.success(f"### 위험도: **{risk}**")
            st.info(f"주변 {cnt}개 지진 기반")
            with st.expander("📋 상세 내역"):
                st.text(details)
    
    st.markdown("---")
    st.caption("🎨 다크 테마 | 지도: CartoDB dark_matter")

# ========== 지진 목록 (접을 수 있음) ==========
with st.expander("📋 상세 지진 목록 (최근 순)", expanded=False):
    display_df = df[["시간", "위도", "경도", "규모", "장소", "위험도"]].copy()
    display_df = display_df.sort_values("시간", ascending=False)
    # 페이지네이션
    page_size = 15
    total_pages = len(display_df) // page_size + 1
    page_num = st.number_input("페이지 번호", min_value=1, max_value=total_pages, value=1, key="page")
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True, height=400)
    st.caption(f"총 {len(display_df)}개 지진 기록 (최근 30일, USGS)")

if strong_eq > 0:
    st.warning(f"⚠️ 주의: 지난 30일 동안 규모 5.0 이상의 강진이 {strong_eq}회 발생했습니다.", icon="⚠️")
