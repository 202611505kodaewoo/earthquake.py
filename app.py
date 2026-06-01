import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    earthquakes = []
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]
        props = feature["properties"]
        earthquakes.append({
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2],
            "magnitude": props["mag"],
            "place": props["place"],
            "time": datetime.fromtimestamp(props["time"] / 1000)
        })
    
    df = pd.DataFrame(earthquakes)
    df = df.dropna(subset=["magnitude", "latitude", "longitude"])
    df = df[df["magnitude"] > 0]  # 유효한 규모만
    return df

# -------------------------------
# 2. 특성 엔지니어링 (모델 학습용)
# -------------------------------
def engineer_features(df):
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["lon_sin"] = np.sin(np.radians(df["longitude"]))
    df["lon_cos"] = np.cos(np.radians(df["longitude"]))
    df["lat_sin"] = np.sin(np.radians(df["latitude"]))
    df["lat_cos"] = np.cos(np.radians(df["latitude"]))
    return df

# -------------------------------
# 3. 모델 학습 (캐싱)
# -------------------------------
@st.cache_resource
def train_model(df):
    df_feat = engineer_features(df)
    feature_cols = ["depth_km", "hour", "day_of_year", "month", "day_of_week",
                    "lon_sin", "lon_cos", "lat_sin", "lat_cos"]
    X = df_feat[feature_cols]
    y = df_feat["magnitude"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, feature_cols, mae, r2

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="지진 규모 예측 대시보드", layout="wide")
st.title("🔮 지진 규모 예측 시스템")
st.markdown("**USGS 실시간 데이터 기반** - 기계학습으로 지진 규모를 예측하고 시계열 패턴을 분석합니다.")

# 데이터 로드
with st.spinner("최근 30일 지진 데이터를 불러오는 중..."):
    df = fetch_earthquake_data(days=30)

if df.empty:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# 모델 학습
with st.spinner("예측 모델 학습 중..."):
    model, scaler, feature_cols, mae, r2 = train_model(df)

# -------------------------------
# 사이드바: 모델 성능
# -------------------------------
st.sidebar.header("📊 모델 성능 지표")
st.sidebar.metric("평균 절대 오차 (MAE)", f"{mae:.3f}")
st.sidebar.metric("R² 결정계수", f"{r2:.3f}")
st.sidebar.markdown("---")
st.sidebar.info("이 모델은 과거 30일 데이터로 학습하여, 깊이, 시간, 위치 정보로 지진 규모를 예측합니다.")

# -------------------------------
# 메인 영역: 예측 + 시각화
# -------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ 예측 입력값")
    depth = st.slider("깊이 (km)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
    hour = st.slider("시간 (0~23시)", min_value=0, max_value=23, value=12)
    day_of_year = st.slider("일년 중 날짜 (1~365)", min_value=1, max_value=365, value=180)
    month = st.slider("월", min_value=1, max_value=12, value=6)
    day_of_week = st.slider("요일 (0=월, 6=일)", min_value=0, max_value=6, value=2)
    lon = st.slider("경도", min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
    lat = st.slider("위도", min_value=-90.0, max_value=90.0, value=0.0, step=0.1)
    
    # 계산된 특성
    lon_sin = np.sin(np.radians(lon))
    lon_cos = np.cos(np.radians(lon))
    lat_sin = np.sin(np.radians(lat))
    lat_cos = np.cos(np.radians(lat))
    
    input_array = np.array([[depth, hour, day_of_year, month, day_of_week,
                             lon_sin, lon_cos, lat_sin, lat_cos]])
    input_scaled = scaler.transform(input_array)
    pred_mag = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.metric("예측된 지진 규모", f"{pred_mag:.2f}")
    if pred_mag >= 5.0:
        st.error("⚠️ 강진 가능성이 높습니다.")
    elif pred_mag >= 3.0:
        st.warning("⚠️ 중간 규모 예측")
    else:
        st.success("미약한 지진으로 예상됨")

with col_right:
    st.subheader("📈 최근 30일 지진 추세")
    df_trend = df.set_index("time").resample("D").size().reset_index()
    df_trend.columns = ["날짜", "지진 발생 횟수"]
    fig1 = px.line(df_trend, x="날짜", y="지진 발생 횟수", markers=True,
                   title="일별 지진 발생 빈도")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("📉 규모-깊이 관계")
    fig2 = px.scatter(df, x="depth_km", y="magnitude", opacity=0.6,
                      labels={"depth_km": "깊이 (km)", "magnitude": "규모"},
                      title="깊이에 따른 규모 분포", trendline="lowess")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# 특성 중요도 시각화
# -------------------------------
st.subheader("🧠 모델 특성 중요도")
importance = model.feature_importances_
feat_imp = pd.DataFrame({"특성": feature_cols, "중요도": importance}).sort_values("중요도", ascending=False)
fig3 = px.bar(feat_imp, x="중요도", y="특성", orientation='h', title="Random Forest 특성 중요도")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# 원본 데이터 테이블 (축소)
# -------------------------------
with st.expander("📋 원본 지진 데이터 (최근 30일)"):
    display_df = df[["time", "latitude", "longitude", "depth_km", "magnitude", "place"]].copy()
    display_df = display_df.sort_values("time", ascending=False)
    st.dataframe(display_df.head(100), use_container_width=True)
    st.caption(f"총 {len(display_df)}개 지진 기록")

st.markdown("---")
st.caption("데이터 출처: USGS Earthquake Catalog | 모델: Random Forest Regressor")
