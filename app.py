import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. 데이터 로드 및 모델 학습 (캐싱)
# -------------------------------
@st.cache_resource
def load_and_train():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    target_names = wine.target_names
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 스케일링 (RandomForest에는 필요 없지만, 일관성을 위해)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 랜덤 포레스트 분류기
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 정확도 평가
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return clf, scaler, X, y, target_names, acc, cm, wine.feature_names

# 모델 로드
clf, scaler, X_full, y_full, target_names, accuracy, conf_matrix, feature_names = load_and_train()

# -------------------------------
# 2. Streamlit UI
# -------------------------------
st.set_page_config(page_title="와인 품종 예측", layout="wide")
st.title("🍷 와인 품종 예측 대시보드")
st.markdown("화학적 특성을 기반으로 **3가지 와인 품종** 중 하나를 예측합니다.")

# 사이드바: 모델 성능 요약
st.sidebar.header("📈 모델 성능")
st.sidebar.metric("테스트 정확도", f"{accuracy:.2%}")
st.sidebar.subheader("혼동 행렬")
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
ax_cm.set_xlabel("예측")
ax_cm.set_ylabel("실제")
st.sidebar.pyplot(fig_cm)

# 메인 화면: 입력 슬라이더 (13개 특성)
st.subheader("🔬 와인의 화학적 특성 입력")
cols = st.columns(2)
input_values = []
for i, name in enumerate(feature_names):
    col = cols[i % 2]
    # 각 특성의 min/max를 데이터에서 가져와 슬라이더 범위 설정
    min_val = float(X_full[name].min())
    max_val = float(X_full[name].max())
    mean_val = float(X_full[name].mean())
    val = col.slider(
        name, 
        min_value=min_val, 
        max_value=max_val, 
        value=mean_val,
        step=(max_val - min_val) / 100,
        help=f"범위: [{min_val:.2f}, {max_val:.2f}]"
    )
    input_values.append(val)

# 예측 버튼
if st.button("🍾 와인 품종 예측하기", type="primary"):
    # 입력 배열 만들기
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = clf.predict(input_scaled)[0]
    proba = clf.predict_proba(input_scaled)[0]
    
    # 결과 표시
    st.subheader("📋 예측 결과")
    col1, col2 = st.columns(2)
    col1.success(f"**예측된 품종:** {target_names[prediction]}")
    col2.info(f"**신뢰도:** {max(proba)*100:.1f}%")
    
    # 확률 막대 그래프
    st.subheader("품종별 확률 분포")
    prob_df = pd.DataFrame({
        "품종": target_names,
        "확률": proba
    })
    fig, ax = plt.subplots()
    sns.barplot(data=prob_df, x="품종", y="확률", palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("확률")
    for i, p in enumerate(proba):
        ax.text(i, p + 0.02, f"{p:.2%}", ha='center')
    st.pyplot(fig)

# -------------------------------
# 3. 특성 중요도 시각화 (Random Forest)
# -------------------------------
st.subheader("🌟 특성 중요도 (Feature Importance)")
importance = clf.feature_importances_
feat_imp_df = pd.DataFrame({"특성": feature_names, "중요도": importance}).sort_values("중요도", ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="중요도", y="특성", palette="rocket", ax=ax2)
ax2.set_title("Random Forest 특성 중요도")
st.pyplot(fig2)

# -------------------------------
# 4. 원본 데이터 미리보기
# -------------------------------
with st.expander("📊 원본 데이터셋 일부 보기 (UCI Wine Dataset)"):
    wine_data = pd.DataFrame(X_full, columns=feature_names)
    wine_data['target'] = y_full
    wine_data['target_name'] = wine_data['target'].map(lambda i: target_names[i])
    st.dataframe(wine_data.head(100), use_container_width=True)
    st.caption(f"총 {len(wine_data)}개 샘플, 13개 특성, 3개 클래스")

st.markdown("---")
st.caption("데이터 출처: UCI Machine Learning Repository - Wine Dataset")
