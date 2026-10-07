import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="서울 기온 예측기 및 모델 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기 및 회귀 모델 평가")

# ---------------------------------------------------------
# 데이터 불러오기 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 데이터 읽기 및 datetime 변환
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df.dropna(subset=["평균기온"]).groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링 조건: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly_stats[(yearly_stats["연도"] <= 2025) & (yearly_stats["관측일수"] >= 300)].copy()
    
    # 기준 연도(1908년) 차이 계산
    filtered["경과연수"] = filtered["연도"] - 1908
    return filtered

data = load_and_process_data()

# ---------------------------------------------------------
# 1. 훈련 데이터 및 테스트 데이터 분할
# ---------------------------------------------------------
# 훈련 데이터: 최근 50년(1956~2005), 최근 100년(1906~2005)
# 공통 테스트 데이터: 최근 20년(2006~2025)
train_50 = data[(data["연도"] >= 1956) & (data["연도"] <= 2005)]
train_100 = data[(data["연도"] >= 1906) & (data["연도"] <= 2005)]
test_20 = data[(data["연도"] >= 2006) & (data["연도"] <= 2025)]

X_test = test_20[["연도"]]
y_test = test_20["연평균기온"]

# 최근 50년 모델 학습 및 예측
model_50 = LinearRegression()
model_50.fit(train_50[["연도"]], train_50["연평균기온"])
pred_50 = model_50.predict(X_test)

# 최근 100년 모델 학습 및 예측
model_100 = LinearRegression()
model_100.fit(train_100[["연도"]], train_100["연평균기온"])
pred_100 = model_100.predict(X_test)

# 성능 지표 평가 (MAE, MSE, R²)
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)

# ---------------------------------------------------------
# 2. 대시보드 요약 지표 출력
# ---------------------------------------------------------
st.subheader("📌 최근 20년(2006~2025) 공통 테스트 데이터 예측 성능 평가")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟦 최근 50년 학습 모델 (1956~2005)")
    st.metric("회귀선 기울기", f"+{model_50.coef_[0]:.4f} °C/년")
    st.metric("100년당 상승 온도", f"+{model_50.coef_[0]*100:.2f} °C")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae_50:.3f}")
    m2.metric("MSE", f"{mse_50:.3f}")
    m3.metric("R²", f"{r2_50:.3f}")

with col2:
    st.markdown("### 🟧 최근 100년 학습 모델 (1906~2005)")
    st.metric("회귀선 기울기", f"+{model_100.coef_[0]:.4f} °C/년")
    st.metric("100년당 상승 온도", f"+{model_100.coef_[0]*100:.2f} °C")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae_100:.3f}")
    m2.metric("MSE", f"{mse_100:.3f}")
    m3.metric("R²", f"{r2_100:.3f}")

st.divider()

# ---------------------------------------------------------
# 3. 예측 슬라이더 영역
# ---------------------------------------------------------
st.subheader("🔮 연도별 기온 예측 비교")
target_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)

pred_val_50 = model_50.predict([[target_year]])[0]
pred_val_100 = model_100.predict([[target_year]])[0]

pred_col1, pred_col2 = st.columns(2)
pred_col1.metric(label=f"{target_year}년 예상 기온 (50년 학습 모델)", value=f"{pred_val_50:.2f} °C")
pred_col2.metric(label=f"{target_year}년 예상 기온 (100년 학습 모델)", value=f"{pred_val_100:.2f} °C")

# ---------------------------------------------------------
# 4. Plotly 시각화 (회귀선 비교)
# ---------------------------------------------------------
line_years = np.arange(1900, 2101).reshape(-1, 1)
line_y_50 = model_50.predict(line_years)
line_y_100 = model_100.predict(line_years)

fig = go.Figure()

# 관측 산점도
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode="markers",
    name="실제 관측 연평균기온",
    marker=dict(color="royalblue", size=6, opacity=0.6)
))

# 테스트 데이터 영역 하이라이트 (최근 20년)
fig.add_trace(go.Scatter(
    x=test_20["연도"],
    y=test_20["연평균기온"],
    mode="markers",
    name="테스트 데이터 (2006~2025)",
    marker=dict(color="green", size=8, symbol="circle")
))

# 50년 모델 회귀선
fig.add_trace(go.Scatter(
    x=line_years.flatten(),
    y=line_y_50,
    mode="lines",
    name=f"최근 50년 회귀선 (+{model_50.coef_[0]*100:.2f}°C/100년)",
    line=dict(color="firebrick", width=2.5)
))

# 100년 모델 회귀선
fig.add_trace(go.Scatter(
    x=line_years.flatten(),
    y=line_y_100,
    mode="lines",
    name=f"최근 100년 회귀선 (+{model_100.coef_[0]*100:.2f}°C/100년)",
    line=dict(color="orange", width=2.5, dash="dash")
))

# 선택한 연도 예측점 표시
fig.add_trace(go.Scatter(
    x=[target_year, target_year],
    y=[pred_val_50, pred_val_100],
    mode="markers",
    name=f"선택 연도 ({target_year}년) 예측점",
    marker=dict(color="red", size=12, symbol="star")
))

fig.update_layout(
    title="서울 연평균 기온 추이 및 모델별 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
