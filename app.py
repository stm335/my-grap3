import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 불러오기 및 전처리 함수
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 데이터 읽기
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime 형식으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산 (평균기온 결측치 제외)
    yearly_stats = df.dropna(subset=["평균기온"]).groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링 조건: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly_stats[(yearly_stats["연도"] <= 2025) & (yearly_stats["관측일수"] >= 300)].copy()
    
    # 회귀 분석용 독립변수 X: 1908년 기준 경과 연수
    filtered["경과연수"] = filtered["연도"] - 1908
    
    return filtered

# 데이터 로드
data = load_and_process_data()

# ---------------------------------------------------------
# 1. 전체 기간 선형 회귀 계산
# ---------------------------------------------------------
X_all = data["경과연수"].values
y_all = data["연평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, y_all, 1)

corr_all = np.corrcoef(data["연도"], data["연평균기온"])[0, 1]
num_years_all = len(data)
start_year_all = int(data["연도"].min())
end_year_all = int(data["연도"].max())

# 100년당 상승 온도로 변환 (전체 기간)
slope_100y_all = slope_all * 100


# ---------------------------------------------------------
# 2. 최근 20년 선형 회귀 계산
# ---------------------------------------------------------
max_year = data["연도"].max()
data_recent20 = data[data["연도"] >= (max_year - 19)].copy()

X_recent = data_recent20["경과연수"].values
y_recent = data_recent20["연평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)

corr_recent = np.corrcoef(data_recent20["연도"], data_recent20["연평균기온"])[0, 1]
start_year_recent = int(data_recent20["연도"].min())
end_year_recent = int(data_recent20["연도"].max())

# 100년당 상승 온도로 변환 (최근 20년)
slope_100y_recent = slope_recent * 100


# ---------------------------------------------------------
# UI: 회귀 모델 요약 및 100년당 상승 온도 비교
# ---------------------------------------------------------
st.subheader("📌 데이터 개요 및 100년당 기온 변화량 비교")

col_summary1, col_summary2 = st.columns(2)

with col_summary1:
    st.markdown("### 🌐 전체 기간 모델")
    m1, m2, m3 = st.columns(3)
    m1.metric("대상 연도 수", f"{num_years_all}개")
    m2.metric("기간", f"{start_year_all}~{end_year_all}")
    m3.metric("상관계수 (r)", f"{corr_all:.4f}")
    
    # 100년당 오르는 온도 크게 표시
    st.metric(
        label="🔥 전체 기간 기준 100년당 상승 온도", 
        value=f"+{slope_100y_all:.2f} °C"
    )

with col_summary2:
    st.markdown("### 🚀 최근 20년 모델")
    m1, m2, m3 = st.columns(3)
    m1.metric("대상 연도 수", f"{len(data_recent20)}개")
    m2.metric("기간", f"{start_year_recent}~{end_year_recent}")
    m3.metric("상관계수 (r)", f"{corr_recent:.4f}")
    
    # 최근 20년 추세의 100년 환산 상승 온도 크게 표시 (전체 기간 대비 delta 포함)
    diff = slope_100y_recent - slope_100y_all
    st.metric(
        label="🔥 최근 20년 기준 100년당 상승 온도 (가속도)", 
        value=f"+{slope_100y_recent:.2f} °C",
        delta=f"전체 평균 대비 {diff:+.2f} °C 빠르게 상승 중"
    )

st.divider()

# ---------------------------------------------------------
# UI: 예측 슬라이더 영역 (전체 기간 기준 모델 사용)
# ---------------------------------------------------------
st.subheader("🔮 연도별 예상 기온 예측 (전체 기간 회귀선 기준)")
target_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)

# 선택한 연도의 예측 기온 계산
predicted_temp_all = slope_all * (target_year - 1908) + intercept_all
predicted_temp_recent = slope_recent * (target_year - 1908) + intercept_recent

pred_col1, pred_col2 = st.columns(2)
pred_col1.metric(label=f"{target_year}년 예상 연평균 기온 (전체 기간 추세)", value=f"{predicted_temp_all:.2f} °C")
pred_col2.metric(label=f"{target_year}년 예상 연평균 기온 (최근 20년 추세)", value=f"{predicted_temp_recent:.2f} °C")

# ---------------------------------------------------------
# UI: Plotly 시각화 (두 회귀선 비교)
# ---------------------------------------------------------
line_years = np.arange(1900, 2101)
line_x = line_years - 1908

# 회귀선 데이터 계산
line_y_all = slope_all * line_x + intercept_all
line_y_recent = slope_recent * line_x + intercept_recent

fig = go.Figure()

# 1. 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode="markers",
    name="관측 연평균기온",
    marker=dict(color="royalblue", size=6, opacity=0.7)
))

# 2. 전체 기간 회귀 직선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y_all,
    mode="lines",
    name=f"전체 기간 회귀선 (+{slope_100y_all:.2f}°C/100년)",
    line=dict(color="firebrick", width=2)
))

# 3. 최근 20년 회귀 직선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y_recent,
    mode="lines",
    name=f"최근 20년 회귀선 (+{slope_100y_recent:.2f}°C/100년)",
    line=dict(color="darkorange", width=2, dash="dash")
))

# 4. 선택한 연도의 예측점 표시 (전체 기간 모델 기준)
fig.add_trace(go.Scatter(
    x=[target_year],
    y=[predicted_temp_all],
    mode="markers",
    name=f"선택한 연도 ({target_year}년)",
    marker=dict(color="red", size=14, symbol="star")
))

fig.update_layout(
    title="서울 연평균 기온 추이 및 회귀선 비교 (전체 기간 vs 최근 20년)",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 1906~2025 서울 연평균 기온 데이터 생성 (기상청 기후통계 기준 경향 반영)
np.random.seed(42)
years = np.arange(1906, 2026)

# 비선형적인 기온 상승 트렌드 모델링 (1950년 이후 가속화)
base_temp = 10.8
trend = 0.012 * (years - 1906) + 0.00018 * np.maximum(0, years - 1950)**2
noise = np.random.normal(0, 0.55, len(years))
temp_data = base_temp + trend + noise

df = pd.DataFrame({'Year': years, 'Temp': temp_data})

# 2. 데이터셋 분할
train_50 = df[(df['Year'] >= 1956) & (df['Year'] <= 2005)]
train_100 = df[(df['Year'] >= 1906) & (df['Year'] <= 2005)]
test_20 = df[(df['Year'] >= 2006) & (df['Year'] <= 2025)]

X_test = test_20[['Year']]
y_test = test_20['Temp']

# 3. 모델 학습 및 예측
# Model A: 최근 50년 (1956-2005)
model_50 = LinearRegression()
model_50.fit(train_50[['Year']], train_50['Temp'])
pred_50 = model_50.predict(X_test)

# Model B: 최근 100년 (1906-2005)
model_100 = LinearRegression()
model_100.fit(train_100[['Year']], train_100['Temp'])
pred_100 = model_100.predict(X_test)

# 4. 성능 평가 함수
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

mae_50, mse_50, r2_50 = evaluate(y_test, pred_50)
mae_100, mse_100, r2_100 = evaluate(y_test, pred_100)

print(f"[최근 50년 모델] 기울기: {model_50.coef_[0]:.4f} | MAE: {mae_50:.3f} | MSE: {mse_50:.3f} | R²: {r2_50:.3f}")
print(f"[최근 100년 모델] 기울기: {model_100.coef_[0]:.4f} | MAE: {mae_100:.3f} | MSE: {mse_100:.3f} | R²: {r2_100:.3f}")
