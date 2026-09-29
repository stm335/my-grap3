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

# 선형 회귀 계산 (1차 다항식 피팅)
X = data["경과연수"].values
y = data["연평균기온"].values
slope, intercept = np.polyfit(X, y, 1)

# 상관계수 계산
corr = np.corrcoef(data["연도"], data["연평균기온"])[0, 1]

# 정보 요약 수치
num_years = len(data)
start_year = int(data["연도"].min())
end_year = int(data["연도"].max())

# 사이드바 / 메인 영역 구성
st.subheader("📌 회귀 모델 요약 정보")
col1, col2, col3, col4 = st.columns(4)
col1.metric("사용된 연도 수", f"{num_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{corr:.4f}")

st.divider()

# 예측 슬라이더 영역
st.subheader("🔮 연도별 예상 기온 예측")
target_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)

# 선택한 연도의 예측 기온 계산
predicted_temp = slope * (target_year - 1908) + intercept
st.metric(label=f"{target_year}년 예상 연평균 기온", value=f"{predicted_temp:.2f} °C")

# Plotly 시각화
# 회귀 직선용 연도 범위 (1900~2100년)
line_years = np.arange(1900, 2101)
line_x = line_years - 1908
line_y = slope * line_x + intercept

fig = go.Figure()

# 1. 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode="markers",
    name="관측 연평균기온",
    marker=dict(color="royalblue", size=7, opacity=0.8)
))

# 2. 회귀 직선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y,
    mode="lines",
    name="선형 회귀선",
    line=dict(color="firebrick", width=2)
))

# 3. 선택한 연도의 예측점 표시
fig.add_trace(go.Scatter(
    x=[target_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택한 연도 ({target_year}년)",
    marker=dict(color="orange", size=14, symbol="star")
))

fig.update_layout(
    title="서울 연평균 기온 추이 및 선형 회귀 모델",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
