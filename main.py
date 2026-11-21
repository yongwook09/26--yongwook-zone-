import streamlit as st
import pandas as pd
import numpy as np
import io
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ── 한글 폰트 CSS ──
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── CSV 읽기 (인코딩 대응) ──
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ──
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ── 페이지 설정 ──
st.set_page_config(page_title="극지식물 상관관계 분석", layout="wide")
st.title("🌱 지상부 요소로 지하부 상태 예측하기 : 극지식물 생육 상관관계 분석")

# ── 사이드바 ──
st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

st.sidebar.markdown("### 📊 생육 측정 항목")
metric_options = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_metrics = st.sidebar.multiselect(
    "표시할 항목 선택",
    metric_options,
    default=metric_options,
)

# ── 데이터 로딩 ──
env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = "우리 학교"

with st.spinner("데이터를 불러오는 중..."):
    if env_file is not None:
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)
        name_part = env_file.name.split("_")[0]
        if name_part and not name_part[0].isdigit():
            school_name = name_part
    if growth_file is not None:
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)
        name_part = growth_file.name.split("_")[0]
        if name_part and not name_part[0].isdigit():
            school_name = name_part

# ── 탭 ──
tab1, tab2, tab3 = st.tabs(
    ["📖 1장 핵심 질문 · 가설", "📊 2장 상관관계 분석", "🔍 3장 실생활 활용"]
)

# ═══════════════════════════════════════
# 탭 1: 핵심 질문 · 가설
# ═══════════════════════════════════════
with tab1:
    try:
        st.markdown(f"### 🏫 {school_name} 극지식물 생육 연구")

        st.subheader("❓ 핵심 질문")
        st.markdown(
            """
**이 연구에서 던지는 질문**

> "지상부에서 관찰되는 특징(잎 수, 지상부 길이 등)을 알면,
> 눈에 보이지 않는 지하부 상태(지하부 길이, 지하부 질량)를
> 예측할 수 있을까?"

이 질문은 왜 중요할까?

- 극지식물은 잎과 지상부가 작아 보여도 지하부가 매우 발달한 경우가 있음
- 지상부만 보고 지하부 상태를 추정할 수 있다면, **흙을 파지 않고도** 개체 건강 상태를 파악할 수 있음
- 이는 실험 시간을 줄이고 개체를 손상시키지 않는 **비파괴 측정**으로 이어짐
"""
        )

        st.subheader("💡 질문을 통해 알 수 있는 실용적 이익")
        st.markdown(
            """
**이 분석을 통해 얻을 수 있는 것**

1. **측정 효율 향상**
   - 지하부 측정은 식물을 캐내야 해서 시간이 오래 걸림
   - 지상부 특징과의 상관관계를 알면, 일부 개체만 정밀 측정하고 나머지는 추정 가능

2. **개체 선발의 객관성**
   - "잎이 많으니까 지하부도 튼튼할 것"이라는 막연한 예상을 **수치로 검증**
   - 상관계수가 높으면 예측에 대한 확신이 생기고, 낮으면 다른 지표를 찾아야 한다는 결론

3. **실험 설계 개선**
   - 어떤 지상부 지표가 지하부와 가장 관련이 높은지 알면, 다음 실험에서 그 지표를 집중 측정
   - 자원(시간·노력)을 가장 효율적인 곳에 배분
"""
        )

        st.subheader("🧪 가설")
        st.markdown(
            """
**가설 1: 지하부 길이 vs 잎 수**

- 예측: **약한 양의 상관관계** 또는 **상관관계가 거의 없을 것**
- 이유: 극지식물은 잎 수가 적어도 지하부가 길 수 있고, 잎이 많아도 지하부가 짧을 수 있음
- 지상부 잎 수는 광합성 기관이지만, 지하부 길이와 직접 연동되지 않을 가능성이 큼

**가설 2: 지하부 길이 vs 지상부 길이**

- 예측: **강한 양의 상관관계**가 있을 것
- 이유: 지상부가 길다는 것은 전체 식물체가 크다는 뜻 → 지하부도 함께 발달했을 가능성이 높음
- 식물의 전체 크기와 지하부 길이는 일반적으로 함께 증가하는 경향

**가설 3 (확장): 지상부 길이 + 잎 수로 지하부 길이 예측**

- 두 지상부 지표를 함께 쓰면 지하부 길이 예측 정확도가 올라갈 가능성
- 다만 이 분석에서는 개별 상관관계(산점도 + 상관계수)로 먼저 확인
"""
        )

        st.subheader("📌 통계 용어 설명")
        st.markdown(
            """
- **상관계수(r)**: 두 변수의 관계를 -1 ~ 1로 나타냄
  - 1에 가까울수록 함께 증가 (양의 상관)
  - -1에 가까울수록 한쪽 증가 시 다른 쪽 감소 (음의 상관)
  - 0에 가까울수록 관계 없음
- **산점도**: 두 변수의 관계를 점(dot)으로 나타낸 그래프
  - 점이 우상향 직선 모양이면 양의 상관, 우하향이면 음의 상관
- **추세선**: 산점도 점들의 경향을 나타내는 직선 (여기서는 np.polyfit으로 계산)
- **결정계수(R²)**: 추세선이 데이터를 얼마나 잘 설명하는지 (0 ~ 1)
"""
        )

        # 데이터 준비 상태 표시
        if growth_df.empty:
            st.info("생육 CSV를 업로드하면 위 가설을 실제 데이터로 검증할 수 있습니다.")
        else:
            st.success(f"✅ {school_name} 생육 데이터가 로드되었습니다. 탭2에서 가설을 검증해 보세요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════
# 탭 2: 상관관계 분석 (산점도)
# ═══════════════════════════════════════
with tab2:
    try:
        if growth_df.empty:
            st.info(f"{school_name}의 생육 CSV를 업로드하면 여기에 상관관계 분석 결과가 표시됩니다.")
        else:
            st.subheader("🔗 상관관계 분석 개요")
            st.markdown(
                """
이 탭에서는 **지상부 지표 2개**와 **지하부 지표(지하부 길이)** 의 관계를
산점도(scatter plot)로 visualizes합니다.

| X 축 (지상부) | Y 축 (지하부) | 예상 상관계수 |
|---|---|---|
| 잎 수 | 지하부 길이 | 약함 (가설 1) |
| 지상부 길이 | 지하부 길이 | 강함 (가설 2) |

각 산점도에는 **추세선**과 **상관계수(r)** 가 함께 표시됩니다.
"""
            )

            # ── 산점도 생성 헬퍼 ──
            def make_scatter_plot(x_col, y_col, df):
                """x_col vs y_col 산점도 + 추세선 + 상관계수 반환"""
                sub = df[[x_col, y_col]].dropna()
                if len(sub) < 2:
                    return None, None, None
                r_val = sub[x_col].corr(sub[y_col])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sub[x_col].tolist(),
                    y=sub[y_col].tolist(),
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="#636EFA",
                        line=dict(width=1, color="white"),
                        opacity=0.8,
                    ),
                    hovertemplate=f"{x_col}: %x<br>{y_col}: %y<extra></extra>",
                    name="개체",
                ))
                # 추세선 (np.polyfit)
                mask = sub[x_col].notna() & sub[y_col].notna()
                xc = sub[x_col][mask]
                yc = sub[y_col][mask]
                if len(xc) >= 2:
                    slope, intercept = np.polyfit(xc, yc, 1)
                    x_line = np.linspace(xc.min(), xc.max(), 100)
                    y_line = slope * x_line + intercept
                    r2 = r_val ** 2
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#EF553B", dash="dash", width=2),
                        name=f"추세선",
                        hovertemplate=f"기울기: {slope:.3f}<br>R²: {r2:.3f}<extra></extra>",
                    ))
                fig.update_layout(
                    title=f"{x_col} vs {y_col}",
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=400,
                    margin=dict(l=50, r=30, t=50, b=50),
                    showlegend=True,
                )
                return fig, r_val, len(sub)

            # ── 산점도 1: 잎 수 vs 지하부 길이 ──
            st.subheader("📌 산점도 1: 잎 수 vs 지하부 길이")
            if "잎 수" in growth_df.columns and "지하부 길이" in growth_df.columns:
                fig1, r1, n1 = make_scatter_plot("잎 수", "지하부 길이", growth_df)
                if fig1 is not None:
                    st.plotly_chart(fig1, width="stretch")
                    col1a, col1b, col1c = st.columns(3)
                    with col1a:
                        st.metric("상관계수(r)", f"{r1:.3f}")
                    with col1b:
                        strength = (
                            "강한 양의 상관" if r1 > 0.7
                            else "약한 양의 상관" if r1 > 0.3
                            else "상관 거의 없음" if abs(r1) < 0.3
                            else "약한 음의 상관" if r1 > -0.7
                            else "강한 음의 상관"
                        )
                        st.metric("상관 강도", strength)
                    with col1c:
                        st.metric("데이터 수", f"{n1}개체")
                    st.info(
                        "💡 가설 1 검증: 잎 수와 지하부 길이의 상관이 약하면, "
                        "잎 수만으로는 지하부 길이를 예측하기 어렵다는 결론"
                    )
                else:
                    st.info("잎 수 또는 지하부 길이 데이터가 부족합니다.")
            else:
                st.info("잎 수 또는 지하부 길이 열이 데이터에 없습니다.")

            st.markdown("---")

            # ── 산점도 2: 지상부 길이 vs 지하부 길이 ──
            st.subheader("📌 산점도 2: 지상부 길이 vs 지하부 길이")
            if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                fig2, r2, n2 = make_scatter_plot("지상부 길이", "지하부 길이", growth_df)
                if fig2 is not None:
                    st.plotly_chart(fig2, width="stretch")
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        st.metric("상관계수(r)", f"{r2:.3f}")
                    with col2b:
                        strength2 = (
                            "강한 양의 상관" if r2 > 0.7
                            else "약한 양의 상관" if r2 > 0.3
                            else "상관 거의 없음" if abs(r2) < 0.3
                            else "약한 음의 상관" if r2 > -0.7
                            else "강한 음의 상관"
                        )
                        st.metric("상관 강도", strength2)
                    with col2c:
                        st.metric("데이터 수", f"{n2}개체")
                    st.info(
                        "💡 가설 2 검증: 지상부 길이와 지하부 길이의 상관이 강하면, "
                        "지상부 길이로 지하부 길이를 비교적 잘 예측할 수 있다는 결론"
                    )
                else:
                    st.info("지상부 길이 또는 지하부 길이 데이터가 부족합니다.")
            else:
                st.info("지상부 길이 또는 지하부 길이 열이 데이터에 없습니다.")

            st.markdown("---")

            # ── 두 상관관계 비교 요약 ──
            st.subheader("📊 두 상관관계 비교 요약")
            if "잎 수" in growth_df.columns and "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                sub1 = growth_df[["잎 수", "지하부 길이"]].dropna()
                sub2 = growth_df[["지상부 길이", "지하부 길이"]].dropna()
                if len(sub1) >= 2 and len(sub2) >= 2:
                    r_leaf = sub1["잎 수"].corr(sub1["지하부 길이"])
                    r_shoot = sub2["지상부 길이"].corr(sub2["지하부 길이"])

                    comp_rows = []
                    comp_rows.append({
                        "X (지상부 지표)": "잎 수",
                        "Y (지하부 지표)": "지하부 길이",
                        "상관계수(r)": round(r_leaf, 3),
                        "관계 강도": (
                            "강한 양의 상관" if r_leaf > 0.7
                            else "약한 양의 상관" if r_leaf > 0.3
                            else "상관 거의 없음" if abs(r_leaf) < 0.3
                            else "약한 음의 상관" if r_leaf > -0.7
                            else "강한 음의 상관"
                        ),
                        "예측 적합성": "낮음" if abs(r_leaf) < 0.5 else "보통" if abs(r_leaf) < 0.7 else "높음",
                        "데이터 수": len(sub1),
                    })
                    comp_rows.append({
                        "X (지상부 지표)": "지상부 길이",
                        "Y (지하부 지표)": "지하부 길이",
                        "상관계수(r)": round(r_shoot, 3),
                        "관계 강도": (
                            "강한 양의 상관" if r_shoot > 0.7
                            else "약한 양의 상관" if r_shoot > 0.3
                            else "상관 거의 없음" if abs(r_shoot) < 0.3
                            else "약한 음의 상관" if r_shoot > -0.7
                            else "강한 음의 상관"
                        ),
                        "예측 적합성": "낮음" if abs(r_shoot) < 0.5 else "보통" if abs(r_shoot) < 0.7 else "높음",
                        "데이터 수": len(sub2),
                    })
                    st.dataframe(pd.DataFrame(comp_rows), width="stretch")

                    # 인사이트 박스
                    if abs(r_shoot) > abs(r_leaf):
                        st.success(
                            f"✅ 지상부 길이(r={r_shoot:.2f})가 잎 수(r={r_leaf:.2f})보다 "
                            "지하부 길이와의 상관이 더 강합니다. "
                            "지상부 길이를 지하부 예측의 주요 지표로 사용하는 것이 더 적합합니다."
                        )
                    else:
                        st.info(
                            f"잎 수(r={r_leaf:.2f})와 지상부 길이(r={r_shoot:.2f})의 "
                            "지하부 길이에 대한 상관 정도가 비슷합니다."
                        )
                else:
                    st.info("비교에 필요한 데이터가 부족합니다.")
            else:
                st.info("비교를 위해 잎 수, 지상부 길이, 지하부 길이가 모두 필요합니다.")

            # ── 개체별 측정값 테이블 ──
            st.subheader("📄 개체별 측정값 (산점도 데이터)")
            plot_cols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
            if plot_cols:
                st.dataframe(growth_df[plot_cols], width="stretch")
            else:
                st.info("표시할 측정 항목이 없습니다.")

            # ── 원본 다운로드 ──
            with st.expander("📥 생육 데이터 CSV 다운로드"):
                if not growth_df.empty:
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name}_생육결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("다운로드할 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════
# 탭 3: 실생활 활용
# ═══════════════════════════════════════
with tab3:
    try:
        st.subheader("🔍 생육 결과로부터 산출한 요소별 실생활 대입")

        if growth_df.empty:
            st.info("생육 CSV를 업로드하면 실생활 활용 예시가 여기에 표시됩니다.")
        else:
            # 데이터 기반 수치 계산
            has_shoot_len = "지상부 길이" in growth_df.columns
            has_root_len = "지하부 길이" in growth_df.columns
            has_leaves = "잎 수" in growth_df.columns
            has_bw = "생중량" in growth_df.columns

            st.markdown(
                """
### 🌿 실생활 활용 아이디어

이 분석에서 얻은 상관관계 결과를 실제 상황에 적용하면
다음과 같은 활용이 가능합니다.
"""
            )

            # 활용 1: 비파괴 개체 선발
            st.subheader("1️⃣ 비파괴 개체 선발 도구")
            if has_shoot_len and has_root_len:
                sub = growth_df[["지상부 길이", "지하부 길이"]].dropna()
                if len(sub) >= 2:
                    r_val = sub["지상부 길이"].corr(sub["지하부 길이"])
                    st.markdown(
                        f"""
**상황:** 실험실에서 극지식물 개체 중 "건강하고 잘 자란 개체"를 골라야 할 때,
지하부를 확인하려면 식물을 캐내야 해서 개체가 손상됨.

**활용:** 지상부 길이만 측정해도 지하부 길이를 추정할 수 있다면
**개체를 손상시키지 않고** 건강한 개체를 고를 수 있음.

- 우리 학교 데이터: 지상부 길이-지하부 길이 상관계수 **r = {r_val:.2f}**
- r이 {0.7 if r_val>0.7 else 0.5 if r_val>0.5 else 0.3} 이상이면 → 지상부 길이로 지하부 길이 예측 가능
- 추청 공식(추세선): **지하부 길이 ≈ {sub["지상부 길이"].corr(sub["지하부 길이"]) * (sub["지상부 길이"].std()/sub["지상부 길이"].std()):.2f} × 지상부 길이 + 상수**
  (실제 상수는 추세선의 절편으로 계산)
                        """
                    )
                else:
                    st.info("상관계수를 계산할 데이터가 부족합니다.")
            else:
                st.info("지상부 길이와 지하부 길이 데이터가 모두 필요합니다.")

            # 활용 2: 측정 항목 최적화
            st.subheader("2️⃣ 측정 항목 최적화 (시간 절약)")
            st.markdown(
                """
**상황:** 실험 시간이 부족해서 모든 개체의 모든 부분을 측정할 수 없을 때,
어떤 항목을 우선 측정할지 결정해야 함.

**활용:**

| 지상부 지표 | 지하부 예측 유용성 | 측정 난이도 | 추천 우선순위 |
|---|---|---|---|
| 지상부 길이 | 높음 (r이 큰 경향) | 낮음 (자 하나로 측정) | ⭐⭐⭐⭐⭐ |
| 잎 수 | 낮음 (r이 작은 경향) | 낮음 (세는 방식) | ⭐⭐ |

- 지상부 길이는 측정도 쉽고 지하부와의 상관도 높은 편 → **우선 측정 지표**
- 잎 수는 측정은 쉽지만 지하부 예측력은 낮은 경향 → 참고용으로만 활용
"""
            )

            # 활용 3: 생장 예측 모델 기초
            st.subheader("3️⃣ 생장 예측 모델의 기초 자료")
            if has_shoot_len and has_root_len:
                sub = growth_df[["지상부 길이", "지하부 길이"]].dropna()
                if len(sub) >= 3:
                    slope, intercept = np.polyfit(sub["지상부 길이"], sub["지하부 길이"], 1)
                    st.markdown(
                        f"""
**상황:** 앞으로 기를 극지식물의 지하부 길이를 미리 알고 싶을 때,
지상부 길이만 재면 대략적인 지하부 길이를 예측 가능.

**추정 공식 (우리 학교 데이터 기준):**

> **지하부 길이(mm) ≈ {slope:.2f} × 지상부 길이(mm) + {intercept:.2f}**

- 이 공식은 **우리 학교 개체들의 데이터**로 만든 것이므로,
  다른 학교·환경에서는 달라질 수 있음
- 새로운 개체의 지상부 길이를 재서 위 공식에 넣으면
  지하부 길이 추정치를 얻을 수 있음
- 단, 이는 **추정**일 뿐 실제 측정과 오차가 있을 수 있으니 주의
                        """
                    )
                else:
                    st.info("추세선을 만들 데이터가 부족합니다.")
            else:
                st.info("지상부 길이와 지하부 길이 데이터가 모두 필요합니다.")

            # 활용 4: 교육적 활용
            st.subheader("4️⃣ 통계·과학 교육 활용")
            st.markdown(
                """
**이 분석을 과학 수업에서 활용할 수 있는 방법:**

1. **가설 설정 연습**
   - "잎 수가 많으면 지하부도 길까?" 같은 가설을 세우고, 실제 데이터로 검증
   - 가설이 맞았을 때와 틀렸을 때 모두 배울 점이 있음

2. **상관관계 vs 인과관계 구분**
   - 상관이 있다고 해서 "잎 수가 지하부 길이를 CAUSED한다"는 아님
   - 둘 다 전체 식물 크기가 커서 함께 증가한 것일 수 있음
   - 이 차이를 배우는 것이 과학적 사고력 향상에 도움

3. **데이터 시각화 해석**
   - 산점도 점의 분포를 보고 상관의 방향과 강도를 읽는 연습
   - 추세선과 상관계수를 함께 보는 습관
"""
            )

            # 요약 카드
            st.subheader("📋 연구 요약 카드")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("#### 🔬 연구 질문")
                st.write(
                    "지상부 특징(잎 수, 지상부 길이)으로\n"
                    "지하부 상태(지하부 길이)를\n"
                    "예측할 수 있을까?"
                )

            with c2:
                st.markdown("#### 📊 핵심 발견")
                if has_shoot_len and has_root_len:
                    sub = growth_df[["지상부 길이", "지하부 길이"]].dropna()
                    if len(sub) >= 2:
                        r_shoot = sub["지상부 길이"].corr(sub["지하부 길이"])
                        st.write(f"지상부 길이 ↔ 지하부 길이\n**r = {r_shoot:.2f}**")
                    else:
                        st.write("데이터 부족")
                else:
                    st.write("데이터 없음")

            with c3:
                st.markdown("#### 🛠️ 실용 가치")
                st.write(
                    "• 비파괴 개체 선발\n"
                    "• 측정 항목 최적화\n"
                    "• 생장 추정 공식\n"
                    "• 과학 교육 자료"
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
