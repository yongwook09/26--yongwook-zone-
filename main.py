import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ※ set_page_config 는 반드시 첫 번째 Streamlit 명령이어야 합니다.
st.set_page_config(page_title="극지식물 생육 상관관계 분석", layout="wide")

FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── 캐시된 CSV 읽기 (바이트 기준으로 캐시) ─────────────────────────────────
@st.cache_data
def read_csv_bytes(data):
    if not data:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def read_csv(file):
    if file is None:
        return pd.DataFrame()
    return read_csv_bytes(file.getvalue())


# ── 열 이름 표준화 ─────────────────────────────────────────────────────────
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
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ── 상관계수 / 기울기 계산 ─────────────────────────────────────────────────
def calc_pair(df, xl, yl):
    """(r, slope, intercept, valid) 반환. 계산 불가 시 r=None."""
    valid = df[[xl, yl]].dropna()
    if len(valid) < 2:
        return None, None, None, valid
    xx = valid[xl].values.astype(float)
    yy = valid[yl].values.astype(float)
    if np.std(xx) == 0 or np.std(yy) == 0:
        return None, None, None, valid
    r = float(np.corrcoef(xx, yy)[0, 1])
    slope, intercept = np.polyfit(xx, yy, 1)
    return r, float(slope), float(intercept), valid


def interpret_r(r):
    if r is None:
        return "계산 불가 (값이 일정하거나 표본 부족)"
    if r >= 0.7:
        return "강한 양의 상관 ✅"
    if r >= 0.3:
        return "약한 양의 상관"
    if r > -0.3:
        return "거의 상관 없음 ➖"
    if r >= -0.7:
        return "약한 음의 상관"
    return "강한 음의 상관"


# ── 학교명 추출 (파일명에서) ───────────────────────────────────────────────
def school_name(filename):
    if filename is None:
        return "미확인 학교"
    base = filename.split(".")[0]
    base = base.replace("_", " ").replace("-", " ")
    words = base.split()
    parts = []
    for w in words:
        buf = ""
        for ch in w:
            if ch.isupper() and buf:
                parts.append(buf)
                buf = ch
            else:
                buf += ch
        if buf:
            parts.append(buf)
    return " ".join(p for p in parts if p).strip() or "미확인 학교"


# ── 사이드바 ──────────────────────────────────────────────────────────────
st.sidebar.title("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

growth_measures = st.sidebar.multiselect(
    "📊 생육 측정 항목 (그래프에 포함)",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"],
    default=["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
)

# 데이터 로드
with st.spinner("데이터 로딩 중..."):
    raw_env = read_csv(env_file)
    raw_growth = read_csv(growth_file)
    env_df = std_cols(raw_env)
    growth_df = std_cols(raw_growth)

school = school_name(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)

st.title("🌿 지상부 요소로 지하부 상태 예측하기: 극지식물 생육 상관관계 분석")
st.caption(f"분석 대상: {school}")

tab1, tab2, tab3 = st.tabs(["1장 핵심 질문", "2장 생육 결과", "3장 실생활 활용"])

# ══════════════════════════════════════════════════════════════════════════
# 탭 1 ─ 핵심 질문 / 실용적 이익 / 가설
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        if env_df.empty:
            st.info("환경 데이터가 업로드되지 않았어요. 이 탭의 내용은 생육 데이터만으로도 볼 수 있어요.")
        st.header("1장 핵심 질문")
        st.markdown("""
**❓ 질문 1: 지상부(잎·줄기)를 보면 지하부(뿌리) 상태를 예측할 수 있을까?**

극지식물은 땅 위보다 땅속 환경이 더 가혹한 경우가 많습니다.
땅 위에서 관찰한 잎 수, 줄기 길이 같은 값으로 뿌리의 길이·질량을
대략 예측할 수 있다면, 뿌리를 파보지 않고도 식물 건강 상태를 빠르게
가늠할 수 있습니다.
        """)
        st.markdown("""
**❓ 질문 2: 잎 수와 지하부 길이는 정말 관계가 있을까?**

일반적으로 잎이 많으면 뿌리도 발달했을 것 같지만, 극지방에서는
이 관계가 생각보다 약할 수 있습니다. 실제로 이 데이터에서
두 값의 상관관계를 확인해 봅니다.

**❓ 질문 3: 지상부 길이와 지하부 길이는 얼마나 강하게 연관되어 있을까?**

줄기가 길면 뿌리도 깊게 뻗는다는 직관이 맞는지, 상관계수로 숫자로
확인합니다.
        """)

        st.header("질문을 통해 알 수 있는 실용적 이익")
        st.markdown("""
- **비파괴 측정 가능**: 뿌리를 캐내지 않고 지상부만 측정해도 지하부
  상태를 추정할 수 있습니다.
- **Quick Screening**: 대량 개체 조사 시 시간이 크게 절약됩니다.
- **생육 이상 조기 감지**: 지상부 대비 지하부가 불균형한 개체를
  골라낼 수 있습니다.
        """)

        st.header("가설")
        st.markdown("""
| 가설 | 내용 |
|------|------|
| 가설 1 | 잎 수가 많을수록 지하부 길이도 길어질 것이다 (약한 양의 상관관계 예상) |
| 가설 2 | 지상부 길이가 길수록 지하부 길이도 길어질 것이다 (강한 양의 상관관계 예상) |
| 가설 3 | 지상부 질량·지하부 질량도 지상부 길이와 유사한 패턴을 보일 것이다 |
        """)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════════════════════════════════
# 탭 2 ─ 생육 결과 (산점도 + 상관계수)
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if growth_df.empty:
            st.info("📎 생육 결과 CSV를 업로드하면 이 탭의 분석을 볼 수 있어요.")
        else:

            st.header("2장 생육 결과 — 상관관계 분석")

            # ── 기본 지표 카드 ─────────────────────────────────────────────
            for label in ("지상부 길이", "지하부 길이"):
                if label in growth_df.columns:
                    v = growth_df[label].dropna()
                    if len(v) > 0:
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"{label} 평균 (mm)", f"{v.mean():.1f}")
                        c2.metric(f"{label} 최대 (mm)", f"{v.max():.1f}")
                        c3.metric(f"{label} 최소 (mm)", f"{v.min():.1f}")

            # ── 분석 대상 열 (사이드바 선택 반영) ─────────────────────────
            all_cols = ["잎 수", "지상부 길이", "지하부 길이",
                        "지상부 질량", "지하부 질량", "생중량"]
            gcols = [c for c in all_cols
                     if c in growth_df.columns and c in growth_measures]

            if not gcols:
                st.info("표시할 생육 측정 항목이 없어요. 사이드바에서 항목을 선택하거나 CSV의 열 이름을 확인해 주세요.")
            else:
                # ── 상관관계 산점도 (격자 배치) ─────────────────
                st.subheader("🔗 상관관계 산점도 (추세선 + 상관계수)")

                x_candidates = [c for c in ["잎 수", "지상부 길이", "지상부 질량", "생중량"]
                                if c in growth_df.columns and c in growth_measures]
                y_candidates = [c for c in ["지하부 길이", "지하부 질량"]
                                if c in growth_df.columns and c in growth_measures]

                pairs = []  # (xl, yl, r, slope, intercept, valid)
                for xl in x_candidates:
                    for yl in y_candidates:
                        r, slope, intercept, valid = calc_pair(growth_df, xl, yl)
                        pairs.append((xl, yl, r, slope, intercept, valid))

                if not pairs:
                    st.info("상관관계 분석에 필요한 열이 생육 CSV에 없어요. (X: 잎 수·지상부 길이·지상부 질량·생중량 / Y: 지하부 길이·지하부 질량)")
                else:
                    n = len(pairs)
                    ncols = min(n, 3)
                    nrows = math.ceil(n / ncols)

                    titles = []
                    for xl, yl, r, slope, _, _ in pairs:
                        r_txt = f"r={r:.3f}" if r is not None else "r=계산불가"
                        titles.append(f"{xl} vs {yl}<br><sup>{r_txt}</sup>")

                    fig_corr = make_subplots(
                        rows=nrows, cols=ncols,
                        subplot_titles=titles,
                        horizontal_spacing=0.08,
                        vertical_spacing=0.20 if nrows > 1 else 0.1,
                    )

                    for idx, (xl, yl, r, slope, intercept, valid) in enumerate(pairs):
                        row = idx // ncols + 1
                        col = idx % ncols + 1
                        if len(valid) >= 1:
                            fig_corr.add_trace(
                                go.Scatter(
                                    x=valid[xl].values,
                                    y=valid[yl].values,
                                    mode="markers",
                                    marker=dict(
                                        color=PALETTE[idx % len(PALETTE)],
                                        size=8, opacity=0.7,
                                        line=dict(width=0.5, color="white"),
                                    ),
                                    name=f"{xl} vs {yl}",
                                    hovertemplate=f"{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
                                    showlegend=False,
                                ),
                                row=row, col=col,
                            )
                        if r is not None:
                            xx = valid[xl].values.astype(float)
                            line_x = np.array([xx.min(), xx.max()])
                            line_y = slope * line_x + intercept
                            fig_corr.add_trace(
                                go.Scatter(
                                    x=line_x, y=line_y,
                                    mode="lines",
                                    line=dict(color="red", dash="dash", width=2),
                                    name=f"추세선 (기울기={slope:.3f})",
                                    hovertemplate=f"기울기={slope:.3f}<extra></extra>",
                                    showlegend=False,
                                ),
                                row=row, col=col,
                            )
                        fig_corr.update_xaxes(title_text=xl, row=row, col=col)
                        fig_corr.update_yaxes(title_text=yl, row=row, col=col)

                    fig_corr.update_annotations(font_size=13)
                    fig_corr.update_layout(
                        showlegend=False,
                        font=dict(family=FONT_FAMILY),
                        height=380 * nrows,
                        margin=dict(l=50, r=30, t=80, b=50),
                    )
                    st.plotly_chart(fig_corr, width="stretch")

                    # ── 상관계수 표 ────────────────────────────────────────
                    rows_tbl = []
                    for xl, yl, r, slope, _, valid in pairs:
                        rows_tbl.append({
                            "X (지상부 요소)": xl,
                            "Y (지하부 요소)": yl,
                            "표본 수 (n)": len(valid),
                            "상관계수 r": f"{r:.3f}" if r is not None else "-",
                            "기울기": f"{slope:.3f}" if slope is not None else "-",
                            "해석": interpret_r(r),
                        })
                    st.dataframe(
                        pd.DataFrame(rows_tbl),
                        width="stretch",
                        hide_index=True,
                    )


            # ── 환경 데이터 요약 ───────────────────────────────────────────
            if not env_df.empty:
                st.divider()
                st.subheader("🌡️ 환경 데이터 간략 요약")
                env_num = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns
                ]
                if env_num:
                    evals = {}
                    for col_name in env_num:
                        v = env_df[col_name].dropna()
                        if len(v) >= 2:
                            evals[col_name] = {
                                "평균": round(v.mean(), 2),
                                "최대": round(v.max(), 2),
                                "최소": round(v.min(), 2),
                            }
                    if evals:
                        edf = pd.DataFrame(evals).T
                        edf.index.name = "측정 항목"
                        st.dataframe(edf, width="stretch", hide_index=False)
                    else:
                        st.info("환경 데이터 값이 충분치 않아요.")
                else:
                    st.info("환경 CSV에 분석 가능한 숫자 열이 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════════════════════════════════
# 탭 3 ─ 실생활 활용
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        st.header("3장 생육 결과에 대한 실생활 활용")

        # ── 목적 (가장 먼저, 분명하게) ────────────────────────────────────
        st.success(
            "🎯 **이 프로젝트의 목적**\n\n"
            "**지상부(잎 수·줄기 길이·지상부 질량)를 봄으로써 지하부(뿌리 길이·질량)의 상태를 예측할 수 있다.**\n\n"
            "뿌리를 캐내지 않고 땅 위에서 관찰한 값만으로 식물의 땅속 상태를 알아내는 것, "
            "이것이 우리가 이 분석을 하는 이유입니다."
        )

        # ── 이 데이터로 확인한 결과 ───────────────────────────────────────
        # 생중량은 지하부 질량을 포함한 값이라 예측 근거에서 제외합니다.
        pred_pairs = []
        if not growth_df.empty:
            for xl in ["잎 수", "지상부 길이", "지상부 질량"]:
                if xl not in growth_df.columns:
                    continue
                for yl in ["지하부 길이", "지하부 질량"]:
                    if yl not in growth_df.columns:
                        continue
                    r, slope, intercept, valid = calc_pair(growth_df, xl, yl)
                    if r is not None:
                        pred_pairs.append((xl, yl, r, slope, intercept, valid))

        st.subheader("📊 우리 데이터로 확인한 결과")
        if growth_df.empty:
            st.info("생육 CSV를 업로드하면 '지상부로 지하부를 예측할 수 있는지'를 데이터로 확인할 수 있어요.")
        elif not pred_pairs:
            st.info("예측 근거를 계산할 열(지상부·지하부 길이/질량)이 생육 CSV에 부족해요.")
        else:
            tbl = []
            for xl, yl, r, slope, _, valid in pred_pairs:
                if abs(r) >= 0.7:
                    power = "예측에 활용 가능 ✅"
                elif abs(r) >= 0.3:
                    power = "참고용 (오차 클 수 있음)"
                else:
                    power = "예측 어려움 ➖"
                tbl.append({
                    "지상부 요소 (보는 값)": xl,
                    "지하부 요소 (예측할 값)": yl,
                    "표본 수 (n)": len(valid),
                    "상관계수 r": f"{r:.3f}",
                    "설명력 R²": f"{r * r:.2f}",
                    "예측 가능성": power,
                })
            st.dataframe(pd.DataFrame(tbl), width="stretch", hide_index=True)

            best = max(pred_pairs, key=lambda p: abs(p[2]))
            bx, by, br = best[0], best[1], best[2]
            if abs(br) >= 0.7 and br > 0:
                st.success(
                    f"✅ **결론: 지상부로 지하부를 예측할 수 있습니다.** "
                    f"가장 강한 관계는 **{bx} → {by}** (r = {br:.3f}, R² = {br * br:.2f})입니다. "
                    f"{bx}만 측정해도 {by}의 약 {br * br * 100:.0f}%가 설명됩니다."
                )
            elif abs(br) >= 0.3:
                st.warning(
                    f"⚠️ **결론: 부분적으로만 예측할 수 있습니다.** "
                    f"가장 강한 관계는 **{bx} → {by}** (r = {br:.3f}, R² = {br * br:.2f})로, "
                    f"대략적인 경향을 보는 참고용으로만 쓰는 것이 좋습니다."
                )
            else:
                st.error(
                    f"❌ **결론: 이 데이터에서는 지상부로 지하부를 예측하기 어렵습니다.** "
                    f"가장 강한 관계도 {bx} → {by} (r = {br:.3f})로 약합니다."
                )

            # ── 지상부 값으로 지하부 예측해 보기 ──────────────────────────
            st.subheader("🧮 지상부 값으로 지하부 예측해 보기")
            labels = [f"{p[0]} → {p[1]} (r={p[2]:.3f})" for p in pred_pairs]
            default_idx = pred_pairs.index(best)
            choice = st.selectbox("예측할 관계 선택", labels, index=default_idx)
            xl, yl, r, slope, intercept, valid = pred_pairs[labels.index(choice)]
            x_min = float(valid[xl].min())
            x_max = float(valid[xl].max())
            x_val = st.number_input(
                f"{xl} 값 입력",
                value=float(round(valid[xl].mean(), 2)),
                step=0.1,
            )
            y_pred = slope * x_val + intercept
            st.metric(f"예측된 {yl}", f"{y_pred:.2f}")
            if x_val < x_min or x_val > x_max:
                st.warning(
                    f"입력값이 측정 범위({x_min:.1f} ~ {x_max:.1f})를 벗어나 예측이 부정확할 수 있어요."
                )
            st.caption(
                f"예측식: {yl} = {slope:.3f} × {xl} + {intercept:.3f}  |  R² = {r * r:.2f}"
            )

        # ── 실생활 활용 ───────────────────────────────────────────────────
        st.subheader("🌍 이렇게 활용할 수 있어요")

        st.markdown("""
**🌱 뿌리를 캐지 않는 비파괴 생육 진단**

땅 위의 줄기 길이·잎 수만 재고도 뿌리의 길이와 질량을 추정할 수 있다면, 식물을
뽑아 죽이지 않고 같은 개체를 계속 관찰할 수 있습니다. 개체 수가 적고 소중한
극지식물 연구에서 특히 큰 장점입니다.
        """)

        st.markdown("""
**⚡ 대량 개체 빠른 선별 (Quick Screening)**

많은 개체를 조사할 때 지상부만 측정해 지하부 상태를 먼저 추정하고, 예상과 크게
다른 개체만 골라 정밀 조사하면 시간과 노력을 크게 줄일 수 있습니다.
        """)

        st.markdown("""
**🚨 생육 이상 조기 발견**

지상부 값으로 예측한 지하부 값과 실제 값이 크게 다른 개체는 뿌리 발달에 문제가
있을 가능성이 있습니다. 이런 개체를 일찍 찾아 환경(온도·습도·pH·EC)을 점검할 수
있습니다.
        """)

        st.markdown("""
**🧪 현장 조사 지표 선택의 근거**

위 표에서 상관이 강한 지상부 요소는 현장에서 우선 측정할 지표로, 상관이 약한
요소는 단독 판단 지표로 쓰지 말아야 할 지표로 구분할 수 있습니다.
        """)

        st.markdown("""
**📋 교육용 생육 추적 프로젝트**

학생들이 직접 측정한 값으로 상관계수와 예측식을 만들어 보며, 데이터로 가설을
검증하는 과정을 경험할 수 있습니다.
        """)

        st.caption(
            "※ 상관관계는 인과관계가 아니며, 예측식은 이 데이터의 측정 범위와 표본 수 안에서만 믿을 수 있어요. "
            "표본이 적으면 결과가 크게 달라질 수 있습니다."
        )

        # ── 생육 결과 다운로드 ────────────────────────────────────────────
        if not growth_df.empty and "지상부 길이" in growth_df.columns:
            st.divider()
            st.subheader("📥 생육 결과 데이터 다운로드")
            csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 생육 결과 CSV 다운로드 (utf-8-sig)",
                data=csv_bytes,
                file_name="극지식물_생육결과.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
