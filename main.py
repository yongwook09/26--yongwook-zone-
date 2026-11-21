import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── 캐시된 CSV 읽기 ────────────────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

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
                errors="coerce"
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

# ── 추세선 추가 헬퍼 ───────────────────────────────────────────────────────
def add_trendline(fig, x, y, color="red", dash="dash"):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return
    xx = valid.iloc[:, 0].values
    yy = valid.iloc[:, 1].values
    slope, intercept = np.polyfit(xx, yy, 1)
    line_x = np.array([xx.min(), xx.max()])
    line_y = slope * line_x + intercept
    fig.add_trace(
        go.Scatter(
            x=line_x, y=line_y,
            mode="lines",
            line=dict(color=color, dash=dash, width=2),
            name=f"추세선 (기울기={slope:.3f})",
            hoverinfo="skip",
        ),
    )

# ── 산점도 트레이스 추가 헬퍼 ───────────────────────────────────────────────
def scatter_trace(fig, x, y, color="skyblue"):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return None, None
    xx = valid.iloc[:, 0].values
    yy = valid.iloc[:, 1].values
    r = np.corrcoef(xx, yy)[0, 1]
    fig.add_trace(
        go.Scatter(
            x=xx, y=yy,
            mode="markers",
            marker=dict(color=color, size=8, opacity=0.7,
                        line=dict(width=0.5)),
            name=f"{x.name} vs {y.name}",
            hoverinfo="x+y",
        ),
    )
    return r, valid

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

# 학교명 추출 (파일명에서)
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

school = school_name(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)

st.set_page_config(page_title="극지식물 생육 상관관계 분석", layout="wide")
st.title("🌿 지상부 요소로 지하부 상태 예측하기: 극지식물 생육 상관관계 분석")
st.caption(f"분석 대상: {school}")

# ══════════════════════════════════════════════════════════════════════════
# 탭 1 ─ 핵심 질문 / 실용적 이익 / 가설
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["1장 핵심 질문", "2장 생육 결과", "3장 실생활 활용"])

with tab1:
    try:
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
            has_id = "개체번호" in growth_df.columns
            avail = [c for c in growth_df.columns if c not in ("개체번호", "time")]
            sel = [m for m in growth_measures if m in avail]

            st.header("2장 생육 결과 — 상관관계 분석")

            # ── 기본 지표 카드 ─────────────────────────────────────────────
            if "지상부 길이" in growth_df.columns:
                v = growth_df["지상부 길이"].dropna()
                c1, c2, c3 = st.columns(3)
                c1.metric("지상부 길이 평균 (mm)", f"{v.mean():.1f}")
                c2.metric("지상부 길이 최대 (mm)", f"{v.max():.1f}")
                c3.metric("지상부 길이 최소 (mm)", f"{v.min():.1f}")

            if "지하부 길이" in growth_df.columns:
                v = growth_df["지하부 길이"].dropna()
                c1, c2, c3 = st.columns(3)
                c1.metric("지하부 길이 평균 (mm)", f"{v.mean():.1f}")
                c2.metric("지하부 길이 최대 (mm)", f"{v.max():.1f}")
                c3.metric("지하부 길이 최소 (mm)", f"{v.min():.1f}")

            # ── 분석 대상 열 ───────────────────────────────────────────────
            gcols = [
                c for c in [
                    "잎 수", "지상부 길이", "지하부 길이",
                    "지상부 질량", "지하부 질량", "생중량",
                ]
                if c in growth_df.columns
            ]

            if not gcols:
                st.info("생육 CSV에 분석 가능한 숫자 열이 없어요.")
            else:
                # ── 그래프 1: 개체별 지표 분포 (박스 플롯) ────────────────
                st.subheader("📦 개체별 생육 지표 분포")
                fig_box = make_subplots(
                    rows=1, cols=len(gcols),
                    subplot_titles=gcols,
                    horizontal_spacing=0.08,
                )
                box_colors = [
                    "#4C72B0", "#55A868", "#C44E52",
                    "#8172B3", "#CCB974", "#64B5CD",
                ]
                for i, col_name in enumerate(gcols):
                    vals = growth_df[[col_name]].dropna()
                    if has_id and "개체번호" in growth_df.columns:
                        ids = growth_df.loc[vals.index, "개체번호"]
                    else:
                        ids = [str(i) for i in range(len(vals))]
                    fig_box.add_trace(
                        go.Box(
                            y=vals[col_name].values,
                            name=col_name,
                            fillcolor=box_colors[i % len(box_colors)],
                            marker=dict(color=box_colors[i % len(box_colors)]),
                            line=dict(color=box_colors[i % len(box_colors)]),
                            hovertext=list(ids),
                            hoverinfo="y+text",
                        ),
                        row=1, col=i + 1,
                    )
                fig_box.update_layout(
                    showlegend=False,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=350,
                )
                st.plotly_chart(fig_box, width="stretch")

                # ── 산점도: 상관계수 & 추세선 ─────────────────────────────
                st.subheader("🔗 상관관계 산점도 (추세선 + 상관계수)")

                xlabel_candidates = ["잎 수", "지상부 길이", "지상부 질량", "생중량"]
                ylabel_candidates = ["지하부 길이", "지하부 질량"]

                # 각 pair에 대해 r값을 미리 계산해 둠
                pairs_with_r = []
                for xl in xlabel_candidates:
                    if xl not in growth_df.columns:
                        continue
                    for yl in ylabel_candidates:
                        if yl not in growth_df.columns:
                            continue
                        valid = growth_df[[xl, yl]].dropna()
                        if len(valid) >= 2:
                            r = np.corrcoef(valid[xl], valid[yl])[0, 1]
                        else:
                            r = 0.0
                        pairs_with_r.append((xl, yl, r, valid))

                if not pairs_with_r:
                    st.info("상관관계 분석에 필요한 열이 생육 CSV에 없어요.")
                else:
                    # ── subplot title에 r값까지 포함 ───────────────────────
                    subplot_titles = [
                        f"{xl} vs {yl} (r={r:.3f})"
                        for xl, yl, r, _ in pairs_with_r
                    ]
                    n = len(pairs_with_r)
                    fig_corr = make_subplots(
                        rows=1, cols=n,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.10,
                    )
                    colors_list = [
                        "#4C72B0", "#55A868", "#C44E52",
                        "#8172B3", "#CCB974", "#64B5CD",
                    ]
                    for idx, (xl, yl, r, valid) in enumerate(pairs_with_r):
                        scatter_trace(
                            fig_corr,
                            growth_df[xl],
                            growth_df[yl],
                            color=colors_list[idx % len(colors_list)],
                        )
                        add_trendline(fig_corr, valid.iloc[:, 0], valid.iloc[:, 1])

                    fig_corr.update_layout(
                        showlegend=True,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=420,
                        margin=dict(l=50, r=30, t=60, b=40),
                    )
                    st.plotly_chart(fig_corr, width="stretch")

                # ── 상관계수 표 ────────────────────────────────────────────
                if pairs_with_r:
                    rows = []
                    for xl, yl, r, valid in pairs_with_r:
                        n_pts = len(valid)
                        if r >= 0.7:
                            interp = "강한 양의 상관 ✅"
                        elif r >= 0.3:
                            interp = "약한 양의 상관"
                        elif r > -0.3:
                            interp = "거의 상관 없음 ➖"
                        elif r >= -0.7:
                            interp = "약한 음의 상관"
                        else:
                            interp = "강한 음의 상관"
                        rows.append({
                            "X (지상부 요소)": xl,
                            "Y (지하부 요소)": yl,
                            "표본 수 (n)": n_pts,
                            "상관계수 r": f"{r:.3f}",
                            "해석": interp,
                        })
                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("상관계수를 계산할 데이터가 충분하지 않아요.")

                # ── 개체번호별 생중량 상위·하위 5개체 강조 ────────────────
                if "생중량" in growth_df.columns and has_id:
                    st.subheader("🏆 개체별 생중량 순위 (상위·하위 5)")
                    ranked = (
                        growth_df[["개체번호", "생중량"]]
                        .dropna()
                        .sort_values("생중량", ascending=False)
                    )
                    if len(ranked) >= 5:
                        top5 = ranked.head(5)
                        bot5 = ranked.tail(5).iloc[::-1]
                        fig_rank = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=["상위 5개체", "하위 5개체"],
                            horizontal_spacing=0.15,
                        )
                        fig_rank.add_trace(
                            go.Bar(
                                x=top5["개체번호"].astype(str).tolist(),
                                y=top5["생중량"].tolist(),
                                marker=dict(color="#2E86AB", fillcolor="#2E86AB"),
                                name="상위",
                            ),
                            row=1, col=1,
                        )
                        fig_rank.add_trace(
                            go.Bar(
                                x=bot5["개체번호"].astype(str).tolist(),
                                y=bot5["생중량"].tolist(),
                                marker=dict(color="#A23B72", fillcolor="#A23B72"),
                                name="하위",
                            ),
                            row=1, col=2,
                        )
                        fig_rank.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            height=360,
                            showlegend=False,
                        )
                        st.plotly_chart(fig_rank, width="stretch")

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

        st.markdown("""
**🌱 토양 건강 모니터링에 활용**

지상부 길이(줄기)와 지하부 길이(뿌리)의 강한 양의 상관관계가 확인되면,
줄기 길이만으로 뿌리의 발달 정도를 추정할 수 있어 토양 깊이에 따른
식물 반응을 빠르고 비파괴적으로 추적할 수 있습니다.
        """)

        st.markdown("""
**🧪 극지 연구 샘플링 전략 개선**

잎 수와 지하부 길이 사이에 상관이 약하다면, 잎 수만으로 뿌리 발달을
판단하는 것은 부정확할 수 있습니다. 이런 결과는 현장 조사 시 어떤
지표를 우선시할지 결정하는 근거가 됩니다.
        """)

        st.markdown("""
**📋 교육용 생육 추적 프로젝트**

학생들이 개체를 파종부터 수확까지 추적하며 측정하는 프로젝트에서,
이 대시보드를 통해 지상부-지하부의 관계를 직접 시각화할 수 있습니다.
상관계수와 추세선을 함께 보면서 통계적 사고도 익힐 수 있습니다.
        """)

        # ── 생육 결과 다운로드 ────────────────────────────────────────────
        if not growth_df.empty and "지상부 길이" in growth_df.columns:
            st.divider()
            st.subheader("📥 생육 결과 데이터 다운로드")
            dl_df = growth_df.copy()
            csv_bytes = dl_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 생육 결과 CSV 다운로드 (utf-8-sig)",
                data=csv_bytes,
                file_name="극지식물_생육결과.csv",
                mime="text/csv",
            )

        # ── 가장 강한 상관 카드 ───────────────────────────────────────────
        if growth_df.empty:
            st.info("생육 CSV를 업로드하면 상관 요약 카드를 볼 수 있어요.")
        else:
            candidate_x = ["잎 수", "지상부 길이", "지상부 질량", "생중량"]
            candidate_y = ["지하부 길이", "지하부 질량"]
            best = None
            best_r = -1.0
            for xl in candidate_x:
                if xl not in growth_df.columns:
                    continue
                for yl in candidate_y:
                    if yl not in growth_df.columns:
                        continue
                    valid = growth_df[[xl, yl]].dropna()
                    if len(valid) < 2:
                        continue
                    r = np.corrcoef(valid[xl], valid[yl])[0, 1]
                    if abs(r) > abs(best_r):
                        best_r = r
                        best = (xl, yl)
            if best:
                xl, yl = best
                st.metric(
                    label=f"가장 강한 상관: {xl} ↔ {yl}",
                    value=f"r = {best_r:.3f}",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
