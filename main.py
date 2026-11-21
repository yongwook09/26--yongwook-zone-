

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
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

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
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ═══════════════════════════════════════
# Tab 1: 실험 개요
# ═══════════════════════════════════════
with tab1:
    try:
        st.markdown(f"### 🏫 {school_name} 극지식물 실험")

        st.markdown(
            """
**연구 배경 및 목적**

극지식물은 척박한 환경에서도 생존하는 강한 생명력을 가진 식물입니다.
우리 학교에서는 극지식물 개체를 직접 재배하며 개체별로 생육 차이가 얼마나 나는지 관찰했습니다.

이 대시보드에서는 다음을 분석합니다:

- 개체별 **생중량**(전체 무게)의 분포와 차이
- **잎 수**, **지상부 길이**와 생중량의 관계
- 생중량이 높은 개체와 낮은 개체의 특징 비교

---
"""
        )

        st.subheader("📋 데이터 요약")

        col_summary = st.columns(3)

        with col_summary[0]:
            st.markdown("#### 생육 데이터")
            if growth_df.empty:
                st.info("생육 CSV를 업로드해 주세요.")
            else:
                n_individuals = growth_df["개체번호"].nunique() if "개체번호" in growth_df.columns else len(growth_df)
                available = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                st.write(f"**총 개체 수:** {n_individuals}개체")
                st.write(f"**측정 항목:** {', '.join(available)}")

        with col_summary[1]:
            st.markdown("#### 환경 데이터")
            if env_df.empty:
                st.info("환경 CSV를 업로드해 주세요.")
            else:
                if "time" in env_df.columns:
                    t_min = env_df["time"].min()
                    t_max = env_df["time"].max()
                    st.write(f"**측정 기간:** {t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}")
                else:
                    st.write("**측정 기간:** 정보 없음")
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                st.write(f"**측정 항목:** {', '.join(env_cols) if env_cols else '없음'}")

        with col_summary[2]:
            st.markdown("#### 📌 용어 설명")
            st.write(
                """
- **평균**: 모든 값을 더해 개수로 나눈 대표값
- **표준편차**: 값이 평균에서 얼마나 퍼져 있는지 나타내는 값
- **CV(변동계수)**: 표준편차 ÷ 평균 × 100 → 단위가 다른 것끼리 비교 가능
- **상관계수(r)**: 두 변수의 관계를 -1~1 사이로 나타냄 (1에 가까울수록 함께 증가)
"""
            )

        st.subheader("📈 주요 지표")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            if growth_df.empty:
                st.info("데이터 없음")
            else:
                n = growth_df["개체번호"].nunique() if "개체번호" in growth_df.columns else len(growth_df)
                st.metric("총 개체 수", f"{n}개체")

        with m2:
            if env_df.empty or "temperature" not in env_df.columns:
                st.info("환경 데이터 없음")
            else:
                t_avg = env_df["temperature"].dropna().mean()
                if pd.notna(t_avg):
                    st.metric("평균 온도", f"{t_avg:.1f}°C")
                else:
                    st.info("온도 데이터 없음")

        with m3:
            if env_df.empty or "humidity" not in env_df.columns:
                st.info("환경 데이터 없음")
            else:
                h_avg = env_df["humidity"].dropna().mean()
                if pd.notna(h_avg):
                    st.metric("평균 습도", f"{h_avg:.1f}%")
                else:
                    st.info("습도 데이터 없음")

        with m4:
            if growth_df.empty or "생중량" not in growth_df.columns:
                st.info("생중량 데이터 없음")
            else:
                gw_avg = growth_df["생중량"].dropna().mean()
                if pd.notna(gw_avg):
                    st.metric("평균 생중량", f"{gw_avg:.2f}g")
                else:
                    st.info("생중량 데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════
# Tab 2: 환경 데이터
# ═══════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info(f"{school_name}의 환경 CSV를 업로드하면 여기에 환경 데이터가 표시됩니다.")
        else:
            st.subheader("🌡️ 환경 측정 요약")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                summary_rows = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) == 0:
                        continue
                    summary_rows.append({
                        "측정 항목": c,
                        "평균": round(s.mean(), 2),
                        "최소": round(s.min(), 2),
                        "최대": round(s.max(), 2),
                        "측정 횟수": len(s),
                    })
                if summary_rows:
                    st.dataframe(pd.DataFrame(summary_rows), width="stretch")
                else:
                    st.info("요약할 환경 데이터가 없습니다.")
            else:
                st.info("환경 데이터에 온도·습도·pH·EC 열이 없습니다.")

            if "time" in env_df.columns:
                st.subheader("📅 날짜별 환경 변화")
                env_df_daily = env_df.copy()
                env_df_daily["date"] = env_df_daily["time"].dt.date

                col_temp, col_hum, col_etc = st.columns(3)

                with col_temp:
                    if "temperature" in env_df.columns:
                        daily_temp = env_df_daily.groupby("date")["temperature"].mean().dropna()
                        if len(daily_temp) > 0:
                            fig_temp = go.Figure()
                            fig_temp.add_trace(go.Scatter(
                                x=daily_temp.index.astype(str),
                                y=daily_temp.values,
                                mode="lines+markers",
                                name="평균 온도",
                                line=dict(color="#EF553B", width=2),
                                marker=dict(size=6),
                            ))
                            fig_temp.update_layout(
                                title="일별 평균 온도",
                                xaxis_title="날짜",
                                yaxis_title="온도 (°C)",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=250,
                                margin=dict(l=40, r=20, t=40, b=40),
                            )
                            st.plotly_chart(fig_temp, width="stretch")
                        else:
                            st.info("온도 데이터 없음")
                    else:
                        st.info("온도 열이 없습니다.")

                with col_hum:
                    if "humidity" in env_df.columns:
                        daily_hum = env_df_daily.groupby("date")["humidity"].mean().dropna()
                        if len(daily_hum) > 0:
                            fig_hum = go.Figure()
                            fig_hum.add_trace(go.Scatter(
                                x=daily_hum.index.astype(str),
                                y=daily_hum.values,
                                mode="lines+markers",
                                name="평균 습도",
                                line=dict(color="#00CC96", width=2),
                                marker=dict(size=6),
                            ))
                            fig_hum.update_layout(
                                title="일별 평균 습도",
                                xaxis_title="날짜",
                                yaxis_title="습도 (%)",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=250,
                                margin=dict(l=40, r=20, t=40, b=40),
                            )
                            st.plotly_chart(fig_hum, width="stretch")
                        else:
                            st.info("습도 데이터 없음")
                    else:
                        st.info("습도 열이 없습니다.")

                with col_etc:
                    ph_ec_cols = [c for c in ["ph", "ec"] if c in env_df.columns]
                    if ph_ec_cols:
                        daily_etc = env_df_daily.groupby("date")[ph_ec_cols].mean().dropna()
                        if len(daily_etc) > 0:
                            fig_etc = make_subplots(
                                rows=len(ph_ec_cols),
                                cols=1,
                                subplot_titles=[f"일별 평균 {c.upper()}" for c in ph_ec_cols],
                                vertical_spacing=0.15,
                            )
                            colors = {"ph": "#FF9900", "ec": "#1976D2"}
                            for i, c in enumerate(ph_ec_cols):
                                fig_etc.add_trace(go.Scatter(
                                    x=daily_etc.index.astype(str),
                                    y=daily_etc[c].values,
                                    mode="lines+markers",
                                    name=c.upper(),
                                    line=dict(color=colors.get(c, "#636EFA"), width=2),
                                    marker=dict(size=5),
                                ), row=i + 1, col=1)
                            fig_etc.update_layout(
                                title=f"일별 {'·'.join(c.upper() for c in ph_ec_cols)} 변화",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=120 * len(ph_ec_cols) + 60,
                                margin=dict(l=40, r=20, t=40, b=40),
                            )
                            fig_etc.update_xaxes(title_text="날짜", row=len(ph_ec_cols), col=1)
                            for i in range(len(ph_ec_cols)):
                                fig_etc.update_yaxes(
                                    title_text=f"{ph_ec_cols[i].upper()}",
                                    row=i + 1,
                                    col=1,
                                )
                            st.plotly_chart(fig_etc, width="stretch")
                        else:
                            st.info("pH·EC 데이터 없음")
                    else:
                        st.info("pH·EC 열이 없습니다.")

            with st.expander("📄 환경 데이터 원본 보기 및 다운로드"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name}_환경데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════
# Tab 3: 생육 결과
# ═══════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info(f"{school_name}의 생육 CSV를 업로드하면 여기에 생육 분석 결과가 표시됩니다.")
        else:
            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")

            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) > 0:
                    mean_gw = gw.mean()
                    max_idx = gw.idxmax()
                    min_idx = gw.idxmin()
                    max_id = growth_df.loc[max_idx, "개체번호"] if "개체번호" in growth_df.columns else f"#{max_idx}"
                    min_id = growth_df.loc[min_idx, "개체번호"] if "개체번호" in growth_df.columns else f"#{min_idx}"

                    q75 = gw.quantile(0.75)
                    q25 = gw.quantile(0.25)
                    top_df = growth_df[growth_df["생중량"] >= q75]
                    bottom_df = growth_df[growth_df["생중량"] <= q25]

                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.metric("⭐ 평균 생중량", f"{mean_gw:.2f}g")
                    with k2:
                        st.metric("최대 생중량", f"{gw.max():.2f}g ({max_id})")
                    with k3:
                        st.metric("최소 생중량", f"{gw.min():.2f}g ({min_id})")
                    with k4:
                        cv_val = (gw.std() / gw.mean() * 100) if gw.mean() > 0 else 0
                        st.metric("변동계수(CV)", f"{cv_val:.1f}%")

                    # 상위/하위 25% 비교
                    st.markdown("---")
                    st.subheader("📊 생중량 상위 25% vs 하위 25% 비교")
                    comp_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                    if comp_cols and len(top_df) > 0 and len(bottom_df) > 0:
                        comp_rows = []
                        for c in comp_cols:
                            tv = top_df[c].dropna()
                            bv = bottom_df[c].dropna()
                            comp_rows.append({
                                "측정 항목": c,
                                "상위 25% 평균": round(tv.mean(), 2) if len(tv) > 0 else "-",
                                "하위 25% 평균": round(bv.mean(), 2) if len(bv) > 0 else "-",
                                "차이(상위-하위)": round(tv.mean() - bv.mean(), 2) if len(tv) > 0 and len(bv) > 0 else "-",
                                "상위 개체 수": len(top_df),
                                "하위 개체 수": len(bottom_df),
                            })
                        st.dataframe(pd.DataFrame(comp_rows), width="stretch")
                    else:
                        st.info("비교할 데이터가 충분하지 않습니다.")
                else:
                    st.info("생중량 데이터가 없습니다.")
            else:
                st.info("생중량 열이 없습니다. (지상부 질량 + 지하부 질량으로 자동 계산됩니다)")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")
            bar_metrics = [m for m in selected_metrics if m in growth_df.columns]
            if bar_metrics:
                n_per_row = 2
                for row_start in range(0, len(bar_metrics), n_per_row):
                    row_metrics = bar_metrics[row_start:row_start + n_per_row]
                    cols = st.columns(len(row_metrics))
                    for c_idx, metric in enumerate(row_metrics):
                        with cols[c_idx]:
                            values = growth_df[metric].dropna()
                            if len(values) == 0:
                                st.info(f"{metric} 데이터 없음")
                                continue
                            if "개체번호" in growth_df.columns:
                                ids = growth_df.loc[values.index, "개체번호"]
                            else:
                                ids = values.index
                            is_gw = (metric == "생중량")
                            fig_bar = go.Figure()
                            bar_colors = ["#636EFA"] * len(values)
                            if is_gw and len(values) >= 4:
                                q75 = values.quantile(0.75)
                                q25 = values.quantile(0.25)
                                for i, v in enumerate(values):
                                    if v >= q75:
                                        bar_colors[i] = "#EF553B"
                                    elif v <= q25:
                                        bar_colors[i] = "#00CC96"
                            fig_bar.add_trace(go.Bar(
                                x=[str(x) for x in ids],
                                y=values.tolist(),
                                marker_color=bar_colors,
                                hovertemplate=f"{metric}<br>개체: %x<br>값: %y<extra></extra>",
                            ))
                            title = f"⭐ {metric}" if is_gw else metric
                            fig_bar.update_layout(
                                title=title,
                                xaxis_title="개체번호",
                                yaxis_title=metric,
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=300,
                                margin=dict(l=40, r=20, t=40, b=40),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("표시할 측정 항목이 없습니다.")

            # ── 생중량 분포 (박스플롯) ──
            if "생중량" in growth_df.columns:
                st.subheader("📦 생중량 분포")
                gw = growth_df["생중량"].dropna()
                if len(gw) > 0:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=gw.tolist(),
                        name="생중량",
                        fillcolor="#636EFA",
                        line=dict(color="#636EFA"),
                        hovertemplate="생중량: %y g<extra></extra>",
                    ))
                    fig_box.update_layout(
                        title="개체별 생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=300,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    col_stats = st.columns(4)
                    with col_stats[0]:
                        st.metric("평균", f"{gw.mean():.2f}g")
                    with col_stats[1]:
                        st.metric("표준편차", f"{gw.std():.2f}g")
                    with col_stats[2]:
                        st.metric("CV", f"{(gw.std() / gw.mean() * 100):.1f}%" if gw.mean() > 0 else "0%")
                    with col_stats[3]:
                        st.metric("범위", f"{gw.min():.2f} ~ {gw.max():.2f}g")
                else:
                    st.info("생중량 데이터 없음")
            else:
                st.info("생중량 열이 없습니다.")

            # ── 상관관계 분석 ──
            st.subheader("🔗 상관관계 분석")
            corr_pairs = [("잎 수", "생중량"), ("지상부 길이", "생중량")]
            available_pairs = [(x, y) for x, y in corr_pairs if x in growth_df.columns and y in growth_df.columns]

            if available_pairs:
                n_pairs = len(available_pairs)
                fig_corr = make_subplots(
                    rows=1,
                    cols=n_pairs,
                    subplot_titles=[f"{x} vs {y}" for x, y in available_pairs],
                    horizontal_spacing=0.12,
                )
                corr_rows = []
                for i, (x_col, y_col) in enumerate(available_pairs):
                    sub = growth_df[[x_col, y_col]].dropna()
                    if len(sub) >= 2:
                        r_val = sub[x_col].corr(sub[y_col])
                        fig_corr.add_trace(go.Scatter(
                            x=sub[x_col].tolist(),
                            y=sub[y_col].tolist(),
                            mode="markers",
                            marker=dict(size=8, color="#636EFA", line=dict(width=1, color="white")),
                            name=f"{x_col} vs {y_col}",
                            hovertemplate=f"{x_col}: %x<br>{y_col}: %y<br>r={r_val:.2f}<extra></extra>",
                        ), row=1, col=i + 1)

                        # 추세선 (np.polyfit)
                        mask = sub[x_col].notna() & sub[y_col].notna()
                        xc = sub[x_col][mask]
                        yc = sub[y_col][mask]
                        if len(xc) >= 2:
                            slope, intercept = np.polyfit(xc, yc, 1)
                            x_line = np.linspace(xc.min(), xc.max(), 100)
                            y_line = slope * x_line + intercept
                            fig_corr.add_trace(go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                line=dict(color="#EF553B", dash="dash", width=2),
                                name=f"추세선",
                                hoverinfo="skip",
                            ), row=1, col=i + 1)

                        fig_corr.update_xaxes(title_text=x_col, row=1, col=i + 1)
                        fig_corr.update_yaxes(title_text=y_col, row=1, col=i + 1)

                        corr_rows.append({
                            "X 변수": x_col,
                            "Y 변수": y_col,
                            "상관계수(r)": round(r_val, 3),
                            "데이터 수": len(sub),
                            "해석": (
                                "강한 양의 상관" if r_val > 0.7
                                else "약한 양의 상관" if r_val > 0.3
                                else "상관 거의 없음" if abs(r_val) < 0.3
                                else "약한 음의 상관" if r_val > -0.7
                                else "강한 음의 상관"
                            ),
                        })
                fig_corr.update_layout(
                    title="잎 수·지상부 길이 vs 생중량 상관관계",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=420,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_corr, width="stretch")

                if corr_rows:
                    st.markdown("#### 상관계수 요약")
                    st.dataframe(pd.DataFrame(corr_rows), width="stretch")
            else:
                st.info("상관관계 분석에 필요한 잎 수·지상부 길이·생중량 열이 부족합니다.")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 생육 데이터 원본 보기 및 다운로드"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name}_생육결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
