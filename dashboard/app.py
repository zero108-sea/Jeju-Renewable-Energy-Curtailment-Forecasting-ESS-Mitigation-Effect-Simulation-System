"""제주 계통 잉여 위험 예측 — 시연 대시보드.

  .venv/bin/streamlit run dashboard/app.py

[설계에서 고집한 것 세 가지]
1) **절대 확률을 숫자로 노출하지 않는다.** 경로마다 확률 척도가 다르고(같은 임계값 0.03에서
   발화율이 3.6~33.8%로 9배 벌어진다), 신뢰도 곡선이 단조롭게 어긋나 있어 "0.3 = 30%"가
   성립하지 않는다. 그래서 순위 밴드(상위 5% / 상위 20% / 그 외)로만 보여준다.
   `operational_threshold_reliable`이 참인 경로에서만 임계값 판정을 덧붙인다.
2) **ESS 패널을 운영 화면에서 분리한다.** 운영자가 보는 하루전 판단과, 설비 검토용
   민감도 분석은 의사결정 주체와 시점이 다르다. 한 화면에 섞으면 ESS 슬라이더가
   예측 신뢰도처럼 읽힌다.
3) **서버를 띄우지 않고 직접 호출한다.** 시연 중 FastAPI 프로세스가 죽으면 복구할 시간이
   없다. 같은 서비스 함수를 in-process로 부르므로 동작이 API와 동일하다.

[일부러 넣지 않은 것]
제어량(MWh) 절대값 예측 카드. 2단계 회귀 R²가 0.068이고 산포가 실측의 0.38배다 —
제품 정의가 "몇 MWh"가 아니라 "어느 시간이 위험한가"이므로 화면도 그래야 한다.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.schemas import EssSimulateRequest, PredictRequest
from app.serving_config import PATH_KEYS, band_floor
from app.services import ess_simulation
from app.services.predict import predict

st.set_page_config(page_title="제주 계통 잉여 위험 예측", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# 색 — 위험 등급은 '계열'이 아니라 '상태'다
# ---------------------------------------------------------------------------
# 등급은 정체성(어느 발전소인가)이 아니라 상태(얼마나 위험한가)를 뜻하므로 계열색이 아니라
# 예약된 상태색을 쓴다. 두 색은 검증기를 통과했다 (흰 배경, light):
#   CVD 분리 ΔE 13.9 (deutan) · 일반시야 ΔE 15.7 — 전 항목 PASS
# '낮음'은 상태가 아니라 '상태 없음'이라 중립 회색이다 — 상태색으로 칠하면 '안전함'이라는
# 없는 의미가 생긴다.
#
# 주의: serious(#ec835a)는 흰 배경에서 명도대비 2.64:1로 3:1 미만이다. 그래서 색만으로
# 의미를 싣지 않는다 — 아이콘·범례·등급표를 항상 함께 둔다(이 파일 아래 세 곳 전부).
RISK_CRITICAL = "#d03b3b"   # 상태색 critical
RISK_SERIOUS = "#ec835a"    # 상태색 serious
NEUTRAL = "#898781"         # 크롬 muted — 상태 없음

# 차트 크롬 — 격자·축은 배경에서 한 단계만 떨어뜨린다(실선 헤어라인, 점선 금지)
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_MUTED = "#898781"

BAND_STYLE = {
    "상위 5%": (RISK_CRITICAL, "매우 높음"),
    "상위 20%": (RISK_SERIOUS, "높음"),
    "그 외": (NEUTRAL, "낮음"),
}
BAND_ORDER = ["상위 5%", "상위 20%", "그 외"]
BAND_ICON = {"상위 5%": "⬤", "상위 20%": "◐", "그 외": "○"}


def _axis(title: str, **kw) -> alt.Axis:
    """공통 축 — 눈금과 격자를 배경 쪽으로 눌러 데이터가 앞에 오게 한다."""
    return alt.Axis(title=title, gridColor=GRID, gridWidth=1, domainColor=AXIS,
                    tickColor=AXIS, labelColor=INK_MUTED, titleColor=INK_MUTED,
                    labelFontSize=11, titleFontSize=11, **kw)


# ---------------------------------------------------------------------------
# 데이터 취득
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_weather(target: date, use_kim: bool) -> tuple[list[dict], str, str | None]:
    """기상청 예보 -> weather 24개. 실패하면 사유를 돌려주고 화면에서 표본으로 대체한다."""
    from app.services.kma_forecast import fetch
    src: list[str] = []
    try:
        w = fetch(target, radiation_source=src, use_kim=use_kim)
        return w, src[0] if src else "unknown", None
    except Exception as e:                                        # noqa: BLE001
        return [], "", f"{type(e).__name__}: {e}"


def sample_weather(target: date) -> list[dict]:
    """인증키가 없는 환경에서도 화면이 돌아야 한다 — 청천일사량 기반 합성 입력.

    **예보가 아니라 합성값이다.** 화면에 그 사실을 반드시 띄운다.
    """
    from app.solar_radiation import clear_sky_mj
    dt = pd.Series(pd.date_range(f"{target} 01:00", periods=24, freq="h"))
    cs = clear_sky_mj(dt, "184")
    return [{"hour": h, "temp": round(18 + 6 * np.sin((h - 9) / 24 * 2 * np.pi), 1),
             "cloud": 4.0, "wind_speed": round(5.5 + 2 * np.sin(h / 6), 1),
             "solar_rad": round(float(cs[h - 1]) * 0.62, 3)}
            for h in range(1, 25)]


def run_ess(req: EssSimulateRequest):
    """main.py의 /ess/simulate와 같은 분기. 두 곳에서 갈리면 화면과 API가 달라진다."""
    if req.method == "naive_upper_bound":
        return ess_simulation.simulate_naive_upper_bound(
            sum(1 for v in req.hourly_curtailment_mwh if v > 0),
            req.rated_power_mw, sum(req.hourly_curtailment_mwh))
    if req.method == "hourly_capped":
        return ess_simulation.simulate_hourly_capped(req.hourly_curtailment_mwh, req.rated_power_mw)
    return ess_simulation.simulate_with_storage(
        req.hourly_curtailment_mwh, req.rated_power_mw,
        req.energy_capacity_mwh or req.rated_power_mw * 4.0,
        hour_of_day=req.hour_of_day, round_trip_efficiency=req.round_trip_efficiency,
        soc_min=req.soc_min, soc_max=req.soc_max)


def band_of(rank: int, n: int) -> str:
    if rank < max(1, round(n * 0.05)):
        return "상위 5%"
    if rank < max(1, round(n * 0.20)):
        return "상위 20%"
    return "그 외"


def apply_bands(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, float, bool]:
    """일내 순위 + 바닥값으로 등급을 만든다.

    [왜 바닥값이 필요한가]
    순위는 정의상 척도를 지운다. 제어가 전혀 없는 날에도 1등은 존재하므로, 순위만 쓰면
    무제어일에도 '매우 높음'이 뜬다 — 2023년 테스트에서 무제어일 248일 **전부**가 그랬다.
    바닥값 아래는 전부 '낮음'으로 눌러 '오늘은 위험 시간 없음'이 표시될 수 있게 한다.
    2023년 기준 주경로에서 오경보일 100% -> 19%, 정밀도 28.5% -> 66.8%.

    신뢰 불가 경로(wind 단독)에서는 바닥값을 적용하지 않는다 — 그 경로는 제어일 판별을
    못 해서 바닥값을 올리면 제어일을 통째로 놓친다(README '밴드 바닥값').
    """
    floor, reliable = band_floor(key)
    order = df["curtailment_probability"].rank(ascending=False, method="first").astype(int) - 1
    bands = [band_of(r, len(df)) for r in order]
    if reliable:
        bands = [b if p >= floor else "그 외"
                 for b, p in zip(bands, df["curtailment_probability"])]
    return df.assign(밴드=bands), floor, reliable


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

st.sidebar.header("예측 조건")
target = st.sidebar.date_input("거래일", value=date.today() + timedelta(days=1))
energy = st.sidebar.radio("에너지원", ["wind", "solar"], format_func=lambda x: {"wind": "풍력", "solar": "태양광"}[x])
use_demand = st.sidebar.checkbox("수요예측 사용", value=True,
                                 help="수요를 넣으면 침투율 경로로 전환돼 성능이 크게 오른다 "
                                      "(풍력 PR-AUC 0.391 → 0.645)")
demand_level = st.sidebar.slider("수요 수준 (MW)", 400, 1000, 620, 10,
                                 disabled=not use_demand,
                                 help="시연용 단일 값. 실제 운영에서는 24시간 수요예측 곡선을 넣는다")
st.sidebar.divider()
use_kim = st.sidebar.checkbox("KIM 일사량 예보 사용", value=True,
                             help="끄면 청천일사량 기반 추정으로 폴백한다. "
                                  "정보 손실 +2.99%p가 이 차이다")
st.sidebar.caption("기상청 인증키(`KMA_AUTH_KEY`)가 없으면 합성 입력으로 화면만 보여줍니다.")


# ---------------------------------------------------------------------------
# 예측 실행
# ---------------------------------------------------------------------------

weather, rad_src, err = fetch_weather(target, use_kim)
synthetic = not weather
if synthetic:
    weather = sample_weather(target)

req = PredictRequest(
    energy_type=energy, region="제주", target_date=target, weather=weather,
    demand_forecast_mw=[float(demand_level)] * 24 if use_demand else None)
res = predict(req)

df = pd.DataFrame([h.model_dump() for h in res.hourly])
def path_from_model(model_used: str) -> str:
    """서버가 실제로 쓴 모델 이름에서 경로 키를 얻는다.

    화면에서 조건을 다시 조합해 추측하면 서버의 자동 전환(태양광 기상값이 오면 계통 전체
    침투율 모델로 바뀐다)과 어긋난다. 바닥값이 다른 경로 것으로 적용되면 조용히 틀린다.
    """
    k = model_used.removeprefix("classifier_")
    for suf in ("_calibrated_sigmoid", "_calibrated_isotonic"):
        k = k.removesuffix(suf)
    k = k.removesuffix("_lr")
    if k not in PATH_KEYS:
        raise ValueError(f"알 수 없는 경로: {model_used} -> {k}")
    return k


PATH = path_from_model(res.model_used)
df, FLOOR, FLOOR_OK = apply_bands(df, PATH)

tab_ops, tab_ess, tab_val = st.tabs(["하루전 위험 예측", "ESS 완화효과 (설비 검토)", "검증 리포트"])


# ---------------------------------------------------------------------------
# 탭 1 — 운영 화면
# ---------------------------------------------------------------------------

with tab_ops:
    st.subheader(f"{target} · {'풍력' if energy == 'wind' else '태양광'} · 시간별 잉여 위험")

    if synthetic:
        st.error(f"**합성 입력입니다 — 예보가 아닙니다.** 기상청 예보를 받지 못했습니다: {err}\n\n"
                 "화면 동작 확인용이며 이 등급은 실제 예측이 아닙니다.")
    elif rad_src == "kim_l010":
        st.success("일사량: KIM 국지예보모델 1.5km **예보값** · 그 외 기온·풍속·하늘상태는 단기예보")
    else:
        st.warning(f"일사량: **추정값으로 폴백** ({rad_src}). 정보 손실 +2.99%p가 남아 있습니다.")

    if res.note and "학습 분포를 벗어났" in (res.note or ""):
        st.warning(res.note)

    c1, c2 = st.columns(2)
    top5 = df[df["밴드"] == "상위 5%"]["hour"].tolist()
    c1.metric("매우 높음", f"{len(top5)}시간",
              ", ".join(f"{h}시" for h in top5) or "오늘은 없음", delta_color="off")
    c2.metric("발전량 예측 합계", f"{df['generation_forecast_mwh'].sum():,.0f} MWh")
    # 모델 이름은 수치가 아니다 — metric 타일에 넣으면 지표처럼 읽힌다
    st.caption(f"사용 모델 `{res.model_used}`")

    # 등급은 색만으로 싣지 않는다 — 아이콘을 레이블에 붙여 범례·표·툴팁이 같은 말을 하게 한다
    chart_df = df.assign(
        등급=[f"{BAND_ICON[b]} {BAND_STYLE[b][1]}" for b in df["밴드"]],
        발전량=df["generation_forecast_mwh"].round(1))
    labels = [f"{BAND_ICON[b]} {BAND_STYLE[b][1]}" for b in BAND_ORDER]
    colors = [BAND_STYLE[b][0] for b in BAND_ORDER]

    base = alt.Chart(chart_df)
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        # paddingInner가 막대 사이 간격을 만든다 — 테두리를 그려 분리하지 않는다
        x=alt.X("hour:O", axis=_axis("시각 (구간의 끝)", labelAngle=0),
                scale=alt.Scale(paddingInner=0.18)),
        y=alt.Y("발전량:Q", axis=_axis("발전량 예측 (MWh)")),
        color=alt.Color("등급:N", title=None,
                        scale=alt.Scale(domain=labels, range=colors),
                        legend=alt.Legend(orient="top", direction="horizontal",
                                          labelColor=INK_MUTED, symbolType="square")),
        tooltip=[alt.Tooltip("hour:O", title="시각"),
                 alt.Tooltip("발전량:Q", title="발전량 예측 (MWh)"),
                 alt.Tooltip("등급:N", title="위험 등급")])
    # 직접 레이블은 선택적으로 — 모든 막대에 숫자를 붙이면 읽히지 않는다. 최상위 등급만.
    peak = base.transform_filter(alt.datum.밴드 == "상위 5%").mark_text(
        dy=-7, fontSize=11, color=RISK_CRITICAL, fontWeight="bold").encode(
        x=alt.X("hour:O", scale=alt.Scale(paddingInner=0.18)), y="발전량:Q",
        text=alt.Text("발전량:Q", format=".0f"))
    st.altair_chart((bars + peak).properties(height=260).configure_view(strokeWidth=0),
                    use_container_width=True)

    if FLOOR_OK:
        st.markdown("**시간별 위험 등급** — 그날 24시간 안에서의 순위 밴드이되, "
                    f"**바닥값({FLOOR:.3f}) 미만은 표시하지 않습니다.** "
                    "그래서 위험한 시간이 없는 날은 전부 '낮음'으로 나옵니다.")
    else:
        st.warning(f"**이 경로({PATH})는 바닥값을 쓸 수 없습니다.** 제어일 판별 성능이 낮아"
                   "(PR-AUC 0.391) 바닥값을 올리면 제어일을 통째로 놓칩니다. 지금 화면은 "
                   "하루 안 순위만 쓰므로 **위험이 없는 날에도 등급이 표시됩니다.** "
                   "수요예측을 켜면 이 문제가 없는 경로로 전환됩니다.")
    view = df[["hour", "generation_forecast_mwh", "밴드"]].copy()
    view.columns = ["시각", "발전량 예측 (MWh)", "위험 등급"]
    # 표·범례·툴팁이 같은 아이콘을 쓴다 — 색을 못 읽어도 등급이 전달되어야 한다
    view["위험 등급"] = [f"{BAND_ICON[b]} {BAND_STYLE[b][1]} ({b})" for b in df["밴드"]]
    view["발전량 예측 (MWh)"] = view["발전량 예측 (MWh)"].round(1)
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"시각": st.column_config.NumberColumn(format="%d시")})

    with st.expander("왜 확률(%)을 보여주지 않는가"):
        st.markdown(f"""
독립적인 이유가 셋이고, 하나만으로는 근거가 약합니다.

1. **경로마다 확률 척도가 다릅니다.** 같은 운영 임계값 0.03을 5개 경로에 적용하면
   발화율이 3.6%에서 33.8%까지 **9배** 벌어집니다.
2. **신뢰도 곡선이 단조롭게 어긋나 있습니다.** 0.3을 30%로 읽는 해석이 성립하지 않습니다.
3. **태양광은 보정 구간 양성이 20개**뿐입니다. 이 표본으로 정한 임계값은 표본 잡음입니다.

현재 경로의 선정 임계값은 `{res.operational_threshold}`이고
신뢰 가능 여부는 **`{res.operational_threshold_reliable}`** 입니다.
거짓이면 임계값 판정 대신 등급을 쓰는 것이 설계 의도입니다.
""")

    if res.note and "학습 분포" not in (res.note or ""):
        st.caption(res.note)


# ---------------------------------------------------------------------------
# 탭 2 — ESS (운영 화면과 분리)
# ---------------------------------------------------------------------------

with tab_ess:
    st.subheader("ESS 완화효과 — 설비 용량 민감도")
    st.info("**이 패널은 하루전 운영 판단과 분리돼 있습니다.** 여기서 쓰는 제어량은 "
            "2023년 **실측** 출력제어량이고, 왼쪽 예측값이 아닙니다 — 예측 기댓값을 넣으면 "
            "흡수율이 실측 36.8% 대비 80.6%로 과대평가됩니다(README '알려진 한계').")

    s1, s2 = st.columns(2)
    mw = s1.slider("정격출력 (MW)", 10, 200, 65, 5)
    hours = s2.slider("저장시간 (h)", 1, 8, 4, 1, help="저장용량 = 정격출력 × 저장시간")
    e1, e2, e3 = st.columns(3)
    eff = e1.slider("왕복효율", 0.80, 0.97, 0.90, 0.01)
    soc_lo = e2.slider("SoC 하한", 0.0, 0.3, 0.10, 0.05)
    soc_hi = e3.slider("SoC 상한", 0.7, 1.0, 0.90, 0.05)

    @st.cache_data(show_spinner=False)
    def actual_curtailment() -> tuple[list[float], list[int], str] | None:
        """2023년 실측 제어량. 없으면 None."""
        try:
            from app.data_prep import load_curtailment
            c = load_curtailment("wind")
            c = c[(c["dt"] >= "2023-01-01") & (c["dt"] < "2024-01-01")]
            if c.empty:
                return None
            h = pd.Series(pd.DatetimeIndex(c["dt"]).hour).replace(0, 24).tolist()
            return c["curtailment_mwh"].tolist(), h, "2023년 제주 풍력 실측"
        except Exception:                                         # noqa: BLE001
            return None

    got = actual_curtailment()
    if got is None:
        st.warning("실측 제어량 데이터를 찾지 못해 ESS 시뮬레이션을 건너뜁니다 "
                   "(`data/`에 전력거래소 출력제어 파일이 필요합니다).")
    else:
        series, hod, label = got
        rows = []
        for method, name in (("storage_constrained", "저장용량 제약 (정식)"),
                             ("hourly_capped", "정격출력만 (이론 상한)")):
            r = run_ess(EssSimulateRequest(
                hourly_curtailment_mwh=series, hour_of_day=hod, rated_power_mw=float(mw),
                energy_capacity_mwh=float(mw * hours), round_trip_efficiency=eff,
                soc_min=soc_lo, soc_max=soc_hi, method=method))
            rows.append({"방식": name, "흡수율": r.absorption_rate,
                         "흡수량 (MWh)": r.total_absorbed_mwh,
                         "미흡수 (MWh)": r.total_curtailment_mwh - r.total_absorbed_mwh,
                         "가득 찬 시간": r.hours_full, "환산 사이클": r.annual_cycles})
        t = pd.DataFrame(rows)
        gap = (t.loc[1, "흡수율"] - t.loc[0, "흡수율"]) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("흡수율 (정식)", f"{t.loc[0, '흡수율'] * 100:.1f}%",
                  f"이론 상한 대비 −{gap:.1f}%p", delta_color="inverse")
        m2.metric("미흡수 잔량", f"{t.loc[0, '미흡수 (MWh)']:,.0f} MWh",
                  help="이 잔량이 '언제 쓸 것인가'를 푸는 이 시스템의 존재 이유입니다")
        m3.metric("설비", f"{mw} MW / {mw * hours} MWh")

        st.dataframe(t.assign(흡수율=(t["흡수율"] * 100).round(1)).round(1),
                     hide_index=True, width="stretch")
        st.caption(f"제어량 출처: {label} · 총 {sum(series):,.0f} MWh. "
                   "두 방식의 차이가 곧 '정격출력만 보면 설비가 있으면 다 해결된다고 읽히는' 오해의 크기입니다.")

        # ── 민감도 곡선 ────────────────────────────────────────────────────
        # 이 패널의 이름이 '용량 민감도'인데 그동안 한 점만 보여줬다. 설비 검토의 질문은
        # "65MW면 얼마?"가 아니라 "얼마를 더 쓰면 얼마가 더 오르나"이고, 그건 곡선이라야 보인다.
        # 두 계열 모두 %라 축이 하나다 — 축을 둘로 쪼개지 않는다.
        @st.cache_data(show_spinner="용량별 흡수율을 계산하는 중…")
        def sweep(series: tuple, hod: tuple, cap_h: int, eff: float,
                  lo: float, hi: float) -> pd.DataFrame:
            out = []
            for m in range(10, 201, 10):
                for method, name in (("storage_constrained", "저장용량 제약"),
                                     ("hourly_capped", "정격출력만 (이론 상한)")):
                    r = run_ess(EssSimulateRequest(
                        hourly_curtailment_mwh=list(series), hour_of_day=list(hod),
                        rated_power_mw=float(m), energy_capacity_mwh=float(m * cap_h),
                        round_trip_efficiency=eff, soc_min=lo, soc_max=hi, method=method))
                    out.append({"정격출력": m, "방식": name, "흡수율": r.absorption_rate * 100})
            return pd.DataFrame(out)

        sw = sweep(tuple(series), tuple(hod), hours, eff, soc_lo, soc_hi)
        CAT = ["#2a78d6", "#eb6834"]          # 계열색 1·2 — 전 항목 PASS (ΔE 24.7 protan)
        names = ["저장용량 제약", "정격출력만 (이론 상한)"]

        line = alt.Chart(sw).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=28)).encode(
            x=alt.X("정격출력:Q", axis=_axis("정격출력 (MW)")),
            y=alt.Y("흡수율:Q", axis=_axis("흡수율 (%)"), scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("방식:N", title=None, scale=alt.Scale(domain=names, range=CAT),
                            legend=alt.Legend(orient="top", direction="horizontal",
                                              labelColor=INK_MUTED)),
            tooltip=[alt.Tooltip("정격출력:Q", title="정격출력 (MW)"),
                     alt.Tooltip("방식:N", title="방식"),
                     alt.Tooltip("흡수율:Q", title="흡수율 (%)", format=".1f")])
        # 지금 슬라이더 위치를 세로선으로 — 곡선 위 어디에 서 있는지가 이 패널의 요점이다
        here = alt.Chart(pd.DataFrame({"정격출력": [mw]})).mark_rule(
            color=INK_MUTED, strokeWidth=1).encode(x="정격출력:Q")
        st.altair_chart((line + here).properties(height=260).configure_view(strokeWidth=0),
                        use_container_width=True)
        st.caption(f"세로선이 현재 설정({mw}MW)입니다. "
                   "두 곡선의 벌어짐이 저장용량 제약의 크기이고, 오른쪽으로 갈수록 "
                   "**곡선이 눕는 것**이 증설의 수확 체감입니다.")


# ---------------------------------------------------------------------------
# 탭 3 — 검증 리포트
# ---------------------------------------------------------------------------

with tab_val:
    st.subheader("검증 리포트 — 학습 산출물 그대로")
    st.caption("아래 표는 `models/*.csv`를 읽어 그대로 표시합니다. 발표용으로 다시 타이핑한 값이 아닙니다.")

    from app.model_io import MODELS_DIR

    REPORTS = [
        ("기준선 대비", "baseline_comparison.csv",
         "모델이 지속성·기후값·이용률 단독보다 실제로 나은지. AUC는 불균형에서 둔감하므로 PR-AUC를 본다."),
        ("지표의 불확실성 (일 블록)", "metric_confidence_intervals.csv",
         "출력제어는 날짜 단위로 뭉친다. 시간 단위 리샘플은 구간을 1.6~2.1배 좁게 만든다."),
        ("풍력에 태양광 피처", "wind_solar_feature_experiment.csv",
         "핵심 차별점의 근거. 둘 다 넣으면 오히려 나빠진다(정보 중복)."),
        ("예보 입력 정보 손실", "forecast_degradation.csv",
         "손실 3.03%p 중 2.99%p가 일사량 추정 하나에서 나왔다."),
        ("대리 라벨 검증", "new_regime_proxy_validation.csv",
         "제도 전환 전후 겹치는 구간에서 대리 라벨과 실제 제어 기록을 대조한 결과."),
        ("신제도 구간 성능", "new_regime_eval.csv",
         "전환 직후 6개월 AUC 0.544 — 동전 던지기다. 사용 구간에서 제외했다."),
        ("제어량 2단계 손실함수 비교", "stage2_loss_comparison.csv",
         "5가지 대안이 전부 나빴다. 손실함수 문제가 아니라 정보 문제라는 근거."),
        ("실입력 관통 검사", "serving_real_input_verification.csv",
         "합성 입력이 아니라 실제 학습·테스트 구간 입력으로 5개 경로를 관통해 돌린 결과."),
    ]
    for title, fname, why in REPORTS:
        path = os.path.join(MODELS_DIR or "", fname)
        with st.expander(title, expanded=(fname == "baseline_comparison.csv")):
            st.caption(why)
            if os.path.exists(path):
                st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
            else:
                st.info(f"`{fname}`이 없습니다 — 해당 학습 스크립트를 먼저 실행하세요.")

    st.divider()
    st.markdown("""
**이 화면이 보여주지 않는 것** — 실제 예보 입력으로 낸 운영 성능. 예보 아카이브가 없어
소급 측정이 불가능하고(API 보관이 하루), 예보와 실측을 매일 축적하는 중입니다
(`scripts/kim_validation_log.py`). 현재 수치는 **상한**입니다.
""")
