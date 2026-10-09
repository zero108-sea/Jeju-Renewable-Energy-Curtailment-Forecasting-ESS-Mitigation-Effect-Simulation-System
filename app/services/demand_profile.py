"""24시간 수요 모양을 실측에서 만든다 (일평균 1.0으로 정규화).

[왜 평평한 값을 쓰면 안 되나]
시연 화면은 슬라이더 한 값을 24번 복제해 수요로 넣고 있었다. 그런데
**침투율 = 발전량 ÷ 수요**이고, 제주 수요는 하루 안에서 4시 0.84배 ~ 19시 1.18배로
40% 움직인다. 분모를 평평하게 두면 침투율이 시간대별로 체계적으로 왜곡된다 —
특히 태양광이 몰리는 정오 부근은 실제보다 수요를 높게 잡아 침투율을 **과소**평가한다.

[왜 예보가 아니라 실측 프로파일인가]
수요예보는 이 시스템의 산출물이 아니다 — 운영에서는 Backend가 받은 예보를 그대로 넣는다.
시연에 필요한 것은 '그럴듯한 모양'이고, 그 모양의 가장 정직한 출처는 같은 달·같은 요일형의
실측이다. 그래서 **수준은 사용자가, 모양은 실측이** 정한다. 화면에는 출처를 명시한다.

평일/주말은 정규화 모양 차이가 최대 4.8%로 작지만 groupby 키 하나라 나눈다.
"""
from __future__ import annotations

import pandas as pd

HOURS = 24
# 표본이 한 주(168시간)도 안 되면 그 달의 '모양'이라고 부를 수 없다 — 평탄으로 후퇴한다.
MIN_SAMPLE_HOURS = 24 * 7


def profile(month: int, weekend: bool) -> tuple[list[float], str]:
    """(24개 배수, 출처 문구). 실측을 못 읽으면 평탄 프로파일과 그 사유를 돌려준다.

    배수의 평균은 정확히 1.0이다 — 그래야 사용자가 고른 수준이 그대로 '일평균 수요'가 된다.
    """
    try:
        from app.data_prep import load_demand_actual
        d = load_demand_actual()
        d = d[(d["dt"].dt.month == month) & ((d["dt"].dt.dayofweek >= 5) == weekend)]
        if len(d) < MIN_SAMPLE_HOURS:
            raise ValueError(f"표본 부족 ({len(d)}행 < {MIN_SAMPLE_HOURS})")
        # 이 저장소는 h시를 '구간의 끝'으로 쓴다 — 0시는 24시다
        s = d.groupby(d["dt"].dt.hour.replace(0, HOURS))["demand_mw"].mean()
        s = s.reindex(range(1, HOURS + 1)).interpolate().bfill().ffill()
        if not s.notna().all() or s.mean() <= 0:
            raise ValueError("프로파일에 결측이 남았습니다")
        src = (f"{month}월 {'주말' if weekend else '평일'} 실측 평균 "
               f"({d['dt'].dt.year.min()}~{d['dt'].dt.year.max()}, {len(d):,}시간)")
        return (s / s.mean()).round(4).tolist(), src
    except Exception as e:                                        # noqa: BLE001
        return [1.0] * HOURS, f"평탄 (실측 프로파일 없음: {type(e).__name__}: {e})"


def curve(level_mw: float, month: int, weekend: bool) -> tuple[list[float], str]:
    """일평균 level_mw인 24시간 수요 곡선."""
    p, src = profile(month, weekend)
    return [round(level_mw * x, 1) for x in p], src
