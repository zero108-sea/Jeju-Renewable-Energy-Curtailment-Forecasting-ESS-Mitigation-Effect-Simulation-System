"""KIM 일사량 예보의 정확도를 정방향으로 축적해 검증한다.

[왜 이 방식인가]
원래 계획은 테스트 구간(2025년)에 KIM 과거 예보를 넣어 정보 손실 재측정을 다시 돌리는
것이었다. 실측으로 확인해보니 **API가 과거 예보를 보관하지 않는다** — 2026-09-27 기준
응답이 오는 가장 오래된 분석시각이 하루 전이었다. 그래서 소급 측정은 불가능하고, 남은
방법은 하나다: 오늘부터 예보를 쌓고, 며칠 뒤 ASOS 실측이 올라오면 맞춰보는 것.

  collect  거래일 하루치 예보를 CSV에 덧붙인다 (KIM 값 + 지금 쓰는 추정값 둘 다)
  report   ASOS 실측이 있는 날에 대해 두 경로의 오차를 비교한다

매일 한 번 collect를 돌려야 한다. 예보를 놓친 날은 되돌릴 수 없다(보관이 없으므로).

  .venv/bin/python scripts/kim_validation_log.py collect            # 내일 거래일
  .venv/bin/python scripts/kim_validation_log.py collect 2026-09-29
  .venv/bin/python scripts/kim_validation_log.py collect --force    # 이미 받았어도 다시
  .venv/bin/python scripts/kim_validation_log.py report

이미 24시간을 받아둔 (거래일, 분석시각)이면 **API를 호출하지 않고 끝낸다.** 스케줄을 하루 네 번
걸어두기 때문이다 — 맥이 10시에 잠들어 있으면 launchd가 그 실행을 건너뛰므로(실측: runs=0)
뒤의 시각이 받아내야 하고, 이미 받았으면 조용히 끝나야 한다.

[읽는 법]
ASOS 실측을 정답으로 두고 두 경로의 MAE를 비교한다. KIM이 더 낮으면 교체가 옳았다는
직접 증거가 되고, 정보 손실 +2.99%p 중 얼마를 회수하는지 추정할 근거가 된다. 표본이
며칠뿐이면 그 수치를 결론으로 쓸 수 없다 — 날 수를 항상 함께 보고한다.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from app.services import kim_forecast as K

LAT, LON = 33.5141, 126.5297
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "kim_forecast_log.csv")
COLS = ["target_date", "hour", "tmfc", "kim_mj", "kim_inst_mj", "est_mj",
        "wind80", "temp_c", "collected_at"]


def already_collected(target: dt.date, tmfc: str) -> bool:
    """이미 24시간을 받아둔 (거래일, 분석시각)인지. 중복 API 호출을 막는다.

    스케줄을 하루 여러 번(10/12/14/16시) 걸어두기 때문에 필요하다 — 맥이 10시에 잠들어 있으면
    launchd가 그 실행을 건너뛰므로 뒤의 시각이 받아내야 하고, 이미 받았으면 조용히 끝내야 한다.
    """
    if not os.path.exists(LOG):
        return False
    d = pd.read_csv(LOG, dtype={"tmfc": str, "target_date": str})
    m = (d["target_date"] == target.isoformat()) & (d["tmfc"] == tmfc)
    return int(m.sum()) >= 24


def collect(target: dt.date, force: bool = False) -> None:
    tmfc0 = K.default_tmfc(target)
    if not force and already_collected(target, tmfc0):
        print(f"이미 수집 완료 ({target} / tmfc={tmfc0}) — API를 호출하지 않고 끝냅니다.")
        return
    rows = K.fetch_day(target, lat=LAT, lon=LON)
    if len(rows) != 24:
        print(f"❌ 24시간 중 {len(rows)}개만 받았습니다 — 기록하지 않습니다.")
        return
    tmfc = K.default_tmfc(target)

    # 같은 날 추정값도 함께 남긴다. 나중에 두 경로를 같은 실측에 대고 비교하려면
    # '그때 추정했다면 얼마였는가'가 기록돼 있어야 한다 — 사후 재현은 예보가 사라져 불가능하다.
    est = {}
    try:
        from app.services.kma_forecast import fetch
        est = {w["hour"]: w["solar_rad"] for w in fetch(target, lat=LAT, lon=LON, use_kim=False)}
    except Exception as e:                                        # noqa: BLE001
        print(f"⚠ 추정값을 함께 받지 못했습니다({type(e).__name__}) — KIM 값만 기록합니다.")

    new = pd.DataFrame([{
        "target_date": target.isoformat(), "hour": r.hour, "tmfc": tmfc,
        "kim_mj": r.ghi_mj, "kim_inst_mj": r.inst_mj, "est_mj": est.get(r.hour, np.nan),
        "wind80": r.wind80, "temp_c": r.temp_c,
        "collected_at": dt.datetime.now().isoformat(timespec="seconds")} for r in rows])[COLS]

    if os.path.exists(LOG):
        # tmfc는 반드시 문자열로 읽는다. 기본 추론이면 int64가 되어 문자열 비교가 절대
        # 성립하지 않고, 같은 날을 두 번 수집하면 중복이 그대로 쌓인다(실제로 겪었다).
        old = pd.read_csv(LOG, dtype={"tmfc": str, "target_date": str})
        dup = (old["target_date"] == target.isoformat()) & (old["tmfc"] == tmfc)
        if dup.any():
            print(f"이미 기록돼 있습니다 ({target} / tmfc={tmfc}) — {int(dup.sum())}행을 덮어씁니다.")
            old = old[~dup]
        new = pd.concat([old, new], ignore_index=True)
    # 같은 (거래일, tmfc, 시각)이 두 번 들어오면 마지막 수집만 남긴다
    before = len(new)
    new = new.drop_duplicates(subset=["target_date", "tmfc", "hour"], keep="last")
    if len(new) < before:
        print(f"중복 {before - len(new)}행을 정리했습니다.")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new.to_csv(LOG, index=False)
    days = new["target_date"].nunique()
    print(f"✅ {target} 24시간 기록 (tmfc={tmfc}) · 일적산 KIM {sum(r.ghi_mj for r in rows):.2f} MJ/m²")
    print(f"   누적 {days}일 / {len(new)}행 -> {LOG}")
    print(f"   ASOS 실측은 보통 며칠 뒤 올라옵니다. 그 뒤 report로 비교하세요.")


def report() -> None:
    """비교 계산은 app.services.kim_validation이 한다 — 대시보드와 같은 함수를 써야
    화면과 CLI가 다른 숫자를 말하는 일이 없다. 여기는 출력만 맡는다."""
    from app.services.kim_validation import MIN_DAYS, compare

    if not os.path.exists(LOG):
        sys.exit(f"기록이 없습니다: {LOG}\n먼저 collect를 며칠 돌려야 합니다.")
    c = compare(LOG)
    if c is None:
        print("\nASOS 실측과 겹치는 시간이 아직 없습니다. 관측자료가 올라온 뒤 다시 실행하세요.")
        print("  (기상자료개방포털 ASOS 시간자료를 data/에 내려놓아야 합니다)")
        return
    if c.dup_dropped:
        print(f"⚠ 중복 {c.dup_dropped}행이 있습니다 — collect를 한 번 더 돌리면 정리됩니다.")
    print(f"예보 기록 {c.log_days}일 ({c.first} ~ {c.last})")
    print(f"\n실측과 겹치는 {c.n_days}일 / 주간 {c.n_hours}시간")
    if c.too_few:
        print(f"  ⚠ 표본이 적습니다({MIN_DAYS}일 미만) — 경향만 읽고 결론으로 쓰지 마세요.")
    print()
    print(c.summary.to_string(index=False))
    print("\nMAE·NMAE가 낮을수록 좋다. KIM이 '추정(예보 운량)'보다 낮으면 교체가 옳았다는 증거다.")
    print("'추정(실측 운량)'은 운량 예보가 완벽했다면의 값이다 — 그것과의 차이가 예보 오차 몫이다.")
    print(f"\n일적산 (MJ/m²)\n{c.daily.to_string(index=False)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if mode == "collect":
        args = [a for a in sys.argv[2:] if not a.startswith("-")]
        t = (dt.date.fromisoformat(args[0]) if args
             else dt.date.today() + dt.timedelta(days=1))
        collect(t, force="--force" in sys.argv)
    elif mode == "report":
        report()
    else:
        sys.exit(f"알 수 없는 모드: {mode} (collect | report)")
