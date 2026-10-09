"""KIM 일사량 예보 vs ASOS 실측 — 비교 계산만 담는다.

[왜 app/ 밑으로 뺐나]
이 계산은 원래 `scripts/kim_validation_log.py`의 report() 안에 print와 섞여 있었다.
대시보드에서도 같은 표를 보여주게 되면서 둘 중 하나를 골라야 했다 — 로직을 복제하거나,
계산을 꺼내 공유하거나. 복제하면 **화면과 CLI가 다른 숫자를 말하는 날이 반드시 온다.**
그래서 계산만 여기로 옮기고, 출력(print / st.dataframe)은 각자 하게 했다.

[읽는 법]
ASOS 실측을 정답으로 두고 네 경로를 비교한다. '추정(실측 운량)'은 운량 예보가 완벽했다면의
값이라, 그것과 '추정(예보 운량)'의 차이가 **운량 예보 탓인 오차**다. 이 행이 없으면
과대예측이 모델 탓인지 예보 탓인지 영원히 가를 수 없다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "kim_forecast_log.csv")

PATHS = (("KIM 예보", "kim_mj"), ("KIM 순간값", "kim_inst_mj"),
         ("추정(예보 운량)", "est_mj"), ("추정(실측 운량)", "추정_실측운량"))

# 이 날 수 미만이면 수치를 결론으로 쓸 수 없다. 화면과 CLI가 같은 기준으로 경고해야 한다.
MIN_DAYS = 10


@dataclass
class Comparison:
    summary: pd.DataFrame      # 경로별 MAE·NMAE·편향·상관·일적산비
    daily: pd.DataFrame        # 거래일별 일적산
    n_days: int                # 실측과 겹치는 날 수
    n_hours: int               # 주간 시간 수
    log_days: int              # 수집한 예보 날 수(실측 없는 날 포함)
    first: str
    last: str
    dup_dropped: int

    @property
    def too_few(self) -> bool:
        return self.n_days < MIN_DAYS


def estimate_with_observed_cloud(asos: pd.DataFrame, dts: pd.Series) -> pd.Series:
    """추정 모델에 ASOS 실측 운량을 넣어 돌린다 — 운량 예보 오차를 제거한 상한."""
    from app.data_prep import add_time_features
    from app.model_io import load_artifact
    from app.solar_radiation import add_solar_geometry, sky_from_cloud
    from app.training.train_radiation_model import NIGHT_MJ

    w = asos[asos["dt"].isin(set(dts))].copy().reset_index(drop=True)
    if w.empty:
        return pd.Series(index=dts.index, dtype=float)
    w = add_solar_geometry(add_time_features(w), "184")
    w["sky"] = sky_from_cloud(w["cloud"])
    art = load_artifact("radiation_estimator")
    out = np.zeros(len(w))
    m = (w["clear_sky_mj"] > NIGHT_MJ).to_numpy()
    if m.any():
        p = art["model"].predict(w.loc[m, art["features"]])
        if (art["meta"].get("target") or "").startswith("clear_sky_index"):
            p = np.clip(p, 0, 1.2) * w.loc[m, "clear_sky_mj"].to_numpy()
        out[m] = np.clip(p, 0, None)
    return dts.map(dict(zip(w["dt"], out)))


def compare(log_path: str = LOG) -> Comparison | None:
    """수집 로그와 ASOS 실측을 맞춰 비교표를 만든다. 겹치는 시간이 없으면 None."""
    if not os.path.exists(log_path):
        return None
    log = pd.read_csv(log_path, dtype={"tmfc": str, "target_date": str})
    dup = int(log.duplicated(subset=["target_date", "tmfc", "hour"]).sum())
    if dup:
        log = log.drop_duplicates(subset=["target_date", "tmfc", "hour"], keep="last")
    log["dt"] = (pd.to_datetime(log["target_date"])
                 + pd.to_timedelta(log["hour"], unit="h"))

    from app.data_prep import load_asos
    asos = load_asos("184")
    obs = asos[["dt", "solar_rad"]].rename(columns={"solar_rad": "실측"})
    j = log.merge(obs, on="dt", how="inner").dropna(subset=["실측"])
    if j.empty:
        return None

    # 야간은 양쪽 다 0이라 평균을 희석시킨다 — 한쪽이라도 빛이 있는 시간만 본다
    day = j[(j["실측"] > 0.02) | (j["kim_mj"] > 0.02)].copy()
    day["추정_실측운량"] = estimate_with_observed_cloud(asos, day["dt"])

    rows = []
    for name, col in PATHS:
        if col not in day or day[col].isna().all():
            continue
        d = day[col] - day["실측"]
        mae = d.abs().mean()
        rows.append({
            "경로": name, "MAE": round(mae, 3),
            "NMAE(%)": round(100 * mae / day["실측"].mean(), 1) if day["실측"].mean() else np.nan,
            "편향": round(d.mean(), 3),
            "상관": round(day[col].corr(day["실측"]), 3),
            "일적산비": round(day[col].sum() / day["실측"].sum(), 3)})

    j = j.merge(day[["dt", "추정_실측운량"]], on="dt", how="left")
    daily = j.groupby("target_date")[["kim_mj", "est_mj", "추정_실측운량", "실측"]].sum().round(2)
    daily.columns = ["KIM", "추정(예보운량)", "추정(실측운량)", "실측"]
    return Comparison(
        summary=pd.DataFrame(rows), daily=daily.reset_index(),
        n_days=int(j["target_date"].nunique()), n_hours=int(len(day)),
        log_days=int(log["target_date"].nunique()),
        first=str(log["target_date"].min()), last=str(log["target_date"].max()),
        dup_dropped=dup)
