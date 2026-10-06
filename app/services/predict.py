"""
추론 서비스 — 학습된 컨버터/분류기/회귀모델을 로드해 실제 예측을 수행한다.
app/training/*.py 로 학습된 .joblib 아티팩트를 그대로 사용한다.

[수정 사항]
- 누락된 기상값을 0으로 바꾸던 처리 제거 (schemas.py에서 MISSING_REQUIRED_FIELD로 거부)
- 시간/월 피처를 학습과 동일한 기준(각 시간 구간의 '끝 시각' dt)으로 계산
  -> 24시는 다음날 00:00이므로 월말 24시의 month는 다음 달로 계산된다(학습 데이터와 일치)
- 24시간을 한 번에 DataFrame으로 추론 (행마다 DataFrame 생성하던 방식 제거)
- 실제 사용된 분류모델을 응답의 model_used로 반환
- 아티팩트 로드 시 scikit-learn 버전 불일치 경고 (app/model_io.py)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from app.data_prep import OTHER_SOURCE, add_time_features
from app.model_io import MODELS_DIR, converter_predict_mwh, load_artifact
from app.schemas import EnergyType, HourlyPrediction, PredictRequest, PredictResponse
from app.serving_config import artifact_name

ESS_WARNING = (
    "expected_curtailment_mwh = curtailment_probability x E[제어량|제어 발생]로 계산한 '기댓값'입니다. "
    "총합 추정에는 쓸 수 있으나 개별 시간의 제어량 크기가 아니므로 /ess/simulate의 "
    "hourly_curtailment_mwh로 넣지 마세요 — 흡수율이 크게 과대평가됩니다(README '알려진 한계' 참고)."
)

# 타 발전원 발전량을 '컨버터 예측'으로 학습한 변형(_crossp)을 서빙한다. 실측으로 학습한
# 변형(_cross)은 순위는 비슷하지만 서빙 입력에서 확률의 크기가 눌린다 — 2023 컨버터 입력
# Sum(p)가 398(실제 563시간) vs _crossp 518. expected_curtailment_mwh 총합이 Sum(p)에
# 비례하므로 이 차이가 그대로 총합 오차가 된다.
# 모델군(rf/lr)과 경로별 피처 제외는 serving_config가 정한다.
CROSS_ARTIFACT = artifact_name("wind", use_demand=True, use_cross=True)

CROSS_NOTE = (
    "태양광 기상값(solar_rad·temp·cloud)이 함께 와서 계통 전체 침투율 포함 모델을 사용했습니다 — "
    "출력제어는 재생에너지 합계 기준으로 결정되므로 풍력만 보는 모델보다 정확합니다"
    "(2023 테스트 PR-AUC 0.717 -> 0.805)."
)

CROSS_HINT = (
    " 태양광 기상값(solar_rad·temp·cloud)을 함께 보내면 계통 전체 침투율 포함 모델로 전환돼 "
    "정확도가 더 올라갑니다(PR-AUC 0.717 -> 0.805)."
)

# 하루 24시간 중 이 비율을 넘게 범위를 벗어나면 경고한다. 10% = 2.4시간. [가정]
# 실측 발동률은 README '드리프트 경고' 표와 models/serving_real_input_verification.csv 참고.
DRIFT_THRESHOLD = 0.10

SOLAR_NOTE = (
    "태양광 출력제어량(MWh)은 공개되지 않아(연 단위 값만 의원실 자료요청으로 확인됨) "
    "expected_curtailment_mwh를 제공하지 않습니다. curtailment_probability(발생 확률)만 사용하세요."
)


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return load_artifact(name)


@lru_cache(maxsize=1)
def _thresholds() -> dict:
    """경로별 운영 임계값. select_thresholds.py가 만든다 (없으면 빈 dict).

    아티팩트 메타데이터가 아니라 별도 파일인 이유: 임계값은 모델이 존재한 **뒤에** 다른
    스크립트가 다른 구간에서 고른다. 모델을 재학습하지 않고 임계값만 다시 고르는 일이
    가능해야 하고, 반대로 재학습 후 임계값을 다시 고르지 않으면 값이 비어 그 사실이 드러난다.
    """
    path = os.path.join(MODELS_DIR, "operational_thresholds.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def warmup() -> dict:
    """서버 기동 시 모델을 미리 로드한다. 반환값은 /health가 그대로 쓴다.

    [왜 필요한가]
    _load가 lru_cache라 첫 요청 때 디스크에서 읽는다. 그래서 /health가 ok를 반환해도
    **첫 /predict가 실패할 수 있었다** — 헬스체크의 목적을 달성하지 못한다.
    기동 시 한 번 읽어두면 (1) 아티팩트 누락을 즉시 알 수 있고 (2) 첫 요청 지연(약 800ms)이
    사라진다.

    아티팩트가 없어도 기동 자체는 막지 않는다. 서버가 뜨지 않으면 원인을 볼 창구가 없어진다 —
    대신 /health가 ready=false와 누락 목록을 보여준다.
    """
    from app.serving_config import PATH_KEYS, artifact_name

    want = [f"converter_{e}" for e in ("solar", "wind")]
    want += [artifact_name(*_spec(k)) for k in PATH_KEYS]
    optional = ["curtailment_stage2_wind", "radiation_estimator"]

    loaded, missing = [], []
    for n in want:
        try:
            _load(n); loaded.append(n)
        except Exception:                                        # noqa: BLE001
            missing.append(n)
    for n in optional:
        if _optional(n) is not None:
            loaded.append(n)
    try:
        _thresholds()
    except Exception:                                            # noqa: BLE001
        pass
    return {"loaded": len(loaded), "missing": missing, "ready": not missing}


def _spec(key: str) -> tuple[str, bool, bool]:
    return ("solar" if key.startswith("solar") else "wind",
            "demand" in key, "crossp" in key)


def _optional(name: str) -> dict | None:
    try:
        return _load(name)
    except FileNotFoundError:
        return None


def _worst_out_of_range(artifact: dict, df: pd.DataFrame) -> tuple[float, str] | None:
    """아티팩트의 학습 범위를 가장 크게 벗어난 피처와 그 시간 비율. 없으면 None."""
    ranges = (artifact.get("meta") or {}).get("train_feature_ranges")
    if not ranges:
        return None
    worst: tuple[float, str] | None = None
    for f, (lo, hi) in ranges.items():
        if f not in df.columns or f.startswith(("hour_", "month_")):
            continue  # 시각 피처는 순환값이라 범위를 벗어날 수 없다
        v = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
        frac = float(((v < lo) | (v > hi)).mean())
        if frac > 0 and (worst is None or frac > worst[0]):
            worst = (frac, f"{f} {frac:.0%}(학습범위 {lo:g}~{hi:g})")
    return worst


def _drift_note(df: pd.DataFrame, stages: list[tuple[str, dict]]) -> str | None:
    """입력이 학습 분포를 벗어났으면 경고 문구를 만든다 (없으면 None).

    파이프라인의 모든 단계를 본다 — 교차 경로에서는 컨버터가 둘(자기 발전원 + 타 발전원)이다.
      기상 -> 컨버터        : 원본 기상값(풍속·일사량·기온·전운량)이 컨버터 학습 범위 밖인가
      기상 -> 타발전원컨버터 : 태양광 기상값이 태양광 컨버터 학습 범위 밖인가
      파생 -> 분류기        : capacity_factor·penetration 등이 분류기 학습 범위 밖인가
    분류기 쪽만 보면 컨버터 자신의 외삽 실패를 놓친다 — 컨버터가 발전량을 눌러 출력하면
    그 눌린 값은 분류기 범위 안에 들어와 조용해진다.

    [왜 필요한가] RandomForest는 외삽을 못 한다. 입력이 학습 범위를 벗어나면 경계값에
    포화돼 큰 값들이 서로 구분되지 않고 순위가 무너지는데, 응답만 보면 알 수 없다.
    이 저장소는 실제로 그 함정에 빠진 적이 있다 — 설비 증설로 태양광 발전량이 학습 범위를
    3.8% 초과하면서 PR-AUC가 0.406까지 떨어졌고, 정규화로 고쳤지만 설비용량 대리지표가
    학습 시점 값으로 고정돼 증설이 이어지면 다시 벌어진다. README가 '연 1회 이상 재학습'을
    권고하는 근거이기도 하다. 그 시점을 사람이 달력으로 재는 대신 입력이 말해주게 한다.

    범위는 학습 구간의 min~max다(train_classifier.feature_ranges). 드리프트 감지기는 학습
    데이터에서 절대 울리지 않아야 하므로 분위수를 쓰지 않는다 — p1~p99로 만들었을 때 학습
    구간에서 21.2%의 날이 경고를 받았다(그 함수의 주석 참고).

    임계 10%(하루 24시간 중 2.4시간)는 운영 판단이다 [가정]. 실측 입력으로 확인한 발동률은
    학습 구간 0.0%, 2023 테스트 1.9%, 2025~2026년은 아래 README '드리프트 경고' 표 참고.

    두 단계를 따로 보고하는 이유: 기상값이 범위 밖이면 발전량 예측부터 못 믿을 값이고,
    파생 피처만 범위 밖이면 발전량은 그럴듯한데 분류 확률이 포화된 경우다. 처방이 다르다
    (앞은 컨버터 재학습, 뒤는 분류기 재학습 또는 설비용량 대리지표 갱신).
    """
    parts = []
    for stage, art in stages:
        if art is None:
            continue
        w = _worst_out_of_range(art, df)
        if w and w[0] >= DRIFT_THRESHOLD:
            parts.append(f"{stage} {w[1]}")
    if not parts:
        return None
    return (f"⚠ 입력이 학습 분포를 벗어났습니다 — {' / '.join(parts)}. RandomForest는 외삽하지 "
            f"못해 예측이 경계에 포화되고 순위가 무너질 수 있습니다. 재학습을 권고합니다"
            f"(README '드리프트 경고').")


def predict(req: PredictRequest) -> PredictResponse:
    energy_type: EnergyType = req.energy_type
    converter = _load(f"converter_{energy_type}")

    # 학습 데이터와 같은 기준: h시 = target_date 00:00 + h시간 (24시 -> 다음날 00:00)
    base = datetime.combine(req.target_date, datetime.min.time())
    weather = sorted(req.weather, key=lambda w: w.hour)
    df = pd.DataFrame([w.model_dump() for w in weather])
    df["dt"] = [base + timedelta(hours=w.hour) for w in weather]
    df = add_time_features(df)

    # ---- ②단계: 날씨 -> 발전량 예측치 ----
    # 태양광 컨버터는 타깃이 '이용률'이라 설비용량 대리지표를 곱해 MWh로 환산한다.
    # 곱하는 상수는 분류기가 나눌 때 쓰는 상수와 같아야 왕복이 정확히 상쇄된다
    # (양쪽 모두 data_prep.latest_capacity_proxy로 계산해 아티팩트에 저장).
    df["generation_mwh"] = converter_predict_mwh(converter, df)

    # ---- ③단계: 발전량 -> 정규화 피처 -> 출력제어 확률 ----
    # /predict는 sigmoid 보정 분류기만 로드한다(보정 전 모델을 서빙하는 경로는 없다).
    #
    # [2026-09-25] 분류기 입력이 발전량 절대값에서 정규화 피처로 바뀌었다.
    #   capacity_factor = 발전량 / 설비용량대리
    #   penetration     = 발전량 / 수요
    # 설비용량 대리지표는 미래 시각에 대해 계산할 수 없으므로 학습 시점의 최신값을
    # 아티팩트 메타데이터(capacity_proxy_mwh)에 고정해두고 상수로 쓴다.
    # 설비가 크게 늘면 이 상수가 실제와 벌어지므로 재학습이 필요하다.
    use_demand = req.demand_forecast_mw is not None

    # [2026-09-26] 풍력은 '계통 전체 침투율' 모델로 자동 전환한다.
    # 출력제어는 발전원별이 아니라 재생에너지 합계 기준으로 결정되므로(계통평가세부운영규정
    # 제8.3.1조의 P재생E전망), 풍력 제어를 예측하려면 태양광을 봐야 한다. 2023년 풍력 제어
    # 563시간의 평균 태양광 발전량은 235.2MWh(비제어시 43.3)인데 풍력은 50.7(비제어시 58.6)로
    # 오히려 낮다 — 풍력 이용률 단독의 AUC는 0.476으로 무작위보다 나쁘다.
    #
    # 태양광 발전량은 같은 요청의 태양광 기상값을 태양광 컨버터에 넣어 얻는다. 새 입력을
    # 요구하지 않고 기존 아티팩트를 재사용하는 경로다. 기상값이 없으면 기존 모델로 되돌아간다
    # (demand_forecast_mw의 자동 전환과 같은 방식).
    #
    # 태양광에는 적용하지 않는다. 같은 실험에서 태양광은 top5가 0.8475 -> 0.8333으로 나빠졌다 —
    # 제주 잉여의 주역이 태양광 자신이라 풍력을 더해도 정보가 늘지 않는다. 비대칭이 정상이다.
    solar_fields = ("solar_rad", "temp", "cloud")
    use_cross = (
        energy_type == "wind" and use_demand
        and all(df[f].notna().all() for f in solar_fields if f in df.columns)
        and all(f in df.columns for f in solar_fields)
        and _optional(CROSS_ARTIFACT) is not None
    )

    clf_name = artifact_name(energy_type, use_demand, use_cross)
    classifier = _load(clf_name)

    proxy = (classifier.get("meta") or {}).get("capacity_proxy_mwh")
    if not proxy:
        raise RuntimeError(
            f"{clf_name}: capacity_proxy_mwh 메타데이터가 없습니다 — "
            f"`python -m app.training.train_classifier`로 재학습하세요"
        )
    df["capacity_factor"] = df["generation_mwh"] / float(proxy)
    if use_demand:
        df["demand_mw"] = req.demand_forecast_mw
        df["penetration"] = df["generation_mwh"] / df["demand_mw"]
    if use_cross:
        # 타 발전원 컨버터를 같은 기상값으로 돌린다. 태양광 컨버터는 이용률 타깃이라
        # converter_predict_mwh가 자기 아티팩트의 capacity_proxy_mwh로 MWh로 환산한다.
        other = _load(f"converter_{OTHER_SOURCE[energy_type]}")
        df["other_generation_mwh"] = converter_predict_mwh(other, df)
        df["total_penetration"] = (
            (df["generation_mwh"] + df["other_generation_mwh"]) / df["demand_mw"])

    proba = classifier["model"].predict_proba(df[classifier["features"]])[:, 1]
    # 교차 경로에서는 타 발전원 컨버터도 같은 기상값을 받으므로 그 범위도 함께 본다.
    stages = [("기상->컨버터", converter)]
    if use_cross:
        stages.append((f"기상->{OTHER_SOURCE[energy_type]}컨버터",
                       _load(f"converter_{OTHER_SOURCE[energy_type]}")))
    stages.append(("파생->분류기", classifier))
    drift = _drift_note(df, stages)

    # 풍력 + 수요예측이면 같은 확률에 조건부 제어량을 곱해 기댓값을 만든다.
    # 두 값이 같은 확률을 쓰므로 expected / probability = 조건부 제어량이 성립한다.
    expected = [None] * 24
    if energy_type == "wind" and use_demand:
        stage2 = _optional("curtailment_stage2_wind")
        if stage2 is not None:
            conditional = np.clip(stage2["model"].predict(df[stage2["features"]]), 0, None)
            # 주의: 기댓값이라 개별 시간의 제어량 크기가 아니다 — ESS 계산 금지(ESS_WARNING)
            expected = (proba * conditional).tolist()

    hourly = [
        HourlyPrediction(
            hour=int(h),
            generation_forecast_mwh=round(float(g), 2),
            curtailment_probability=round(float(p), 4),
            expected_curtailment_mwh=round(float(e), 2) if e is not None else None,
        )
        for h, g, p, e in zip(df["hour"], df["generation_mwh"], proba, expected)
    ]

    note = SOLAR_NOTE if energy_type == "solar" else None
    if not use_demand:
        note = (f"demand_forecast_mw가 없어 침투율(발전량/수요) 미포함 모델"
                f"({clf_name})을 사용했습니다. 정확도가 낮으므로 가능하면 수요예측을 함께 보내세요."
                + ("" if energy_type == "solar" else
                   " 제어량 조건부 회귀도 수요 피처가 필요해 expected_curtailment_mwh는 null입니다."))
        if energy_type == "solar":
            note = SOLAR_NOTE + " " + note
    elif energy_type == "wind" and expected[0] is not None:
        note = (CROSS_NOTE + " " + ESS_WARNING) if use_cross else (ESS_WARNING + CROSS_HINT)
    elif energy_type == "wind":
        note = ("조건부 회귀 아티팩트(curtailment_stage2_wind)가 없어 확률만 반환하고 "
                "expected_curtailment_mwh는 null입니다. "
                "`python -m app.training.train_curtailment_regressor`로 학습하세요.")

    # 드리프트 경고는 다른 안내보다 먼저 읽혀야 한다 — 그 아래 수치의 신뢰도를 깎는 정보다.
    if drift:
        note = f"{drift} {note}" if note else drift

    thr = _thresholds().get(clf_name) or {}
    if thr and not thr.get("reliable", True):
        # 신뢰할 수 없는 임계값을 조용히 내려보내면 백엔드가 그대로 경보에 쓴다.
        hint = (f"운영 임계값 {thr['threshold']}은 선정 구간 양성이 "
                f"{thr.get('n_pos_sel')}건뿐이라 신뢰할 수 없습니다 — 임계값 판정 대신 "
                f"등급(상위 5%)으로 표시하세요.")
        note = f"{note} {hint}" if note else hint

    return PredictResponse(
        energy_type=energy_type, region=req.region, target_date=req.target_date,
        hourly=hourly, model_used=clf_name, note=note,
        operational_threshold=thr.get("threshold"),
        operational_threshold_reliable=thr.get("reliable"),
    )
