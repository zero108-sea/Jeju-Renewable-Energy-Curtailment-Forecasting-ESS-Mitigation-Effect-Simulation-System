"""
Backend(Java/Spring Boot) <-> AI 서버(Python) REST 계약.

이 파일이 곧 API 계약 문서다 — 09장 리스크 대응 전략("API 계약을 코드보다 문서로 먼저 고정")에 따라,
필드명·타입·단위·에러코드를 여기 한 곳에서만 정의하고 Backend 쪽에는 이 스키마를 그대로 전달한다.

검증 책임 분담 (09장 반영 그대로):
  - Backend: 필드 존재 여부·기본 타입만 확인 (JSON 파싱 레벨)
  - AI 서버: 도메인 규칙 검증 — "weather 배열이 정확히 24개, 시간 중복 없음" 등
    위반 시 HTTP 422 + error_code="INVALID_HOUR_SET"
  - AI 서버: 발전원별 필수 기상값 누락 검증 [신규]
    위반 시 HTTP 422 + error_code="MISSING_REQUIRED_FIELD"
    (수정 전에는 누락값을 조용히 0으로 바꿔 예측했다 — 예: 풍속 null -> 풍속 0 -> '제어 없음'으로 오예측)

에러 메시지 규약: ValueError("<ERROR_CODE>:<사람이 읽는 메시지>") — main.py가 코드와 메시지를 분리한다.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

EnergyType = Literal["solar", "wind"]

# 발전원별 컨버터 필수 입력 (app/training/train_converter.py의 피처와 일치해야 함)
REQUIRED_WEATHER_FIELDS: dict[str, tuple[str, ...]] = {
    "solar": ("solar_rad", "temp", "cloud"),
    "wind": ("wind_speed",),
}


class WeatherHour(BaseModel):
    """시간별 기상예보 1건. hour는 1~24(24시=자정, 원본 공공데이터 관례와 동일).

    ⚠ **Optional은 '널을 보내도 된다'는 뜻이 아니다.** 타입이 Optional인 이유는 하나의 DTO로
    두 발전원을 모두 받기 위해서이고(태양광 요청에 wind_speed는 불필요), 실제 필수 여부는
    energy_type에 따라 결정된다 — REQUIRED_WEATHER_FIELDS 참고.

      energy_type="solar"  ->  solar_rad, temp, cloud 가 24시간 모두 필요
      energy_type="wind"   ->  wind_speed 가 24시간 모두 필요

    한 시간이라도 null이면 422 MISSING_REQUIRED_FIELD이고, 메시지에 누락 위치가 `13시.solar_rad`
    형태로 최대 5건까지 들어간다. 조용히 0으로 대체하지 않는다 — 풍속 null을 0으로 바꾸면
    '제어 없음'으로 오예측한다(이 동작을 실제로 겪어 고쳤다).

    [값 범위 — 2026-09-30 추가]
    범위를 넣기 전에는 `solar_rad: -1`(음수 일사량), `wind_speed: 500`이 **200으로 통과해
    그럴듯한 숫자를 돌려줬다.** 드리프트 경고가 note에 실리기는 했지만 HTTP 상태는 200이라,
    Backend가 note를 파싱하지 않으면 쓰레기 예측이 DB와 화면까지 간다.

    상한은 제주 ASOS 184 실측(2021~2023)보다 넉넉히 두고 물리 한계만 지킨다 —
    예보값이 실측 범위를 조금 넘는 것은 정상이므로 그것까지 막으면 안 된다.

      필드         실측 범위        허용 범위     근거
      solar_rad   0.0 ~ 3.9       0 ~ 6        음수 불가
      temp        −3.1 ~ 37.0     −30 ~ 50
      cloud       0.0 ~ 10.0      0 ~ 10       기상청 전운량 정의
      wind_speed  0.0 ~ 13.7      0 ~ 60       음수 불가, 태풍 최대풍속 여유
    """
    hour: int = Field(..., ge=1, le=24, description="1~24시 (24시 = 자정)")
    solar_rad: Optional[float] = Field(
        None, ge=0.0, le=6.0,
        description="일사량(MJ/m2) — solar일 때 필수, wind일 때 선택. 0 이상 (음수는 물리적으로 불가)")
    temp: Optional[float] = Field(
        None, ge=-30.0, le=50.0,
        description="기온(°C) — solar일 때 필수, wind일 때 선택")
    cloud: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="전운량 — solar일 때 필수, wind일 때 선택. 기상청 정의상 0~10")
    wind_speed: Optional[float] = Field(
        None, ge=0.0, le=60.0,
        description="풍속(m/s) — wind일 때 필수, solar일 때 선택. 0 이상 (음수는 물리적으로 불가)")


class PredictRequest(BaseModel):
    energy_type: EnergyType
    region: str = Field(
        ..., min_length=1, max_length=100,
        description=(
            "관측/발전 지역 표기. **형식 제약 없는 자유 문자열**이고 응답에 그대로 되돌려준다. "
            "모델은 제주 전역 단일 모델이라 이 값으로 분기하지 않는다 — 즉 '제주', '남원읍', "
            "'제주특별자치도 서귀포시 남원읍' 모두 같은 예측을 낸다. Backend는 화면 표시에 쓸 "
            "문자열을 그대로 보내면 된다. 빈 문자열·공백만 있는 값은 거부한다(422)."
        ))
    target_date: date
    weather: list[WeatherHour] = Field(..., description="정확히 24개, 1~24시 각 1회")
    demand_forecast_mw: Optional[list[float]] = Field(
        None, description="24개(선택). 각 값은 0 초과(제주 실측 378~1,104MW). 있으면 침투율(발전량/수요) 포함 모델을 사용한다 — 태양광·풍력 모두 정확도가 올라간다(태양광 PR-AUC 0.639->0.711). 풍력은 수요 자체도 정식 피처다(05장). 생략하면 미포함 모델로 자동 전환되고 풍력의 expected_curtailment_mwh는 null이 된다"
    )

    @field_validator("weather")
    @classmethod
    def _check_24_hours(cls, v: list[WeatherHour]) -> list[WeatherHour]:
        if len(v) != 24:
            raise ValueError("INVALID_HOUR_SET:weather 배열은 정확히 24개여야 합니다")
        hours = sorted(h.hour for h in v)
        if hours != list(range(1, 25)):
            raise ValueError("INVALID_HOUR_SET:weather 배열의 시간이 1~24시 각 1회가 아닙니다(중복 또는 누락)")
        return v

    @field_validator("region")
    @classmethod
    def _check_region(cls, v: str) -> str:
        # 값 자체는 쓰지 않지만 빈 값을 받아 그대로 echo하면 화면에 빈칸이 뜬다.
        if not v.strip():
            raise ValueError("MISSING_REQUIRED_FIELD:region은 비어 있을 수 없습니다")
        return v

    @model_validator(mode="after")
    def _check_demand_length(self) -> "PredictRequest":
        if self.demand_forecast_mw is not None:
            if len(self.demand_forecast_mw) != 24:
                raise ValueError("INVALID_HOUR_SET:demand_forecast_mw는 24개여야 합니다")
            bad = [f"{i+1}시={v:g}" for i, v in enumerate(self.demand_forecast_mw) if v <= 0]
            if bad:
                raise ValueError(
                    "OUT_OF_RANGE:demand_forecast_mw는 0보다 커야 합니다 "
                    f"(위반: {', '.join(bad[:5])}{f' 외 {len(bad)-5}건' if len(bad) > 5 else ''})")
        return self

    @model_validator(mode="after")
    def _check_required_weather(self) -> "PredictRequest":
        required = REQUIRED_WEATHER_FIELDS[self.energy_type]
        missing = sorted(
            {f"{w.hour}시.{f}" for w in self.weather for f in required if getattr(w, f) is None},
            key=lambda s: (int(s.split("시")[0]), s),
        )
        if missing:
            preview = ", ".join(missing[:5]) + (f" 외 {len(missing) - 5}건" if len(missing) > 5 else "")
            raise ValueError(
                f"MISSING_REQUIRED_FIELD:{self.energy_type} 예측에는 {', '.join(required)}가 모든 시간에 필요합니다 "
                f"(누락: {preview})"
            )
        return self


class HourlyPrediction(BaseModel):
    hour: int
    generation_forecast_mwh: float
    curtailment_probability: float = Field(
        ..., description="출력제어 발생 확률 0~1. sigmoid 보정된 값이라 보정 전 모델과 크기가 "
                         "다르다. 0.5 같은 고정 임계값을 쓰지 말 것(README '운영 임계값' 참고). "
                         "산출 모델은 model_used 참고"
    )
    expected_curtailment_mwh: Optional[float] = Field(
        None,
        description=(
            "제어량 기댓값 = curtailment_probability x E[제어량|제어 발생] (2단계 모델). "
            "같은 확률을 쓰므로 expected_curtailment_mwh / curtailment_probability = 조건부 제어량이 된다. "
            "풍력 + demand_forecast_mw가 있을 때만 값이 존재하고, 태양광은 07장 사유로 null 고정. "
            "⚠ 이 값은 '기댓값'이라 개별 시간의 제어량 크기가 아니다 — /ess/simulate의 "
            "hourly_curtailment_mwh로 넘기지 말 것. 2023년 검증에서 ESS 흡수율이 "
            "실측 36.8% 대비 80.6%로 과대평가됐다(README '알려진 한계')."
        ),
    )


class PredictResponse(BaseModel):
    energy_type: EnergyType
    region: str
    target_date: date
    hourly: list[HourlyPrediction]
    model_used: str = Field(
        ..., description="curtailment_probability를 산출한 sigmoid 보정 모델 이름. "
                         "demand_forecast_mw가 있으면 'classifier_{solar|wind}_demand_calibrated_sigmoid', "
                         "없으면 'classifier_{solar|wind}_calibrated_sigmoid'. "
                         "값을 하드코딩해 분기하지 말 것 — 보정 방식이 바뀌면 이름도 바뀐다"
    )
    operational_threshold: Optional[float] = Field(
        None,
        description=(
            "이 model_used에 대해 선정된 '제어 발생 경보' 임계값. curtailment_probability가 이 값 "
            "이상이면 경보로 취급한다. **하드코딩하지 말고 이 값을 쓸 것** — 확률 척도는 모델의 "
            "피처 구성과 보정 매핑이 함께 만들므로 경로마다 다르다(0.02~0.43). "
            "operational_threshold_reliable이 false면 표본이 부족해 신뢰할 수 없으니 임계값 판정 "
            "대신 등급(상위 5%)으로 표시할 것. 선정 절차는 app/training/select_thresholds.py."
        ),
    )
    operational_threshold_reliable: Optional[bool] = Field(
        None,
        description="임계값 선정 구간의 양성 표본이 충분했는지(50건 이상). false면 임계값 판정 금지",
    )
    model_trained_at: Optional[str] = Field(
        None,
        description=(
            "curtailment_probability를 산출한 모델이 학습된 시각(ISO 8601, UTC). "
            "**어느 아티팩트가 이 예측을 냈는지 추적하는 값이다** — model_used는 경로 이름이라 "
            "재학습해도 바뀌지 않지만 이 값은 바뀐다. 예측 결과를 저장한다면 함께 남길 것. "
            "나중에 '이 수치는 어느 모델이 낸 것인가'를 되물을 때 유일한 단서가 된다."
        ),
    )
    note: Optional[str] = Field(
        None, description="태양광 응답에는 expected_curtailment_mwh가 null인 이유를 항상 포함"
    )


class EssSimulateRequest(BaseModel):
    hourly_curtailment_mwh: list[float] = Field(
        ..., min_length=1,
        description="시간별 출력제어량(MWh), 24개 또는 임의 길이. 빈 배열 불가, 각 값은 0 이상")
    rated_power_mw: float = Field(65.0, gt=0.0, description="ESS 정격출력(MW), 0 초과. — 04장 03번 슬라이더 1축. 기본값은 제주 장주기 BESS 중앙계약시장 물량")
    method: Literal["storage_constrained", "hourly_capped", "naive_upper_bound"] = "storage_constrained"
    # --- method="storage_constrained" 전용 (저장용량 제약 반영) ---
    energy_capacity_mwh: float | None = Field(
        None, description="ESS 저장용량(MWh) — 슬라이더 2축. 미지정 시 rated_power_mw × 4시간(제주 BESS 기준)"
    )
    round_trip_efficiency: float = Field(0.90, gt=0.0, le=1.0, description="왕복효율 — 상용 BESS AC 85~94% 중앙값")
    soc_min: float = Field(0.10, ge=0.0, lt=1.0, description="최소 충전상태 [가정]")
    soc_max: float = Field(0.90, gt=0.0, le=1.0, description="최대 충전상태 [가정]")
    hour_of_day: list[int] | None = Field(
        None, description="각 시간의 시각(1~24). 미지정 시 1시부터 시작하는 연속 시계열로 간주"
    )

    @model_validator(mode="after")
    def _check_storage(self):
        bad = [f"{i}번째={v:g}" for i, v in enumerate(self.hourly_curtailment_mwh) if v < 0]
        if bad:
            raise ValueError("OUT_OF_RANGE:hourly_curtailment_mwh에 음수가 있습니다 "
                             f"(위반: {', '.join(bad[:5])})")
        if self.soc_min >= self.soc_max:
            raise ValueError("INVALID_SOC_RANGE:soc_min이 soc_max보다 작아야 합니다")
        if self.energy_capacity_mwh is not None and self.energy_capacity_mwh <= 0:
            raise ValueError("INVALID_ENERGY_CAPACITY:저장용량은 0보다 커야 합니다")
        if self.hour_of_day is not None and len(self.hour_of_day) != len(self.hourly_curtailment_mwh):
            raise ValueError("LENGTH_MISMATCH:hour_of_day 길이가 hourly_curtailment_mwh와 같아야 합니다")
        return self


class EssSimulateResponse(BaseModel):
    rated_power_mw: float
    method: str
    total_curtailment_mwh: float
    total_absorbed_mwh: float
    absorption_rate: float
    # storage_constrained에서만 채워진다 (정격출력만 쓰는 방식은 저장용량 개념이 없음)
    energy_capacity_mwh: float | None = None
    usable_capacity_mwh: float | None = None
    hours_full: int | None = Field(None, description="ESS가 가득 차서 더 흡수하지 못한 시간 수")
    annual_cycles: float | None = Field(None, description="가용용량 기준 환산 사이클 수")


class ErrorResponse(BaseModel):
    error_code: str
    message: str
