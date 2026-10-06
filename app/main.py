"""
AI 서버 (Python/FastAPI) — Backend(Java/Spring Boot)가 REST로 호출하는 내부 서비스.
계획서 02장·06장·09장에서 정의한 역할: ②(날씨->발전량)·③(출력제어 확률예측) 단계를 담당하고,
④(ESS 충방전 결정)에 필요한 계산도 /ess/simulate로 제공한다.

실행: uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import EssSimulateRequest, EssSimulateResponse, PredictRequest, PredictResponse
from app.services import ess_simulation
from app.services.predict import predict as run_predict, warmup

# 기동 시 모델을 미리 읽어 /health가 실제 준비 상태를 말할 수 있게 한다.
# 지연 로딩이면 health가 ok여도 첫 predict가 실패할 수 있다 (통신규격 v1.1 05장 지적 사항).


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup()        # 첫 요청 지연을 없앤다 (약 800ms -> 160ms)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="출력제어 예측 AI 서버",
    description="제주 재생에너지 출력제어 예측 및 ESS 완화효과 시뮬레이션 — 내부 AI 서버 (Backend 전용)",
    version="0.1.0",
)


# schemas.py가 발행하는 도메인 에러코드 목록 (ValueError("<CODE>:<message>") 규약)
DOMAIN_ERROR_CODES = (
    "INVALID_HOUR_SET", "MISSING_REQUIRED_FIELD",
    "INVALID_SOC_RANGE", "INVALID_ENERGY_CAPACITY", "LENGTH_MISMATCH", "OUT_OF_RANGE",
)


def _error_code_from_message(msg: str) -> tuple[str, str]:
    for code in DOMAIN_ERROR_CODES:
        if msg.startswith(code + ":"):
            return code, msg.split(":", 1)[1]
    return "VALIDATION_ERROR", msg


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # pydantic 검증 실패 시 도메인 에러코드로 변환해 반환한다 (09장 검증 책임 분담: AI 서버가 최종 담당)
    first = exc.errors()[0]
    raw_msg = str(first.get("msg", ""))
    code, msg = _error_code_from_message(raw_msg.replace("Value error, ", ""))
    # pydantic 기본 메시지("Field required")는 어느 필드인지 말해주지 않는다 — 위치를 붙인다.
    # Backend가 422를 그대로 릴레이하므로 여기서 안 붙이면 사용자도 원인을 알 수 없다.
    if code == "VALIDATION_ERROR":
        loc = ".".join(str(x) for x in first.get("loc", ()) if x != "body")
        if loc:
            msg = f"{msg} (위치: {loc})"
    return JSONResponse(status_code=422, content={"error_code": code, "message": msg})


@app.get("/health")
def health():
    """모델 적재 여부까지 보고한다 — 상수 ok는 헬스체크 역할을 못 한다.

    ready=false면 아티팩트가 없다는 뜻이고 missing에 이름이 들어간다. Backend는 이 값을
    보고 AI 서버 호출을 보류하거나 경고를 띄울 수 있다.
    """
    # lifespan이 돌지 않은 경우(컨텍스트 매니저 없이 TestClient를 쓰면 그렇다)에도 정확해야
    # 하므로 매번 확인한다. _load가 lru_cache라 두 번째부터는 dict 조회 수준이다.
    st = warmup()
    return {"status": "ok" if st["ready"] else "degraded", **st}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """weather 배열 24개 검증은 여기(AI 서버)에서 최종 수행 — 09장 참고.
    Backend는 필드 존재·타입 같은 기본 형식만 사전 확인하고, 이 규칙 위반 시
    422 + {"error_code": "INVALID_HOUR_SET", ...}를 그대로 릴레이하면 된다.
    """
    return run_predict(req)


@app.post("/ess/simulate", response_model=EssSimulateResponse)
def ess_simulate(req: EssSimulateRequest):
    """04장 02번(기본 시뮬레이션)·03번(용량 조정 슬라이더)이 공유하는 단일 엔드포인트 —
    정격출력(MW)·저장용량(MWh) 2축을 바꿔 여러 번 호출하면 슬라이더 UI가 된다.

    method 선택 기준:
      storage_constrained (기본) — 저장용량·왕복효율·SoC를 반영한 현실 추정. 대표값
        65MW/260MWh(제주 장주기 BESS 중앙계약시장 물량)에서 2023년 실측 풍력 제어량의 52.5%.
      hourly_capped — 정격출력만 보는 이론적 상한. 같은 설비에서 72.7%. 계획서 07장 기준값.
      naive_upper_bound — 제어 발생 시간 × 정격출력. 참고용.
    """
    if req.method == "naive_upper_bound":
        curtailed_hours = sum(1 for v in req.hourly_curtailment_mwh if v > 0)
        total = sum(req.hourly_curtailment_mwh)
        result = ess_simulation.simulate_naive_upper_bound(curtailed_hours, req.rated_power_mw, total)
    elif req.method == "hourly_capped":
        result = ess_simulation.simulate_hourly_capped(req.hourly_curtailment_mwh, req.rated_power_mw)
    else:
        # 저장용량 미지정 시 4시간 — 제주 BESS 65MW/260MWh의 duration
        capacity = req.energy_capacity_mwh or req.rated_power_mw * 4.0
        result = ess_simulation.simulate_with_storage(
            req.hourly_curtailment_mwh, req.rated_power_mw, capacity,
            hour_of_day=req.hour_of_day, round_trip_efficiency=req.round_trip_efficiency,
            soc_min=req.soc_min, soc_max=req.soc_max,
        )
    d = result.to_dict()
    return EssSimulateResponse(
        rated_power_mw=d["rated_power_mw"], method=d["method"],
        total_curtailment_mwh=d["total_curtailment_mwh"], total_absorbed_mwh=d["total_absorbed_mwh"],
        absorption_rate=d["absorption_rate"],
        energy_capacity_mwh=d.get("energy_capacity_mwh"), usable_capacity_mwh=d.get("usable_capacity_mwh"),
        hours_full=d.get("hours_full"), annual_cycles=d.get("annual_cycles"),
    )
