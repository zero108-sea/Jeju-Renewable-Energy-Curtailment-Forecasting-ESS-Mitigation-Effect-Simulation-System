"""
FastAPI TestClient 기반 엔드투엔드 점검. 실패하면 AssertionError로 중단된다(수정 전: print만 해서 실패를 못 잡음).

선행: 모델 학습(README '실행' 참고)
실행: python smoke_test.py
"""
import sys

sys.path.insert(0, ".")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app
from app.serving_config import artifact_name  # noqa: E402

client = TestClient(app)

weather24 = [{"hour": h, "solar_rad": 1.5 if 7 <= h <= 18 else 0.0, "temp": 20.0, "cloud": 3.0,
              "wind_speed": 6.0} for h in range(1, 25)]
# 풍력만 있고 태양광 기상값이 없는 요청 — 계통 전체 침투율 모델로 전환되지 않아야 한다
wind_only24 = [{"hour": h, "wind_speed": 6.0} for h in range(1, 25)]


def post(body):
    return client.post("/predict", json=body)


def check(name, cond, detail=""):
    assert cond, f"FAIL: {name} {detail}"
    print(f"  ✓ {name}")


print("== /health ==")
_h = client.get("/health").json()
# /health는 상수가 아니라 모델 적재 상태를 보고한다 (v1.2). ready=false면 아티팩트 누락이다.
check("health 응답 모양", {"status", "loaded", "missing", "ready"} <= set(_h))
check(f"모델 적재 완료 (loaded={_h.get('loaded')}, missing={_h.get('missing')})", _h.get("ready") is True)

print("== /predict solar ==")
r = post({"energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15", "weather": weather24})
check("200", r.status_code == 200, r.text)
b = r.json()
check("24시간 응답", [h["hour"] for h in b["hourly"]] == list(range(1, 25)))
check("확률 0~1", all(0 <= h["curtailment_probability"] <= 1 for h in b["hourly"]))
check("발전량 >= 0", all(h["generation_forecast_mwh"] >= 0 for h in b["hourly"]))
check("태양광 제어량 null", all(h["expected_curtailment_mwh"] is None for h in b["hourly"]))
check("태양광 note 존재", bool(b["note"]))
check("model_used=보정 태양광",
      b["model_used"] == artifact_name("solar", use_demand=False, use_cross=False))

print("== /predict wind (수요 + 태양광 기상값 -> 계통 전체 침투율 모델) ==")
r = post({"energy_type": "wind", "region": "한림읍", "target_date": "2023-06-15", "weather": weather24,
          "demand_forecast_mw": [600.0] * 24})
check("200", r.status_code == 200, r.text)
b = r.json()
# 이름을 하드코딩하지 않는다 — 모델군(rf/lr)이 바뀌면 이름도 바뀐다.
check("model_used=계통 전체 침투율 모델",
      b["model_used"] == artifact_name("wind", use_demand=True, use_cross=True), b["model_used"])
check("풍력 제어량 값 존재", all(h["expected_curtailment_mwh"] is not None for h in b["hourly"]))
# 확률 통일 검증: expected = curtailment_probability x E[제어량|제어] 이므로
# expected / probability 가 물리적으로 말이 되는 조건부 제어량(0~500MWh)이어야 한다.
implied = [h["expected_curtailment_mwh"] / h["curtailment_probability"]
           for h in b["hourly"] if h["curtailment_probability"] > 0.01]
check("expected = prob x 조건부 관계 성립", bool(implied) and all(0 <= v <= 500 for v in implied),
      f"implied 조건부 제어량 범위: {min(implied):.1f}~{max(implied):.1f}" if implied else "확률이 모두 0")

print("== /predict wind (태양광 기상값 생략 -> 풍력 단독 모델로 되돌아감) ==")
r = post({"energy_type": "wind", "region": "한림읍", "target_date": "2023-06-15", "weather": wind_only24,
          "demand_forecast_mw": [600.0] * 24})
check("200", r.status_code == 200, r.text)
check("model_used=수요 포함·교차 미포함",
      r.json()["model_used"] == artifact_name("wind", use_demand=True, use_cross=False),
      r.json()["model_used"])

print("== /predict wind (수요 생략 -> 자동 전환, 수정 전에는 500 에러) ==")
r = post({"energy_type": "wind", "region": "한림읍", "target_date": "2023-06-15", "weather": weather24})
check("200", r.status_code == 200, r.text)
check("model_used=보정 수요 미포함",
      r.json()["model_used"] == artifact_name("wind", use_demand=False, use_cross=False))

print("== 에러코드 ==")
r = post({"energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15", "weather": weather24[:23]})
check("23개 -> INVALID_HOUR_SET", r.status_code == 422 and r.json()["error_code"] == "INVALID_HOUR_SET", r.text)
r = post({"energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15",
          "weather": weather24[:23] + [weather24[0]]})
check("중복 -> INVALID_HOUR_SET", r.status_code == 422 and r.json()["error_code"] == "INVALID_HOUR_SET", r.text)
bad = [dict(w) for w in weather24]
bad[5]["wind_speed"] = None
r = post({"energy_type": "wind", "region": "한림읍", "target_date": "2023-06-15", "weather": bad})
check("풍속 누락 -> MISSING_REQUIRED_FIELD",
      r.status_code == 422 and r.json()["error_code"] == "MISSING_REQUIRED_FIELD", r.text)
bad = [dict(w) for w in weather24]
bad[0]["solar_rad"] = None
r = post({"energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15", "weather": bad})
check("일사량 누락 -> MISSING_REQUIRED_FIELD",
      r.status_code == 422 and r.json()["error_code"] == "MISSING_REQUIRED_FIELD", r.text)

print("== /ess/simulate ==")
r = client.post("/ess/simulate", json={"hourly_curtailment_mwh": [10, 30, 5, 0, 40], "rated_power_mw": 22.5,
                                       "method": "hourly_capped"})
b = r.json()
check("hourly_capped 계산", r.status_code == 200 and b["total_absorbed_mwh"] == 60.0 and b["total_curtailment_mwh"] == 85.0, r.text)

# 저장용량 제약 방식(기본) — 40MWh 저장용량에 SoC 10~90%면 가용 32MWh, 효율 0.9라 충전 손실이 걸려
# 총 85MWh를 다 받지 못한다. 정격출력만 보는 hourly_capped(60MWh)보다 반드시 작아야 한다.
r = client.post("/ess/simulate", json={"hourly_curtailment_mwh": [10, 30, 5, 0, 40], "rated_power_mw": 22.5,
                                       "energy_capacity_mwh": 40})
b = r.json()
check("storage_constrained 기본 method", r.status_code == 200 and b["method"] == "storage_constrained", r.text)
check("저장용량 제약이 상한보다 작음", b["total_absorbed_mwh"] < 60.0 and b["hours_full"] >= 1, r.text)
check("가용용량 = 저장용량 x SoC 폭", b["usable_capacity_mwh"] == 32.0, r.text)

r = client.post("/ess/simulate", json={"hourly_curtailment_mwh": [10], "soc_min": 0.9, "soc_max": 0.1})
check("SoC 역전 -> INVALID_SOC_RANGE", r.status_code == 422 and r.json()["error_code"] == "INVALID_SOC_RANGE", r.text)

r = client.post("/ess/simulate", json={"hourly_curtailment_mwh": [10, 20], "hour_of_day": [13]})
check("hour_of_day 길이 불일치 -> LENGTH_MISMATCH", r.status_code == 422 and r.json()["error_code"] == "LENGTH_MISMATCH", r.text)

print("\n모든 점검 통과")
