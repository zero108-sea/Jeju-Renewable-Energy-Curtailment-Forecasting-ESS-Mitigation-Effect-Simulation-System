"""Backend ↔ AI 서버 REST 계약 — docs/통신규격_확정서_v1.1.md가 약속한 동작을 고정한다.

계약 문서와 구현이 갈리면 연동이 조용히 깨진다. 문서에 적은 응답 모양·에러코드·조건부 필수
규칙을 여기서 검증하므로, 이 파일이 깨지면 문서도 함께 고쳐야 한다.

Spring 쪽에서 같은 케이스를 호출해 응답을 비교하면 연동 테스트가 된다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import needs_artifacts

client = TestClient(app, raise_server_exceptions=False)
FULL = [{"hour": h, "temp": 15.0, "cloud": 4.0, "wind_speed": 6.0, "solar_rad": 0.5}
        for h in range(1, 25)]


def body(**kw) -> dict:
    return {"energy_type": "wind", "region": "제주", "target_date": "2026-10-01",
            "weather": FULL, **kw}


def post(**kw):
    return client.post("/predict", json=body(**kw))


class TestErrorShape:
    """모든 4xx는 {error_code, message} 한 가지 모양이다."""

    @pytest.mark.parametrize("payload,code", [
        ({"weather": FULL[:23]}, "INVALID_HOUR_SET"),
        ({"weather": FULL[:23] + [FULL[0]]}, "INVALID_HOUR_SET"),
        ({"demand_forecast_mw": [600.0] * 12}, "INVALID_HOUR_SET"),
        ({"region": "   "}, "MISSING_REQUIRED_FIELD"),
        ({"energy_type": "windd"}, "VALIDATION_ERROR"),
    ])
    def test_에러코드와_모양(self, payload, code):
        r = post(**payload)
        assert r.status_code == 422
        j = r.json()
        assert set(j) == {"error_code", "message"}, j
        assert j["error_code"] == code, j
        assert j["message"], "메시지가 비어 있으면 릴레이해도 쓸모가 없다"

    def test_형식오류는_필드_위치를_알려준다(self):
        """pydantic 기본 메시지는 'Field required'뿐이라 어느 필드인지 알 수 없다 (v1.1에서 추가)."""
        r = client.post("/predict", json={"energy_type": "wind", "region": "제주",
                                          "target_date": "2026-10-01"})
        assert r.status_code == 422
        assert "위치: weather" in r.json()["message"], r.json()


@needs_artifacts
class TestConditionalRequired:
    """Optional은 '널을 보내도 된다'가 아니다 — energy_type이 필수를 결정한다."""

    @pytest.mark.parametrize("energy_type,drop", [
        ("wind", "wind_speed"),
        ("solar", "solar_rad"), ("solar", "temp"), ("solar", "cloud"),
    ])
    def test_발전원별_필수값이_null이면_거부(self, energy_type, drop):
        w = [{**x, drop: None} for x in FULL]
        r = post(energy_type=energy_type, weather=w)
        assert r.status_code == 422
        j = r.json()
        assert j["error_code"] == "MISSING_REQUIRED_FIELD"
        assert drop in j["message"], j

    @pytest.mark.parametrize("energy_type,drop", [
        ("wind", "solar_rad"), ("wind", "cloud"), ("solar", "wind_speed"),
    ])
    def test_그_발전원에_불필요한_값은_null이어도_통과(self, energy_type, drop):
        r = post(energy_type=energy_type, weather=[{**x, drop: None} for x in FULL])
        assert r.status_code == 200, r.json()

    def test_누락_위치를_시각과_필드로_알려준다(self):
        w = FULL[:12] + [{**x, "wind_speed": None} for x in FULL[12:]]
        r = post(weather=w)
        assert "13시.wind_speed" in r.json()["message"], r.json()

    def test_null을_0으로_대체하지_않는다(self):
        """풍속 null을 0으로 바꾸면 '제어 없음'으로 오예측한다 — 실제로 겪어 고친 동작이다."""
        r = post(weather=[{**x, "wind_speed": None} for x in FULL])
        assert r.status_code == 422, "조용히 통과하면 안 된다"


@needs_artifacts
class TestRegion:
    """자유 문자열, 모델은 쓰지 않고 그대로 echo. 빈 값만 거부."""

    @pytest.mark.parametrize("v", ["제주", "남원읍", "제주특별자치도 서귀포시 남원읍", "JEJU"])
    def test_어떤_문자열이든_받고_그대로_돌려준다(self, v):
        r = post(region=v)
        assert r.status_code == 200 and r.json()["region"] == v

    @pytest.mark.parametrize("v", ["", "   ", "x" * 101])
    def test_빈_값과_과길이는_거부(self, v):
        assert post(region=v).status_code == 422

    def test_region이_예측을_바꾸지_않는다(self):
        """제주 전역 단일 모델이므로 지역으로 분기하지 않는다."""
        a = post(region="제주").json()
        b = post(region="남원읍").json()
        assert [h["curtailment_probability"] for h in a["hourly"]] == \
               [h["curtailment_probability"] for h in b["hourly"]]


@needs_artifacts
class TestResponseShape:
    def test_필수_키가_모두_있다(self):
        j = post().json()
        assert {"energy_type", "region", "target_date", "hourly", "model_used"} <= set(j)
        assert len(j["hourly"]) == 24
        assert {"hour", "generation_forecast_mwh", "curtailment_probability"} <= set(j["hourly"][0])

    def test_모델_학습시각이_실린다(self):
        """model_used는 경로 이름이라 재학습해도 안 바뀐다 — 추적에는 학습 시각이 필요하다.

        예측 결과를 저장한 뒤 '이 수치는 어느 모델이 낸 것인가'를 되물을 때 유일한 단서다.
        """
        from datetime import datetime

        j = post().json()
        assert j.get("model_trained_at"), j
        datetime.fromisoformat(j["model_trained_at"])        # ISO 8601이어야 한다

    def test_학습시각이_아티팩트와_일치한다(self):
        from app.model_io import load_artifact

        j = post().json()
        meta = load_artifact(j["model_used"])["meta"]
        assert j["model_trained_at"] == meta["trained_at"], (j["model_trained_at"], meta["trained_at"])

    def test_신제도_구간이면_기댓값의_근거가_약하다고_알린다(self):
        """확률 크기가 검증되지 않은 구간(2024-06~)에서 기댓값을 곱하면 근거가 약하다.

        값을 null로 바꾸면 Backend 계약이 깨지므로 note로 알린다.
        """
        new = post(target_date="2026-10-07", demand_forecast_mw=[620.0] * 24).json()
        old = post(target_date="2023-06-01", demand_forecast_mw=[620.0] * 24).json()
        assert "Brier" in (new.get("note") or ""), new.get("note")
        assert "Brier" not in (old.get("note") or ""), old.get("note")

    def test_태양광은_expected_curtailment가_null이다(self):
        """07장 사유 — 태양광은 제어량이 집계되지 않아 2단계 모델이 없다."""
        j = post(energy_type="solar", demand_forecast_mw=[600.0] * 24).json()
        assert all(h["expected_curtailment_mwh"] is None for h in j["hourly"])

    def test_태양광_기상값을_함께_주면_침투율_모델로_전환된다(self):
        with_solar = post(demand_forecast_mw=[600.0] * 24).json()["model_used"]
        without = post(demand_forecast_mw=[600.0] * 24,
                       weather=[{"hour": x["hour"], "wind_speed": x["wind_speed"]} for x in FULL]
                       ).json()["model_used"]
        assert "crossp" in with_solar and "crossp" not in without


class TestValueRange:
    """물리적으로 불가능한 값은 200이 아니라 422여야 한다 (v1.2).

    범위를 넣기 전에는 solar_rad=-1, wind_speed=500이 200으로 통과해 그럴듯한 숫자를 돌려줬다.
    드리프트 경고가 note에 실리긴 했지만 HTTP는 200이라, Backend가 note를 파싱하지 않으면
    쓰레기 예측이 DB와 화면까지 간다.

    **거부(422)는 모델 없이도 검증된다** — pydantic이 모델 로드 전에 막기 때문이다.
    통과(200)를 확인하는 테스트만 아티팩트가 필요해 아래 클래스로 분리했다.
    """

    @pytest.mark.parametrize("field,value", [
        ("solar_rad", -1.0), ("solar_rad", 999.0),
        ("temp", 999.0), ("temp", -273.0),
        ("cloud", 99.0), ("cloud", -1.0),
        ("wind_speed", -5.0), ("wind_speed", 500.0),
    ])
    def test_범위_밖_기상값은_422(self, field, value):
        w = [{**x, field: value} for x in FULL]
        r = post(weather=w)
        assert r.status_code == 422, f"{field}={value}가 통과했다: {r.json()}"
        assert field in r.json()["message"], r.json()

    def test_수요가_0이하면_422(self):
        r = post(demand_forecast_mw=[-100.0] + [620.0] * 23)
        assert r.status_code == 422
        assert r.json()["error_code"] == "OUT_OF_RANGE"
        assert "1시=-100" in r.json()["message"], r.json()


@needs_artifacts
class TestValueRangeAccepts:
    """실측 범위 안의 값은 200이어야 한다 — 상한을 잘못 조여 정상 예보를 막는 일을 잡는다.

    200을 확인하려면 예측이 끝까지 돌아야 하므로 학습 아티팩트가 필요하다.
    CI에는 models/가 없어(gitignore) skip된다 — 거부 쪽은 TestValueRange가 모델 없이 검증한다.
    """

    @pytest.mark.parametrize("field,value", [
        ("solar_rad", 0.0), ("solar_rad", 3.9),      # 제주 ASOS 184 실측 최대 3.9
        ("temp", -3.1), ("temp", 37.0),              # 실측 −3.1~37.0
        ("cloud", 0.0), ("cloud", 10.0),
        ("wind_speed", 0.0), ("wind_speed", 13.7),   # 실측 최대 13.7
    ])
    def test_실측_범위_안의_값은_통과한다(self, field, value):
        r = post(weather=[{**x, field: value} for x in FULL])
        assert r.status_code == 200, f"{field}={value}가 거부됐다: {r.json()}"


class TestEssValueRange:
    def _ess(self, **kw):
        return client.post("/ess/simulate", json={"hourly_curtailment_mwh": [10.0] * 24, **kw})

    def test_정상(self):
        assert self._ess().status_code == 200

    @pytest.mark.parametrize("kw", [
        {"hourly_curtailment_mwh": []},
        {"rated_power_mw": 0},
        {"rated_power_mw": -10},
    ])
    def test_빈배열과_0이하_정격출력은_422(self, kw):
        """정격출력 -10은 예전에 흡수율 −1.0을 돌려줬다 — 의미 없는 값이다."""
        assert self._ess(**kw).status_code == 422

    def test_음수_제어량은_422(self):
        r = self._ess(hourly_curtailment_mwh=[-5.0] * 24)
        assert r.status_code == 422 and r.json()["error_code"] == "OUT_OF_RANGE"


class TestHealth:
    """상수 ok는 헬스체크 역할을 못 한다 — 모델 적재 여부를 보고해야 한다 (v1.2).

    지연 로딩이던 때는 /health가 ok인데 첫 predict가 실패할 수 있었다.
    """

    def test_200이고_필수_키가_있다(self):
        r = client.get("/health")
        assert r.status_code == 200
        j = r.json()
        assert {"status", "loaded", "missing", "ready"} <= set(j), j

    @needs_artifacts
    def test_아티팩트가_있으면_ready다(self):
        with TestClient(app) as c:          # lifespan을 실제로 돌려 워밍업을 태운다
            j = c.get("/health").json()
        assert j["ready"] is True and j["status"] == "ok", j
        assert j["missing"] == [] and j["loaded"] > 0, j

    def test_ready와_status가_일관된다(self):
        j = client.get("/health").json()
        assert (j["status"] == "ok") == (j["ready"] is True), j
        assert (j["missing"] == []) == (j["ready"] is True), j
