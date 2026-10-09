"""KIM 검증 비교 — 화면과 CLI가 **같은 숫자**를 말한다는 계약을 고정한다.

이 계산은 원래 `scripts/kim_validation_log.py`의 report() 안에서 print와 섞여 있었다.
대시보드에도 같은 표가 필요해지면서 복제냐 공유냐를 골라야 했고, 공유를 택했다.
복제했다면 한쪽만 고치는 날 **화면과 CLI가 다른 숫자를 말하고, 어느 쪽도 못 믿게 된다.**
그 선택이 유지되는지를 여기서 지킨다 — 누군가 로직을 되돌려 붙여 넣으면 깨져야 한다.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from app.services import kim_validation as V

LOG_EXISTS = os.path.exists(V.LOG)
needs_log = pytest.mark.skipif(
    not LOG_EXISTS, reason="수집 로그가 없습니다 — kim_validation_log.py collect를 먼저 돌리세요")


def test_CLI가_공용_모듈을_쓴다():
    """report()가 자기 사본을 쓰면 이 테스트가 깨진다."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "kim_validation_log.py"), encoding="utf-8").read()
    assert "from app.services.kim_validation import" in src, "CLI가 공용 모듈을 쓰지 않습니다"
    # 계산이 스크립트로 되돌아온 흔적 — 복제의 신호다
    assert "def _estimate_with_observed_cloud" not in src, "계산이 스크립트에 복제됐습니다"
    assert src.count("일적산비") == 0, "지표 계산이 스크립트에 복제됐습니다"


def test_표본_경고_기준이_한_곳에만_있다():
    """화면과 CLI가 서로 다른 기준으로 '표본이 적다'고 하면 안 된다."""
    assert V.MIN_DAYS == 10
    app = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "dashboard", "app.py"), encoding="utf-8").read()
    assert "too_few" in app, "대시보드가 공용 경고 기준(too_few)을 쓰지 않습니다"


@needs_log
def test_비교표_모양():
    c = V.compare()
    if c is None:
        pytest.skip("실측과 겹치는 시간이 아직 없습니다")
    assert list(c.summary.columns) == ["경로", "MAE", "NMAE(%)", "편향", "상관", "일적산비"]
    assert list(c.daily.columns) == ["target_date", "KIM", "추정(예보운량)", "추정(실측운량)", "실측"]
    assert c.n_days >= 1 and c.n_hours >= 1
    assert c.n_days <= c.log_days, "겹치는 날이 수집한 날보다 많을 수 없습니다"


@needs_log
def test_야간은_빠진다():
    """양쪽 다 0인 밤을 넣으면 MAE가 희석돼 성능이 실제보다 좋아 보인다."""
    c = V.compare()
    if c is None:
        pytest.skip("실측과 겹치는 시간이 아직 없습니다")
    assert c.n_hours < c.n_days * 24, "주간만 남기는 필터가 동작하지 않습니다"


@needs_log
def test_네_경로가_모두_평가된다():
    """'추정(실측 운량)'이 빠지면 오차가 모델 탓인지 예보 탓인지 가를 수 없다."""
    c = V.compare()
    if c is None:
        pytest.skip("실측과 겹치는 시간이 아직 없습니다")
    assert set(c.summary["경로"]) == {n for n, _ in V.PATHS}


def test_로그가_없으면_None(tmp_path):
    """없는 파일에 예외를 던지면 대시보드 탭 전체가 죽는다."""
    assert V.compare(str(tmp_path / "없는파일.csv")) is None


def test_중복은_제거하고_세어_둔다(tmp_path, monkeypatch):
    """collect를 두 번 돌리면 같은 (거래일, 분석시각, 시각)이 겹친다. 조용히 두 번 세면 안 된다."""
    p = tmp_path / "log.csv"
    row = {"target_date": "2026-09-28", "hour": 12, "tmfc": "2026092618",
           "kim_mj": 1.0, "kim_inst_mj": 1.0, "est_mj": 1.0,
           "wind80": 5.0, "temp_c": 20.0, "collected_at": "2026-09-27T10:00"}
    pd.DataFrame([row, row]).to_csv(p, index=False)
    monkeypatch.setattr(V, "estimate_with_observed_cloud",
                        lambda a, d: pd.Series(0.0, index=d.index))
    monkeypatch.setattr("app.data_prep.load_asos",
                        lambda s="184": pd.DataFrame({
                            "dt": [pd.Timestamp("2026-09-28 12:00")], "solar_rad": [1.0],
                            "cloud": [5.0], "temp": [20.0], "wind_speed": [5.0],
                            "humidity": [60.0]}))
    c = V.compare(str(p))
    assert c is not None and c.dup_dropped == 1
    assert c.n_hours == 1, "중복이 두 번 세어졌습니다"
