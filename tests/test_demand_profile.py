"""24시간 수요 프로파일 — 분모를 평평하게 두지 않는다는 계약.

침투율 = 발전량 ÷ 수요다. 분모의 모양이 결과를 바꾸는데, 시연 화면은 오랫동안 슬라이더
한 값을 24번 복제해 넣고 있었다. 여기서 고정하는 것은 두 가지다.
  ① 배수의 평균이 정확히 1.0일 것 — 그래야 사용자가 고른 값이 그대로 '일평균 수요'다
  ② 실측을 못 읽어도 **예외로 화면을 죽이지 않고** 평탄으로 후퇴하고 그 사실을 말할 것
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services import demand_profile as D


def _fake_demand(n_days: int = 60, peak: float = 700.0, base: float = 500.0) -> pd.DataFrame:
    """19시가 최고, 4시가 최저인 합성 수요 — 실측 데이터 없이도 계약을 시험한다."""
    dts = pd.date_range("2023-10-01 01:00", periods=n_days * 24, freq="h")
    h = pd.Series(dts).dt.hour.replace(0, 24)
    mw = base + (peak - base) * ((h - 4) % 24 / 15).clip(0, 1)
    return pd.DataFrame({"dt": dts, "demand_mw": mw.to_numpy()})


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr("app.data_prep.load_demand_actual", lambda: _fake_demand())


def test_배수의_평균은_정확히_1(fake):
    """평균이 1이 아니면 슬라이더 값과 실제 일평균이 어긋난다 — 조용히 틀린다."""
    p, _ = D.profile(10, False)
    assert len(p) == 24
    assert sum(p) / 24 == pytest.approx(1.0, abs=1e-3)


def test_곡선은_평평하지_않다(fake):
    """이 기능의 존재 이유 자체다."""
    c, _ = D.curve(600.0, 10, False)
    assert len(c) == 24
    assert max(c) / min(c) > 1.1, "수요 곡선이 사실상 평평합니다"


def test_수준이_일평균으로_들어간다(fake):
    c, _ = D.curve(600.0, 10, False)
    assert sum(c) / 24 == pytest.approx(600.0, rel=1e-3)


def test_출처를_함께_돌려준다(fake):
    """화면이 '이게 예보'라고 오해하게 두면 안 된다 — 출처 문구가 비면 안 된다."""
    _, src = D.profile(10, False)
    assert "실측" in src and "10월" in src


def test_표본이_적으면_평탄으로_후퇴한다(monkeypatch):
    monkeypatch.setattr("app.data_prep.load_demand_actual", lambda: _fake_demand(n_days=3))
    p, src = D.profile(10, False)
    assert p == [1.0] * 24
    assert "평탄" in src and "표본 부족" in src


def test_데이터가_없어도_예외를_던지지_않는다(monkeypatch):
    """실측 파일이 없는 환경에서 화면 전체가 죽으면 시연이 끝난다."""
    def boom():
        raise FileNotFoundError("no demand files")
    monkeypatch.setattr("app.data_prep.load_demand_actual", boom)
    p, src = D.profile(10, False)
    assert p == [1.0] * 24
    assert "평탄" in src


def test_평일과_주말을_가른다(monkeypatch):
    seen = []

    def cap():
        d = _fake_demand()
        seen.append(d)
        return d
    monkeypatch.setattr("app.data_prep.load_demand_actual", cap)
    _, wd = D.profile(10, False)
    _, we = D.profile(10, True)
    assert "평일" in wd and "주말" in we


@pytest.mark.skipif(True, reason="실측 데이터 의존 — 로컬에서만 의미가 있다")
def test_실측_프로파일_모양():  # pragma: no cover
    p, src = D.profile(10, False)
    assert p.index(min(p)) + 1 in range(2, 7)      # 최저는 새벽
    assert p.index(max(p)) + 1 in range(17, 22)    # 최고는 저녁
