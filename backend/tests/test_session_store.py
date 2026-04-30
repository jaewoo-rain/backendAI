"""
Stage 8 SessionStore 단위 테스트 — 순수 SQLite CRUD round-trip.

실행 (backend/ 에서):
    pip install -e ".[dev]"
    pytest

커버리지
  - SessionStore 의 insert / get / list_all / update_status / delete.
  - datetime / list[str] / Optional[int] 의 SQLite round-trip 보존.
  - 같은 DB 파일을 두 SessionStore 가 차례로 열었을 때의 일관성
    (백엔드 재시작 시나리오의 lower bound).

커버리지 밖 (의도적)
  - SessionManager 의 asyncio.to_thread + docker SDK 호출 — docker daemon
    + GPU 가 필요한 integration 영역. scripts/eval/run_isolation.sh,
    run_correlation.sh 등이 그 역할.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from app.schemas.session import Session
from app.services.session_store import SessionStore


def _make(
    sid: str = "abc",
    *,
    status: str = "created",
    ratio: float = 0.4,
    exit_code: Optional[int] = None,
    created_at: Optional[datetime] = None,
) -> Session:
    return Session(
        id=sid,
        container_id=f"cid-{sid}",
        container_name=f"fgpu-{sid}",
        ratio=ratio,
        quota_bytes=None,
        image="fgpu-runtime:stage2",
        command=["/opt/fgpu/test_alloc"],
        created_at=created_at or datetime.now(timezone.utc),
        status=status,
        exit_code=exit_code,
    )


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def test_insert_and_get_preserves_all_fields(store: SessionStore) -> None:
    fixed_ts = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    s = _make(sid="abc", ratio=0.7, exit_code=0, created_at=fixed_ts)
    store.insert(s)

    got = store.get("abc")
    assert got is not None
    assert got.id == "abc"
    assert got.container_id == "cid-abc"
    assert got.container_name == "fgpu-abc"
    assert got.ratio == pytest.approx(0.7)
    assert got.image == "fgpu-runtime:stage2"
    assert got.command == ["/opt/fgpu/test_alloc"]
    assert got.exit_code == 0
    assert got.created_at == fixed_ts  # datetime ISO round-trip


def test_get_missing_returns_none(store: SessionStore) -> None:
    assert store.get("does-not-exist") is None


def test_list_all_orders_newest_first(store: SessionStore) -> None:
    early = _make(sid="early",
                  created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = _make(sid="late",
                 created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
    store.insert(early)
    store.insert(late)

    ids = [s.id for s in store.list_all()]
    assert ids == ["late", "early"]


def test_update_status_persists(store: SessionStore) -> None:
    s = _make(sid="x")
    store.insert(s)
    store.update_status("x", "exited", 0)

    got = store.get("x")
    assert got is not None
    assert got.status == "exited"
    assert got.exit_code == 0


def test_update_status_with_none_exit_code(store: SessionStore) -> None:
    """status 만 바꾸고 exit_code 는 None 유지하는 경로 (예: removed)."""
    s = _make(sid="r")
    store.insert(s)
    store.update_status("r", "removed", None)

    got = store.get("r")
    assert got is not None
    assert got.status == "removed"
    assert got.exit_code is None


def test_delete_returns_true_when_present(store: SessionStore) -> None:
    store.insert(_make(sid="x"))
    assert store.delete("x") is True
    assert store.get("x") is None


def test_delete_returns_false_when_missing(store: SessionStore) -> None:
    assert store.delete("never-existed") is False


def test_reopen_same_db_sees_persisted_records(tmp_path) -> None:
    """백엔드 재시작 시뮬: 같은 DB 파일을 새 인스턴스가 열어 같은 데이터 봄."""
    db = tmp_path / "shared.db"
    fixed_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)

    s1 = SessionStore(db)
    s1.insert(_make(sid="persist", ratio=0.55, created_at=fixed_ts))

    # 새 인스턴스 = 새 connection. 같은 파일이므로 같은 데이터.
    s2 = SessionStore(db)
    got = s2.get("persist")
    assert got is not None
    assert got.id == "persist"
    assert got.ratio == pytest.approx(0.55)
    assert got.created_at == fixed_ts
