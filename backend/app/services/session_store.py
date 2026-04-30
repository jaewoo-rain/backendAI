"""
SQLite-backed session record store (Stage 8).

설계
  - stdlib sqlite3 사용. 새 dep 없음. 작은 prototype 에 충분.
  - 모든 메서드는 sync — SessionManager 가 asyncio.to_thread() 로 감싸서
    이벤트 루프를 안 막음.
  - 매 호출마다 새 connection — sqlite3.Connection 이 thread-safe 보장
    안 하므로, to_thread 가 아무 worker 에서 도는 걸 안전하게 받기 위함.
  - WAL 모드는 켜지 않음. 워크로드가 매우 가볍고 (수 ~ 수십 row) 쿼리도
    짧아서 default rollback journal 로도 충분.

Schema 는 한 번 정해진 뒤 변경 안 됨. 스키마 변경 시 `data/sessions.db`
파일을 수동 삭제하는 게 upgrade path. (production 이라면 alembic 같은
migration tool 도입.)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.schemas.session import Session


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    container_id    TEXT NOT NULL,
    container_name  TEXT NOT NULL,
    ratio           REAL NOT NULL,
    quota_bytes     INTEGER,
    image           TEXT NOT NULL,
    command_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    exit_code       INTEGER,
    gpu_index       INTEGER
);
"""

# Stage 9 minimal: 기존 v1 DB (gpu_index 없음) 를 위한 idempotent migration.
# SQLite 는 ADD COLUMN IF NOT EXISTS 미지원이라 PRAGMA table_info 로 검사.
_MIGRATIONS = [
    ("gpu_index", "ALTER TABLE sessions ADD COLUMN gpu_index INTEGER"),
]

_SELECT_COLS = (
    "id, container_id, container_name, ratio, quota_bytes, "
    "image, command_json, created_at, status, exit_code, gpu_index"
)


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.execute(SCHEMA_SQL)
            # 기존 v1 DB 가 있으면 누락 컬럼 추가 — idempotent.
            existing = {row[1] for row in c.execute("PRAGMA table_info(sessions)")}
            for col, ddl in _MIGRATIONS:
                if col not in existing:
                    c.execute(ddl)

    def _conn(self) -> sqlite3.Connection:
        # isolation_level=None = autocommit. 단순 단일 statement 만 실행하므로
        # 명시적 트랜잭션 관리 안 해도 안전.
        return sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)

    @staticmethod
    def _row_to_session(row: tuple) -> Session:
        return Session(
            id=row[0],
            container_id=row[1],
            container_name=row[2],
            ratio=row[3],
            quota_bytes=row[4],
            image=row[5],
            command=json.loads(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            status=row[8],
            exit_code=row[9],
            gpu_index=row[10],
        )

    # ---- CRUD --------------------------------------------------------- #
    def insert(self, s: Session) -> None:
        with closing(self._conn()) as c:
            c.execute(
                "INSERT INTO sessions ("
                + _SELECT_COLS
                + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s.id,
                    s.container_id,
                    s.container_name,
                    s.ratio,
                    s.quota_bytes,
                    s.image,
                    json.dumps(s.command),
                    s.created_at.isoformat(),
                    s.status,
                    s.exit_code,
                    s.gpu_index,
                ),
            )

    def get(self, sid: str) -> Optional[Session]:
        with closing(self._conn()) as c:
            row = c.execute(
                f"SELECT {_SELECT_COLS} FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
        return None if row is None else self._row_to_session(row)

    def list_all(self) -> list[Session]:
        with closing(self._conn()) as c:
            rows = c.execute(
                f"SELECT {_SELECT_COLS} FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def update_status(
        self, sid: str, status: str, exit_code: Optional[int]
    ) -> None:
        with closing(self._conn()) as c:
            c.execute(
                "UPDATE sessions SET status = ?, exit_code = ? WHERE id = ?",
                (status, exit_code, sid),
            )

    def delete(self, sid: str) -> bool:
        with closing(self._conn()) as c:
            cur = c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        return cur.rowcount > 0
