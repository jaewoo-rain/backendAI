"""
세션 라이프사이클 매니저 (Stage 8).

저장소: SQLite via SessionStore (이전엔 in-memory dict). 백엔드가 재시작
되어도 세션 record 는 살아남고, lazy reconcile 로 docker daemon 의
실제 status 를 맞춰줌.

비동기성: docker SDK 와 sqlite3 는 모두 sync 라이브러리. 이벤트 루프를
막지 않기 위해 모든 blocking 호출을 asyncio.to_thread() 로 감쌈.
이전엔 한 POST /sessions 가 docker.run() 동안 다른 요청을 막았는데,
이제 진짜 동시 처리 가능. (5-A 격리 실험의 두 컨테이너 spawn 도
실제로 병렬화됨.)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import docker.errors

from app.schemas.session import Session
from app.services.docker_manager import DockerManager
from app.services.session_store import SessionStore


class SessionManager:
    def __init__(
        self,
        docker_manager: DockerManager,
        runtime_image: str,
        default_command: list[str],
        store: SessionStore,
    ) -> None:
        self.docker = docker_manager
        self.runtime_image = runtime_image
        self.default_command = default_command
        self.store = store

    # ---- create ------------------------------------------------------- #
    async def create(
        self,
        ratio: float,
        command: Optional[list[str]] = None,
        quota_bytes: Optional[int] = None,
        image: Optional[str] = None,
        gpu_index: Optional[int] = None,
    ) -> Session:
        sid = uuid.uuid4().hex[:12]
        name = f"fgpu-{sid}"
        cmd = command or list(self.default_command)
        img = image or self.runtime_image

        # docker SDK 는 sync 라 to_thread.
        c = await asyncio.to_thread(
            self.docker.create_container,
            name=name,
            ratio=ratio,
            command=cmd,
            quota_bytes=quota_bytes,
            image=img,
            gpu_index=gpu_index,
        )
        rec = Session(
            id=sid,
            container_id=c.id,
            container_name=name,
            ratio=ratio,
            quota_bytes=quota_bytes,
            image=img,
            command=cmd,
            created_at=datetime.now(timezone.utc),
            status=c.status or "created",
            gpu_index=gpu_index,
        )
        await asyncio.to_thread(self.store.insert, rec)
        return rec

    # ---- read -------------------------------------------------------- #
    async def get(self, sid: str) -> Optional[Session]:
        rec = await asyncio.to_thread(self.store.get, sid)
        if rec is None:
            return None
        # docker daemon 에 status reconcile.
        try:
            status, exit_code = await asyncio.to_thread(
                self.docker.get_status, rec.container_id
            )
        except docker.errors.NotFound:
            # 컨테이너가 daemon 에서 사라짐 — record 는 보존, 상태만 갱신.
            if rec.status != "removed":
                await asyncio.to_thread(
                    self.store.update_status, sid, "removed", rec.exit_code
                )
            rec.status = "removed"
            return rec

        if rec.status != status or rec.exit_code != exit_code:
            await asyncio.to_thread(
                self.store.update_status, sid, status, exit_code
            )
        rec.status = status
        rec.exit_code = exit_code
        return rec

    async def list_all(self) -> list[Session]:
        recs = await asyncio.to_thread(self.store.list_all)
        # 각 레코드 reconcile — 동시에 진행해 list 응답 latency 감소.
        results = await asyncio.gather(
            *(self.get(r.id) for r in recs), return_exceptions=False
        )
        return [r for r in results if r is not None]

    async def get_logs(self, sid: str, tail: int = 200) -> Optional[str]:
        rec = await self.get(sid)
        if rec is None:
            return None
        try:
            return await asyncio.to_thread(
                self.docker.get_logs, rec.container_id, tail
            )
        except docker.errors.NotFound:
            return ""

    # ---- mutate ------------------------------------------------------ #
    async def stop(self, sid: str) -> Optional[Session]:
        rec = await self.get(sid)
        if rec is None:
            return None
        try:
            await asyncio.to_thread(self.docker.stop_container, rec.container_id)
        except docker.errors.NotFound:
            pass
        return await self.get(sid)

    async def delete(self, sid: str) -> bool:
        rec = await asyncio.to_thread(self.store.get, sid)
        if rec is None:
            return False
        try:
            await asyncio.to_thread(
                self.docker.remove_container, rec.container_id, True
            )
        except docker.errors.NotFound:
            pass
        await asyncio.to_thread(self.store.delete, sid)
        return True
