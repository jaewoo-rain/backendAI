"""
FastAPI app factory.

실행:
  cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

설정:
  scripts/run_backend.sh 가 venv + env 셋업까지 알아서 해 줌.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.sessions import router as sessions_router
from app.core.config import get_settings
from app.services.docker_manager import DockerManager
from app.services.session_manager import SessionManager
from app.services.session_store import SessionStore


# Stage 5-B: 단일 HTML UI. StaticFiles 마운트 안 하고 한 줄 라우트로 충분.
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


logger = logging.getLogger("fgpu.backend")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="fGPU Backend",
        version="0.1.0",
        description="Spawns user containers with libfgpu.so injected via LD_PRELOAD.",
    )

    # 시작 시점에 host hook 경로 존재 여부 확인 (실패해도 부팅은 함 — 경고만).
    if not os.path.isfile(settings.host_hook_path):
        logger.warning(
            "host hook .so not found at %s — sessions will fail to start until "
            "scripts/build_hook.sh produces it.",
            settings.host_hook_path,
        )

    docker_mgr = DockerManager(
        host_hook_path=settings.host_hook_path,
        container_hook_path=settings.container_hook_path,
        runtime_image=settings.runtime_image,
    )
    session_store = SessionStore(settings.db_path)
    session_mgr = SessionManager(
        docker_manager=docker_mgr,
        runtime_image=settings.runtime_image,
        default_command=settings.default_command,
        store=session_store,
    )
    app.state.docker_manager = docker_mgr
    app.state.session_store = session_store
    app.state.session_manager = session_mgr
    # Stage 9 minimal: bearer token 인증. 빈 문자열이면 인증 비활성.
    app.state.api_token = settings.api_token

    if settings.api_token:
        logger.info("FGPU_API_TOKEN set → /sessions routes require Bearer auth")
    else:
        logger.info("FGPU_API_TOKEN not set → /sessions routes are unauthenticated")

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "ok": True,
            "runtime_image": settings.runtime_image,
            "host_hook_path": settings.host_hook_path,
            "host_hook_exists": os.path.isfile(settings.host_hook_path),
            "db_path": settings.db_path,
            "db_exists": os.path.isfile(settings.db_path),
            "auth_enabled": bool(settings.api_token),
        }

    @app.get("/", include_in_schema=False)
    def ui_index() -> FileResponse:
        return FileResponse(STATIC_INDEX)

    app.include_router(sessions_router)
    return app


app = create_app()
