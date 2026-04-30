"""
설정 (env 로 override 가능).

사용 가능한 환경변수 (모두 FGPU_ 접두사):
  FGPU_RUNTIME_IMAGE       기본 'fgpu-runtime:stage2'
  FGPU_HOST_HOOK_PATH      기본: <repo>/build/libfgpu.so 자동 탐지
  FGPU_CONTAINER_HOOK_PATH 기본 '/opt/fgpu/libfgpu.so'
  FGPU_DEFAULT_COMMAND     기본 '["/opt/fgpu/test_alloc"]' (JSON 형식)
  FGPU_DB_PATH             기본: <repo>/data/sessions.db (Stage 8 persistence)
  FGPU_API_TOKEN           기본 '' (= 인증 비활성). 값 설정 시 /sessions
                           라우트가 'Authorization: Bearer <token>' 요구.
                           /healthz 와 / (UI) 는 항상 public.

호스트 hook 경로(libfgpu.so)는 docker -v 마운트로 컨테이너에 들어간다.
백엔드 프로세스 자체는 GPU 권한이 필요 없으며, docker socket 만 필요.
DB 파일은 백엔드 재시작 시 세션 record 를 복원하는 용도 — 컨테이너 자체는
docker daemon 이 들고 있으므로 reconcile 로 status 를 다시 맞춘다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_repo_root() -> str:
    """이 파일 위치 기준으로 repo root (newback/) 를 추정."""
    # __file__ = .../backend/app/core/config.py
    here = os.path.abspath(__file__)
    # 4 단계 위로: core -> app -> backend -> newback
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


class Settings(BaseSettings):
    runtime_image: str = "fgpu-runtime:stage2"
    container_hook_path: str = "/opt/fgpu/libfgpu.so"
    host_hook_path: str = ""  # 빈 값이면 아래 get_settings 에서 auto-detect
    default_command: list[str] = Field(default_factory=lambda: ["/opt/fgpu/test_alloc"])
    db_path: str = ""        # 빈 값이면 <repo>/data/sessions.db 자동 설정
    api_token: str = ""      # 빈 값이면 인증 비활성 (Stage 9 최소 단계)

    model_config = SettingsConfigDict(env_prefix="FGPU_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.host_hook_path:
        s.host_hook_path = os.path.join(_default_repo_root(), "build", "libfgpu.so")
    if not s.db_path:
        s.db_path = os.path.join(_default_repo_root(), "data", "sessions.db")
    return s
