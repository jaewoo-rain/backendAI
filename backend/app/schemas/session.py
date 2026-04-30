"""
세션 관련 Pydantic 모델.

설계 메모: Session 클래스는 *내부 record* 와 *API 응답* 을 겸한다.
Stage 3 MVP 에서 별도 DTO 를 둘 만큼 변환 로직이 필요하지 않으므로
모델 하나로 통합. Stage 8 (Redis 저장) 에서 분리할 가능성 있음.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """POST /sessions body."""
    ratio: float = Field(..., gt=0.0, le=1.0,
                         description="GPU 메모리 비율 (0.0 < ratio <= 1.0)")
    command: Optional[list[str]] = Field(
        default=None,
        description="컨테이너 내부에서 실행할 명령. None 이면 이미지 default CMD.",
    )
    image: Optional[str] = Field(
        default=None,
        description="사용할 docker image. None 이면 settings.runtime_image.",
    )
    quota_bytes: Optional[int] = Field(
        default=None, gt=0,
        description="절대 quota (bytes). 설정 시 ratio 보다 우선.",
    )
    gpu_index: Optional[int] = Field(
        default=None, ge=0,
        description="멀티-GPU 호스트에서 특정 device 만 노출. None 이면 모든 GPU.",
    )


class Session(BaseModel):
    """세션 record + API 응답 공용."""
    id: str
    container_id: str
    container_name: str
    status: str = "created"        # docker container status
    ratio: float
    quota_bytes: Optional[int] = None
    image: str
    command: list[str]
    created_at: datetime
    exit_code: Optional[int] = None
    gpu_index: Optional[int] = None  # Stage 9 minimal — 사용된 GPU device id


class SessionLogs(BaseModel):
    id: str
    logs: str
