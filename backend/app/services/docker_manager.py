"""
Docker SDK 얇은 래퍼.

책임:
  - fGPU 컨테이너를 spawn 할 때 필요한 모든 옵션 (--gpus all, hook .so 마운트,
    FGPU_RATIO/LD_PRELOAD env) 을 한 곳에서 조립.
  - 컨테이너 status / logs 조회, stop / remove.

세션 ID 와 컨테이너 ID 의 관계는 SessionManager 에서 관리.
"""

from __future__ import annotations

import os
from typing import Optional
import docker
from docker.types import DeviceRequest


# 백엔드 프로세스의 env 에 설정돼 있으면 컨테이너로도 그대로 전달되는 변수.
# 5-A 확장 (run_correlation.sh) 에서 launch counter dump 주기를 컨테이너
# 별로 통제하기 위함. 명시적 화이트리스트라 임의 env leak 방지.
_PASSTHROUGH_ENV = ("FGPU_LAUNCH_LOG_EVERY",)


class DockerManager:
    def __init__(
        self,
        host_hook_path: str,
        container_hook_path: str,
        runtime_image: str,
    ) -> None:
        self.client = docker.from_env()
        self.host_hook_path = host_hook_path
        self.container_hook_path = container_hook_path
        self.runtime_image = runtime_image

    # ---- spawn ------------------------------------------------------- #
    def create_container(
        self,
        name: str,
        ratio: float,
        command: list[str],
        quota_bytes: Optional[int] = None,
        image: Optional[str] = None,
        gpu_index: Optional[int] = None,
    ):
        env = {
            "FGPU_RATIO": str(ratio),
            "LD_PRELOAD": self.container_hook_path,
        }
        if quota_bytes is not None:
            env["FGPU_QUOTA_BYTES"] = str(quota_bytes)

        # 백엔드 프로세스 env 에 화이트리스트 변수가 있으면 컨테이너로 forward.
        # 운영자가 `FGPU_LAUNCH_LOG_EVERY=500 ./scripts/run_backend.sh` 로
        # 띄우면 이후 모든 세션이 그 값을 상속.
        for key in _PASSTHROUGH_ENV:
            v = os.environ.get(key)
            if v is not None and key not in env:
                env[key] = v

        # docker run --gpus all 또는 --gpus device=N 패턴.
        # gpu_index=None → count=-1 (전 GPU 노출, 기본 동작).
        # gpu_index=N    → device_ids=["N"] (멀티-GPU 호스트에서 특정 디바이스만).
        if gpu_index is None:
            device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
        else:
            device_requests = [
                DeviceRequest(device_ids=[str(gpu_index)],
                              capabilities=[["gpu"]])
            ]

        # host 의 libfgpu.so 를 컨테이너 안 hook 경로에 read-only 로 bind mount.
        volumes = {
            self.host_hook_path: {
                "bind": self.container_hook_path,
                "mode": "ro",
            }
        }

        return self.client.containers.run(
            image=image or self.runtime_image,
            command=command,
            name=name,
            detach=True,
            remove=False,            # 종료 후에도 logs 조회 가능하도록 보존
            device_requests=device_requests,
            volumes=volumes,
            environment=env,
        )

    # ---- query ------------------------------------------------------- #
    def get_status(self, container_id: str) -> tuple[str, Optional[int]]:
        c = self.client.containers.get(container_id)
        c.reload()
        return c.status, c.attrs.get("State", {}).get("ExitCode")

    def get_logs(self, container_id: str, tail: int = 200) -> str:
        c = self.client.containers.get(container_id)
        # stdout + stderr 합쳐서 가져온다 — entrypoint, [fgpu], [test] 모두 포함.
        raw = c.logs(stdout=True, stderr=True, tail=tail)
        return raw.decode("utf-8", errors="replace")

    # ---- lifecycle --------------------------------------------------- #
    def stop_container(self, container_id: str, timeout: int = 10) -> None:
        c = self.client.containers.get(container_id)
        c.stop(timeout=timeout)

    def remove_container(self, container_id: str, force: bool = True) -> None:
        c = self.client.containers.get(container_id)
        c.remove(force=force)
