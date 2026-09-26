import asyncio
import re
import uuid

from mandri.runtime.docker_client import DockerClient
from mandri.runtime.errors.docker import DockerExecutionError


class DockerVersions:
    def __init__(self, client: DockerClient, owner: str) -> None:
        self._client = client
        self._owner = owner
        self._cache: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def version(self, image: str, harness: str) -> str:
        if harness not in {"codex", "claude", "opencode", "agy", "pi"}:
            raise DockerExecutionError("docker_image_incompatible", "Unknown native executable")
        async with self._lock:
            key = image, harness
            if key not in self._cache:
                self._cache[key] = await self._probe(image, harness)
            return self._cache[key]

    async def _probe(self, image: str, harness: str) -> str:
        name = f"mandri-version-{uuid.uuid4().hex}"
        try:
            result = await self._client.run(
                "run",
                "--rm",
                "--name",
                name,
                "--label",
                "io.mandri.managed=true",
                "--label",
                f"io.mandri.owner={self._owner}",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "1000:1000",
                "--memory",
                "512m",
                "--pids-limit",
                "64",
                "--cpus",
                "1",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=64m",
                "--tmpfs",
                "/home/worker:rw,nosuid,nodev,size=16m,uid=1000,gid=1000",
                "--env",
                "HOME=/home/worker",
                "--entrypoint",
                harness,
                image,
                "--version",
            )
            match = re.search(r"\b(\d+\.\d+\.\d+)\b", result)
            if match is None:
                raise DockerExecutionError(
                    "docker_image_incompatible", "Native executable version is unavailable"
                )
            return match[1]
        finally:
            await self._client.remove([name])
