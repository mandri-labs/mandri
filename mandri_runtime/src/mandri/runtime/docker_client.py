import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mandri.runtime.errors.docker import DockerExecutionError

_CLIENT_ENV = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "HOME", "USERPROFILE")


class DockerClient:
    def __init__(self, binary: str = "docker", timeout: float = 60.0) -> None:
        self.binary = binary
        self.timeout = timeout

    def environment(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = {key: os.environ[key] for key in _CLIENT_ENV if key in os.environ}
        if extra:
            env.update(extra)
        return env

    async def run(
        self,
        *args: str,
        env: Mapping[str, str] | None = None,
        check: bool = True,
        include_stderr: bool = False,
    ) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                self.binary,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.environment(env),
            )
        except OSError as error:
            raise DockerExecutionError(
                "docker_unavailable", "Docker could not be started"
            ) from error
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), self.timeout)
        except BaseException as error:
            if process.returncode is None:
                process.kill()
            await process.wait()
            if isinstance(error, TimeoutError):
                raise DockerExecutionError(
                    "docker_operation_timed_out", "The Docker operation timed out"
                ) from None
            raise
        if process.returncode and check:
            if args[:2] == ("image", "inspect") and b"no such image" in stderr.lower():
                raise DockerExecutionError("docker_image_missing", "The worker image is not cached")
            raise DockerExecutionError(
                "docker_unavailable",
                f"Docker {args[0]} failed: {stderr.decode(errors='replace').strip()}",
            )
        return (stdout + stderr if include_stderr else stdout).decode().strip()

    async def pull(self, reference: str, timeout: float) -> None:
        await self._acquire(["image", "pull", "--platform", "linux/amd64", reference], timeout)

    async def build(self, context: Path, reference: str, label: str, timeout: float) -> None:
        await self._acquire(
            [
                "build",
                "--platform",
                "linux/amd64",
                "--label",
                label,
                "--tag",
                reference,
                str(context),
            ],
            timeout,
        )

    async def _acquire(self, args: list[str], timeout: float) -> None:
        operation = "build" if args[0] == "build" else "pull"
        try:
            process = await asyncio.create_subprocess_exec(
                self.binary,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=self.environment(),
            )
        except OSError:
            raise DockerExecutionError(
                "docker_unavailable", "Docker could not be started"
            ) from None
        try:
            async with asyncio.timeout(timeout):
                stderr, code = await asyncio.gather(_capture_error(process.stderr), process.wait())
        except BaseException as error:
            if process.returncode is None:
                process.kill()
            await process.wait()
            if isinstance(error, TimeoutError):
                raise DockerExecutionError(
                    f"docker_image_{operation}_timeout", "Worker image acquisition timed out"
                ) from None
            raise
        if code:
            if any(
                text in stderr.lower()
                for text in (
                    b"cannot connect to the docker daemon",
                    b"is the docker daemon running",
                    b"error during connect",
                    b"docker.sock",
                )
            ):
                raise DockerExecutionError(
                    "docker_unavailable",
                    "The Docker engine became unavailable during image acquisition",
                )
            raise DockerExecutionError(
                f"docker_image_{operation}_failed", "Worker image acquisition failed"
            )

    async def json(self, *args: str) -> Any:
        try:
            return json.loads(await self.run(*args))
        except ValueError as error:
            raise DockerExecutionError(
                "docker_unavailable", "Docker returned invalid data"
            ) from error

    async def remove(self, containers: Sequence[str], network: str | None = None) -> None:
        for container in containers:
            await self.run("rm", "--force", container, check=False)
            selector = f"id={container}" if len(container) == 64 else f"name=^{container}$"
            if await self.run("container", "ls", "-aq", "--filter", selector):
                raise DockerExecutionError(
                    "docker_cleanup_failed", "An owned worker container could not be removed"
                )
        if network:
            await self.run("network", "rm", network, check=False)
            if await self.run("network", "ls", "-q", "--filter", f"name=^{network}$"):
                raise DockerExecutionError(
                    "docker_cleanup_failed", "An owned worker network could not be removed"
                )


async def _capture_error(stream: asyncio.StreamReader | None) -> bytes:
    captured = bytearray()
    if stream is None:
        return bytes(captured)
    while chunk := await stream.read(16_384):
        captured.extend(chunk[: max(0, 16_384 - len(captured))])
    return bytes(captured)
