import asyncio
import contextlib
import subprocess
from collections.abc import Awaitable, Callable

from mandri.runtime.docker_client import DockerClient
from mandri.runtime.process import ManagedProcess


class DockerProcess(ManagedProcess):
    def __init__(
        self,
        attach: ManagedProcess,
        client: DockerClient,
        container_id: str,
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(attach.process)
        self.container_id = container_id
        self._attach = attach
        self._client = client
        self._cleanup = cleanup
        self._exit_code: int | None = None
        self.oom_killed = False
        self.execution_context: dict[str, str] = {}
        self.native_argv: list[str] = []
        self._stopped = False
        self._cleaned = False
        self._force_stopping = False
        self._watcher = asyncio.create_task(self._observe_exit())

    @property
    def returncode(self) -> int | None:
        return self._exit_code

    async def _observe_exit(self) -> int:
        await self._attach.wait()
        try:
            state = await self._client.json(
                "inspect", "--format", "{{json .State}}", self.container_id
            )
            if state.get("Running"):
                if self._force_stopping:
                    await self._client.run("kill", self.container_id)
                else:
                    await self._client.run("stop", "--time", "5", self.container_id)
                state = await self._client.json(
                    "inspect", "--format", "{{json .State}}", self.container_id
                )
            self.oom_killed = bool(state.get("OOMKilled"))
            self._exit_code = int(state["ExitCode"])
        except Exception:
            self._exit_code = 125
        return self._exit_code

    async def wait(self) -> int:
        return await asyncio.shield(self._watcher)

    async def stop(self, grace: float = 5.0) -> int:
        if self._stopped:
            if not self._cleaned:
                await self._cleanup()
                self._cleaned = True
            return self._exit_code if self._exit_code is not None else 125
        self._stopped = True
        try:
            if self._exit_code is None:
                await self._close_stdin()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(self._watcher), grace)
                if self._exit_code is None:
                    await self._client.run(
                        "stop", "--time", str(max(1, int(grace))), self.container_id, check=False
                    )
                if self._exit_code is None:
                    await self._client.run("kill", self.container_id, check=False)
                await self._attach.stop(grace)
            return await self.wait()
        finally:
            await self._cleanup()
            self._cleaned = True

    async def interrupt(self) -> bool:
        if self.returncode is not None:
            return False
        await self._client.run("kill", "--signal", "SIGINT", self.container_id)
        return True

    async def kill(self) -> int:
        if self._cleaned:
            return self._exit_code if self._exit_code is not None else 125
        self._force_stopping = True
        await self._client.run("kill", self.container_id, check=False)
        await self._attach.kill()
        code = await self.wait()
        self._stopped = True
        await self._cleanup()
        self._cleaned = True
        return code

    def kill_now(self) -> None:
        subprocess.Popen(
            [self._client.binary, "kill", self.container_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._client.environment(),
        )
