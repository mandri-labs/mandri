import asyncio
import contextlib
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from mandri.runtime.control.agents.base import HttpCall
from mandri.runtime.control.errors import ControlTransportError


class OpencodeDiffRecovery:
    def __init__(self, call: HttpCall, publish: Callable[[dict[str, Any]], None]) -> None:
        self._call = call
        self._publish = publish
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, owner: str, turn: str) -> None:
        task = asyncio.create_task(self._recover(owner, turn), name=f"opencode-diff:{turn}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _recover(self, owner: str, turn: str) -> None:
        with contextlib.suppress(ControlTransportError, TimeoutError, ValueError):
            async with asyncio.timeout(10):
                response = await self._call(
                    "GET",
                    f"/api/session/{quote(owner, safe='')}/diff?from={quote(turn, safe='')}",
                    None,
                )
            if response.status_code != 200:
                return
            payload = response.json()
            files = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(files, list) or not files:
                return
            self._publish(
                {
                    "type": "session.diff",
                    "properties": {
                        "sessionID": owner,
                        "diff": files,
                        "scope": "turn",
                        "turnID": turn,
                    },
                }
            )
