import asyncio
import json
from typing import Any

from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.process import ManagedProcess


def response_data(record: dict[str, Any], command: str) -> dict[str, Any]:
    if record.get("command") != command:
        raise ControlTransportError("Pi returned a mismatched command response")
    if record.get("success") is not True:
        error = record.get("error")
        raise ControlTransportError(str(error or f"Pi rejected {command}"))
    data = record.get("data")
    return data if isinstance(data, dict) else {}


class PiRpcConnection:
    def __init__(self, process: ManagedProcess) -> None:
        self._process = process
        self._request_id = 0
        self._lock = asyncio.Lock()

    async def send(self, record: dict[str, Any]) -> None:
        await self._process.write_stdin((json.dumps(record) + "\n").encode("utf-8"))

    async def call(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._request_id += 1
            identifier = str(self._request_id)
            await self.send({**params, "type": command, "id": identifier})
            while line := await self._process.read_stdout_line():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (
                    isinstance(record, dict)
                    and record.get("type") == "response"
                    and record.get("id") == identifier
                ):
                    return response_data(record, command)
            raise ControlTransportError("Pi closed its control stream")
