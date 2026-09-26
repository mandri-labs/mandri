import json
from typing import Any

from mandri.runtime.control.errors import ControlTransportError, ThreadOwnershipError
from mandri.runtime.process import ManagedProcess


class StdioRpcConnection:
    def __init__(self, process: ManagedProcess) -> None:
        self.process = process
        self.request_id = 0

    async def send(self, message: dict[str, Any]) -> None:
        await self.process.write_stdin((json.dumps(message) + "\n").encode())

    async def call(
        self, method: str, params: dict[str, Any], *, claude: bool = False
    ) -> dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        if claude:
            await self.send(
                {
                    "type": "control_request",
                    "request_id": str(request_id),
                    "request": {"subtype": method, **params},
                }
            )
        else:
            await self.send({"id": request_id, "method": method, "params": params})
        while line := await self.process.read_stdout_line():
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            if not isinstance(frame, dict):
                continue
            if claude:
                response = frame.get("response", {})
                if not isinstance(response, dict) or response.get("request_id") != str(request_id):
                    continue
                if response.get("subtype") != "success":
                    raise ControlTransportError("Claude rejected control request")
                return dict(response.get("response") or {})
            if frame.get("id") != request_id:
                continue
            if "error" in frame:
                error = frame["error"]
                message = error.get("message", "") if isinstance(error, dict) else ""
                if method == "thread/resume" and "active writer" in str(message):
                    raise ThreadOwnershipError("Native conversation has an active writer")
                raise ControlTransportError(f"Codex rejected {method}")
            return dict(frame.get("result") or {})
        raise ControlTransportError("Native harness closed its output")
