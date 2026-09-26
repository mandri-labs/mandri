"""ControlSink over a managed child process stdin."""

from typing import final

from mandri.runtime.process import ManagedProcess


@final
class ProcessStdinSink:
    def __init__(self, process: ManagedProcess) -> None:
        self._process = process
        self._buffer = bytearray()

    def write(self, data: bytes) -> None:
        self._buffer.extend(data)

    async def drain(self) -> None:
        data = bytes(self._buffer)
        self._buffer.clear()
        if data:
            await self._process.write_stdin(data)
