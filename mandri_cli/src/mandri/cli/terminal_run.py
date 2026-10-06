import asyncio
import json
import os
import shutil
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mandri.cli.daemon_client import base_url
from mandri.cli.daemon_command import DaemonCommand
from mandri.cli.run_errors import RunError
from mandri.cli.types import RunSpec
from mandri.core.terminal import TerminalSize, TerminalStart
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

if sys.platform != "win32":
    import termios
    import tty


def terminal_size() -> TerminalSize:
    size = shutil.get_terminal_size()
    return TerminalSize(rows=min(size.lines, 1000), columns=min(size.columns, 1000))


def mark_ready(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


@contextmanager
def raw_terminal() -> Iterator[None]:
    if sys.platform != "win32":
        fd = sys.stdin.fileno()
        settings = termios.tcgetattr(fd) if os.isatty(fd) else None
        try:
            if settings is not None:
                tty.setraw(fd)
            yield
        finally:
            if settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, settings)
    else:
        raise RunError("Managed terminals require a POSIX host")


async def forward_input(connection: ClientConnection) -> None:
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    while True:
        ready: asyncio.Future[None] = loop.create_future()
        try:
            loop.add_reader(fd, mark_ready, ready)
        except PermissionError:
            chunk = await asyncio.to_thread(os.read, fd, 65536)
        else:
            try:
                await ready
                chunk = os.read(fd, 65536)
            finally:
                loop.remove_reader(fd)
        await connection.send(chunk or b"\x04")
        if not chunk:
            return


async def forward_output(connection: ClientConnection) -> int:
    async for frame in connection:
        if isinstance(frame, bytes):
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()
        else:
            message = json.loads(frame)
            if message["type"] == "exit":
                return int(message["code"])
            if message["type"] == "error":
                raise RunError(message["message"])
    raise RunError("Terminal connection closed before the harness exited")


async def run_terminal(url: str, spec: TerminalStart) -> int:
    if sys.platform != "win32":
        try:
            async with connect(url, max_size=20 * 1024 * 1024, proxy=None) as connection:
                await connection.send(spec.model_dump_json())
                first = await connection.recv()
                if not isinstance(first, str):
                    raise RunError("Invalid terminal startup response")
                message = json.loads(first)
                if message["type"] != "started":
                    raise RunError(message.get("message", "Terminal could not start"))
                loop = asyncio.get_running_loop()
                resized = asyncio.Event()

                async def resize() -> None:
                    while True:
                        await resized.wait()
                        resized.clear()
                        await connection.send(terminal_size().model_dump_json())

                loop.add_signal_handler(signal.SIGWINCH, resized.set)
                resized.set()
                with raw_terminal():
                    output = asyncio.create_task(forward_output(connection))
                    tasks: list[asyncio.Task[int] | asyncio.Task[None]] = [
                        asyncio.create_task(forward_input(connection)),
                        output,
                        asyncio.create_task(resize()),
                    ]
                    try:
                        pending = set(tasks)
                        while output in pending:
                            done, pending = await asyncio.wait(
                                pending, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done:
                                task.result()
                        return output.result()
                    finally:
                        loop.remove_signal_handler(signal.SIGWINCH)
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
        except (OSError, WebSocketException) as error:
            raise RunError(f"Terminal connection failed: {error}") from error
    else:
        raise RunError("Managed terminals require a POSIX host")


class TerminalRunCommand(DaemonCommand):
    def __init__(self, spec: RunSpec) -> None:
        super().__init__()
        self._spec = spec

    def run(self) -> int:
        if sys.platform == "win32":
            raise RunError("Managed terminals require a POSIX host")
        address = self._ensure_running(self._spec.base_dir)
        size = terminal_size()
        spec = TerminalStart(
            harness=self._spec.harness,
            model=self._spec.model_arg,
            cwd=str((self._spec.cwd or Path.cwd()).resolve()),
            effort=self._spec.effort,
            execution_backend=self._spec.execution_backend or ExecutionBackend.HOST,
            privacy_mode=self._spec.privacy_mode or PrivacyMode.NONE,
            args=list(self._spec.passthrough_args),
            rows=size.rows,
            columns=size.columns,
            term=os.environ.get("TERM", "xterm-256color"),
        )
        url = base_url(address).replace("http://", "ws://", 1) + "/v1/runtime/terminal"
        return asyncio.run(run_terminal(url, spec))
