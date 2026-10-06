import asyncio
import contextlib
import errno
import os
import signal
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from mandri.core.terminal import TerminalSize
from mandri.core.types.execution import ProtectionError
from mandri.runtime.errors import ProcessIOError, ProcessSpawnError
from mandri.runtime.process import ManagedProcess, ProcessGroupTerminator

if sys.platform != "win32":
    import fcntl
    import pty
    import termios


class TerminalReader(asyncio.StreamReader):
    def set_exception(self, exc: Exception) -> None:
        if isinstance(exc, OSError) and exc.errno == errno.EIO:
            self.feed_eof()
        else:
            super().set_exception(exc)


def mark_ready(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class TerminalProcess(ManagedProcess):
    def __init__(
        self, process: asyncio.subprocess.Process, master: int, transport: asyncio.ReadTransport
    ) -> None:
        super().__init__(process, tree_terminator=ProcessGroupTerminator(process.pid))
        self._master = master
        self._transport = transport
        self._closed = False

    async def write_stdin(self, data: bytes) -> None:
        loop = asyncio.get_running_loop()
        while data:
            try:
                written = os.write(self._master, data)
                data = data[written:]
            except BlockingIOError:
                ready: asyncio.Future[None] = loop.create_future()
                loop.add_writer(self._master, mark_ready, ready)
                try:
                    await ready
                finally:
                    loop.remove_writer(self._master)
            except OSError as error:
                raise ProcessIOError("Terminal input is unavailable") from error

    def resize(self, size: TerminalSize) -> None:
        if sys.platform != "win32":
            if self._closed or self.returncode is not None:
                return
            fcntl.ioctl(
                self._master, termios.TIOCSWINSZ, struct.pack("HHHH", size.rows, size.columns, 0, 0)
            )
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.process.pid, signal.SIGWINCH)

        else:
            raise ProtectionError("terminal_unavailable", "Managed terminals require a POSIX host")

    def close_terminal(self) -> None:
        if not self._closed:
            self._closed = True
            self._transport.close()
            os.close(self._master)

    async def stop(self, grace: float = 5.0) -> int:
        try:
            return await super().stop(grace)
        finally:
            self.close_terminal()

    async def kill(self) -> int:
        try:
            return await super().kill()
        finally:
            self.close_terminal()


async def spawn_terminal(
    argv: Sequence[str],
    size: TerminalSize,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> TerminalProcess:
    if sys.platform != "win32":
        master, slave = pty.openpty()
        process = None
        transport = None
        try:
            fcntl.ioctl(
                slave, termios.TIOCSWINSZ, struct.pack("HHHH", size.rows, size.columns, 0, 0)
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "mandri.runtime.terminal_exec",
                *argv,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=cwd,
                env=dict(env) if env is not None else None,
                start_new_session=True,
            )
            reader = TerminalReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: protocol, os.fdopen(os.dup(master), "rb", buffering=0)
            )
            process.stdout = reader
            os.set_blocking(master, False)
            return TerminalProcess(process, master, transport)
        except BaseException as error:
            if process is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            if transport is not None:
                transport.close()
            os.close(master)
            if isinstance(error, (OSError, ValueError)):
                raise ProcessSpawnError(f"Terminal spawn failed for {argv[0]!r}") from error
            raise
        finally:
            os.close(slave)
    else:
        raise ProtectionError("terminal_unavailable", "Managed terminals require a POSIX host")
