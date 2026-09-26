import contextlib
import os
import signal
import time
from collections.abc import Callable, Iterator
from types import FrameType

import uvicorn

FORCE_WINDOW_SECONDS = 5.0


class DaemonServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class InterruptController:
    def __init__(self, server: uvicorn.Server, kill_children: Callable[[], None]) -> None:
        self.server = server
        self.kill_children = kill_children
        self.last_interrupt: float | None = None

    def handle(self, sig: int, frame: FrameType | None) -> None:
        self.server.should_exit = True
        if sig == signal.SIGINT:
            now = time.monotonic()
            if (
                self.last_interrupt is not None
                and now - self.last_interrupt <= FORCE_WINDOW_SECONDS
            ):
                try:
                    self.kill_children()
                finally:
                    os._exit(130)
            self.last_interrupt = now
            with contextlib.suppress(OSError):
                os.write(2, b"Shutting down. Press Ctrl+C again within 5 seconds to force exit.\n")

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        previous = {sig: signal.signal(sig, self.handle) for sig in signals}
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
