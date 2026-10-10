import contextlib
import threading

from mandri.cli.daemon_client import request
from mandri.cli.run_errors import DaemonUnreachableError
from mandri.core.ids import HarnessKind
from mandri.core.types.mcp import McpBinding


class CliMcpBinding:
    def __init__(self, url: str, harness: HarnessKind) -> None:
        self.url = url
        self.binding: McpBinding | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if harness is HarnessKind.AGY:
            return
        try:
            response = request(
                "POST", f"{url}/v1/mcp/bindings", body={"harness": harness.value}, timeout=1
            )
            if response.is_success:
                self.binding = McpBinding.model_validate(response.json())
        except (DaemonUnreachableError, ValueError):
            return
        if self.binding is not None:
            self._thread = threading.Thread(target=self._renew, daemon=True)
            self._thread.start()

    def _renew(self) -> None:
        while not self._stop.wait(30):
            if self.binding is not None:
                with contextlib.suppress(DaemonUnreachableError):
                    request("PUT", f"{self.url}/v1/mcp/bindings/{self.binding.id}", timeout=2)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self.binding is not None:
            with contextlib.suppress(DaemonUnreachableError):
                request("DELETE", f"{self.url}/v1/mcp/bindings/{self.binding.id}", timeout=2)
