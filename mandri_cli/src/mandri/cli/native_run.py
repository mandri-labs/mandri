import os
import subprocess
import sys
from pathlib import Path

from mandri.cli.daemon_client import base_url, error_message, request
from mandri.cli.daemon_command import DaemonCommand
from mandri.cli.run_errors import RunError
from mandri.cli.types import RunSpec
from mandri.core.native_run import NativeRunPlan, NativeRunStart
from mandri.core.types.execution import ExecutionBackend, PrivacyMode


class NativeRunCommand(DaemonCommand):
    def __init__(self, spec: RunSpec) -> None:
        super().__init__()
        self._spec = spec

    def run(self) -> int:
        address = self._ensure_running(self._spec.base_dir)
        spec = NativeRunStart(
            harness=self._spec.harness,
            model=self._spec.model_arg,
            cwd=str((self._spec.cwd or Path.cwd()).resolve()),
            effort=self._spec.effort,
            execution_backend=self._spec.execution_backend or ExecutionBackend.HOST,
            privacy_mode=self._spec.privacy_mode or PrivacyMode.NONE,
            args=list(self._spec.passthrough_args),
            tty=sys.stdin.isatty() and sys.stdout.isatty(),
            term=os.environ.get("TERM", "xterm-256color"),
        )
        url = base_url(address) + "/v1/runtime/runs"
        with self._console.status("Preparing harness..."):
            response = request("POST", url, body=spec.model_dump(mode="json"), timeout=1800)
        if response.status_code >= 400:
            raise RunError(error_message(response))
        plan = NativeRunPlan.model_validate(response.json())
        try:
            process = subprocess.Popen(
                plan.argv,
                cwd=self._spec.cwd,
                env=plan.env if spec.execution_backend is ExecutionBackend.HOST else None,
            )
            try:
                return process.wait()
            except KeyboardInterrupt:
                try:
                    return process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
                    return 130
        finally:
            response = request("DELETE", f"{url}/{plan.id}", timeout=60)
            if response.status_code >= 400:
                raise RunError(error_message(response))
