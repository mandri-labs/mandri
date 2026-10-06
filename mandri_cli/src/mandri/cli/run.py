"""Run command plan composition and foreground harness execution."""

import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from codex_cli_bin import bundled_codex_path
from mandri.cli.agy_run import AgyRunProfile, close_agy_run, prepare_agy_run, wait_agy_process
from mandri.cli.daemon_client import base_url, error_message, request
from mandri.cli.daemon_command import DaemonCommand
from mandri.cli.run_errors import (
    DaemonUnreachableError,
    HarnessBinaryNotFoundError,
    ModelResolutionError,
    RunError,
)
from mandri.cli.terminal_run import TerminalRunCommand
from mandri.cli.types import RunSpec
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.ids import HARNESS_WIRE_FORMATS, HarnessKind
from mandri.core.launch import ModelMetadata, build_harness_launch, merged_env
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.sessions.agy_binary import find_agy_binary


@dataclass(frozen=True)
class RunPlan:
    """Everything needed to exec a harness against a dedicated gateway route."""

    route_id: str
    model: str
    token: str
    env: dict[str, str]
    args: tuple[str, ...] = ()


class RunCommand(DaemonCommand):
    """Compose a run plan and exec the harness against it."""

    def __init__(self, spec: RunSpec) -> None:
        super().__init__()
        self._spec = spec
        self._plan: RunPlan | None = None
        self._binary: Path | None = None
        self._agy_profile: AgyRunProfile | None = None

    def run(self) -> int:
        defaults = TomlConfigAdapter(self._spec.base_dir).load().defaults
        backend = self._spec.execution_backend or defaults.execution_backend
        privacy = self._spec.privacy_mode or defaults.privacy_mode
        if backend is not ExecutionBackend.HOST or privacy is not PrivacyMode.NONE:
            self._validate_model_arg(self._spec.model_arg)
            return TerminalRunCommand(
                replace(self._spec, execution_backend=backend, privacy_mode=privacy)
            ).run()
        self._validate_model_arg(self._spec.model_arg)
        self._binary = self._resolve_binary()
        if self._spec.harness is HarnessKind.AGY:
            self._agy_profile = prepare_agy_run(self._spec)
        try:
            self._plan = self._build_run_plan()
            return self._exec_harness()
        finally:
            try:
                if self._plan is not None:
                    self._delete_route()
            finally:
                if self._agy_profile is not None:
                    close_agy_run(self._agy_profile)

    def _build_run_plan(self) -> RunPlan:
        spec = self._spec
        self._validate_model_arg(spec.model_arg)
        address = self._ensure_running(spec.base_dir)
        url = base_url(address)
        with self._console.status("Preparing harness..."):
            route_id, token = self._create_route(url, spec.model_arg, spec.harness, spec.effort)
            metadata = self._fetch_route_metadata(url, spec.model_arg)
        launch = build_harness_launch(
            spec.harness,
            address.port,
            route_id,
            spec.model_arg,
            token,
            metadata,
            effort=spec.effort,
        )
        return RunPlan(
            route_id=route_id,
            model=spec.model_arg,
            token=token,
            env=merged_env(os.environ, launch),
            args=launch.args,
        )

    @staticmethod
    def _fetch_route_metadata(url: str, model_arg: str) -> ModelMetadata | None:
        try:
            response = request(
                "GET", f"{url}/v1/gateway/model-metadata", params={"model": model_arg}
            )
        except DaemonUnreachableError:
            return None
        if response.status_code >= 400:
            return None
        try:
            return ModelMetadata.from_payload(response.json())
        except ValueError:
            return None

    @staticmethod
    def _validate_model_arg(model: str) -> None:
        provider_name, separator, model_id = model.partition("/")
        if not separator or not provider_name.strip() or not model_id.strip():
            raise ModelResolutionError(f"model must be '<provider>/<model-id>', got {model!r}")

    @staticmethod
    def _create_route(
        base_url: str, model: str, harness: HarnessKind, effort: str | None = None
    ) -> tuple[str, str]:
        body: dict[str, str | list[str]] = {
            "model": model,
            "formats": [f.value for f in HARNESS_WIRE_FORMATS[harness]],
        }
        if effort is not None:
            body["effort"] = effort
        response = request("POST", f"{base_url}/v1/gateway/routes", body=body)
        if response.status_code == 404:
            raise ModelResolutionError(error_message(response))
        if response.status_code >= 400:
            raise RunError(f"route creation failed: {error_message(response)}")
        payload = response.json()
        return str(payload["id"]), str(payload["child_token"])

    def _resolve_binary(self) -> Path:
        harness = self._spec.harness
        if harness is HarnessKind.AGY:
            binary = find_agy_binary()
            if binary is None:
                raise HarnessBinaryNotFoundError("Antigravity CLI (agy) is not installed")
            return binary
        if harness is HarnessKind.CODEX:
            try:
                return bundled_codex_path()
            except FileNotFoundError as error:
                raise HarnessBinaryNotFoundError("Pinned Codex runtime is unavailable") from error
        found = shutil.which(harness.value)
        if found is None:
            raise HarnessBinaryNotFoundError(f"harness binary not found on PATH: {harness.value}")
        return Path(found)

    def _harness_command(self) -> list[str]:
        assert self._binary is not None and self._plan is not None
        return [
            str(self._binary),
            *self._plan.args,
            *(self._agy_profile.args if self._agy_profile else self._spec.passthrough_args),
        ]

    def _log_harness_start(self) -> None:
        model = self._spec.model_arg
        label = {
            HarnessKind.CLAUDE: "[orange1]claude[/orange1]",
            HarnessKind.CODEX: "[deep_sky_blue1]codex[/deep_sky_blue1]",
            HarnessKind.OPENCODE: "opencode",
        }.get(self._spec.harness, self._spec.harness.value)
        self._console.log(f"starting {label} with model {model}")

    def _exec_harness(self) -> int:
        assert self._plan is not None
        try:
            self._log_harness_start()
            process = subprocess.Popen(
                self._harness_command(),
                env=self._plan.env,
                cwd=self._spec.cwd,
                stdin=None,
                stdout=None,
                stderr=None,
            )
            if self._agy_profile:
                return wait_agy_process(process, self._agy_profile)
            return process.wait()
        finally:
            self._console.print()

    def _delete_route(self) -> None:
        assert self._address is not None and self._plan is not None
        url = f"{base_url(self._address)}/v1/gateway/routes/{self._plan.route_id}"
        try:
            request("DELETE", url)
        except DaemonUnreachableError:
            return
