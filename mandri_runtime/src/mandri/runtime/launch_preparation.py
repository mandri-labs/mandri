import os
import socket
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.pi import managed_extension_path
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway.model_metadata import ModelMetadata
from mandri.runtime import wiring
from mandri.runtime.codex_catalog import materialize_catalog
from mandri.runtime.control import modes
from mandri.runtime.native_launch import native_launch
from mandri.runtime.protected_launch import protected_launch
from mandri.sessions.pi_store import PiSessionStore


@dataclass(frozen=True)
class PreparedLaunch:
    argv: list[str]
    env: dict[str, str]
    listen_port: int | None


class LaunchPreparation:
    def __init__(
        self,
        gateway_port: int | None,
        token_issuer: Callable[[str], str] | None,
        parent_env: Mapping[str, str] | None = None,
    ) -> None:
        self._gateway_port = gateway_port
        self._token_issuer = token_issuer
        self._parent_env = parent_env
        self._catalogs: tempfile.TemporaryDirectory[str] | None = None

    def prepare(
        self,
        command: list[str],
        kind: HarnessKind | None,
        model: str,
        effort: str | None,
        native: bool,
        route_id: str | None,
        metadata: ModelMetadata | None,
        mode: modes.LaunchMode | None,
        env_wiring: Mapping[str, str] | None = None,
        resume_native_id: HarnessSessionId | None = None,
        privacy_mode: PrivacyMode = PrivacyMode.NONE,
    ) -> PreparedLaunch:
        if privacy_mode is PrivacyMode.SURROGATE and (
            native
            or kind is None
            or self._gateway_port is None
            or self._token_issuer is None
            or route_id is None
        ):
            raise ProtectionError(
                "privacy_state_unavailable", "Protected helpers require a scoped gateway launch"
            )
        if kind is HarnessKind.PI and resume_native_id and self._parent_env is None:
            path = PiSessionStore().resolve(str(resume_native_id))
            if path is not None:
                resume_native_id = HarnessSessionId(str(path))
        plan: wiring.HarnessLaunch | None = None
        if native and kind is not None:
            plan = native_launch(kind, model, effort, resume_native_id)
        elif (
            kind is not None
            and self._gateway_port is not None
            and self._token_issuer is not None
            and route_id is not None
        ):
            token = self._token_issuer(route_id)
            plan = wiring.build_harness_launch(
                kind,
                self._gateway_port,
                route_id,
                model,
                token,
                metadata,
                mode.opencode_config if mode is not None else None,
                resume_native_id=resume_native_id
                if kind in (HarnessKind.CLAUDE, HarnessKind.AGY, HarnessKind.PI)
                else None,
                effort=effort,
            )
            if privacy_mode is PrivacyMode.SURROGATE:
                plan = protected_launch(plan, kind, self._gateway_port, route_id, token, model)
        if kind is HarnessKind.PI:
            command = [*command, "--extension", managed_extension_path()]
        env = self.child_env(plan, env_wiring)
        if kind is HarnessKind.PI:
            env["MANDRI_PI_PERMISSION_MODE"] = mode.mode if mode and mode.mode else "default"
        argv, listen_port = self.expand_args(command, route_id)
        argv = [*argv, *mode.claude_args] if mode else argv
        if plan is not None:
            argv = [*argv, *plan.args]
        if self._parent_env is None and "MANDRI_CODEX_MODEL_CATALOG" in env:
            if self._catalogs is None:
                self._catalogs = tempfile.TemporaryDirectory(prefix="mandri-codex-models-")
            materialize_catalog(argv, env, Path(self._catalogs.name))
        return PreparedLaunch(argv, env, listen_port)

    def expand_args(self, command: list[str], route_id: str | None) -> tuple[list[str], int | None]:
        expanded: list[str] = []
        listen_port: int | None = None
        for arg in command:
            if "{listen_port}" in arg:
                listen_port = ephemeral_port()
                arg = arg.replace("{listen_port}", str(listen_port))
            if "{gateway_port}" in arg:
                arg = arg.replace("{gateway_port}", str(self._gateway_port))
            if "{route_id}" in arg:
                arg = arg.replace("{route_id}", route_id or "")
            expanded.append(arg)
        return expanded, listen_port

    def child_env(
        self,
        launch_plan: wiring.HarnessLaunch | None,
        env_wiring: Mapping[str, str] | None,
    ) -> dict[str, str]:
        if launch_plan is None:
            env = dict(os.environ if self._parent_env is None else self._parent_env)
            if env_wiring is not None:
                env.update(env_wiring)
            return env
        return wiring.merged_env(
            os.environ if self._parent_env is None else self._parent_env, launch_plan, env_wiring
        )


def ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def harness_kind(harness: str) -> HarnessKind | None:
    try:
        return HarnessKind(harness)
    except ValueError:
        return None
