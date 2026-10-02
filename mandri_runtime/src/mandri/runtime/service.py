"""Runtime service: spawns harness processes and tracks daemon-started sessions."""

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import secrets
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.codex_versions import CODEX_FORK_VERSIONS
from mandri.core.hub import Hub, Topic
from mandri.core.ids import (
    HARNESS_WIRE_FORMATS,
    ApprovalDecision,
    ApprovalId,
    HarnessKind,
    HarnessSessionId,
    ModeApplication,
    ProjectPath,
    RawEvent,
    RouteId,
    SessionId,
    SessionState,
    SessionStopCause,
)
from mandri.core.ports.control import HarnessControl, PromptOutcome, PromptState
from mandri.core.ports.executions import ExecutionRepositoryPort
from mandri.core.ports.privacy import PrivacyScopePort
from mandri.core.types.approvals import ApprovalRequest
from mandri.core.types.availability import SessionAvailability, SessionOwner
from mandri.core.types.config import SessionModeConfig
from mandri.core.types.execution import (
    ExecutionBackend,
    ExecutionPhase,
    PrivacyMode,
    ProtectionError,
    SessionPolicy,
)
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.prompt import UserPrompt
from mandri.core.types.sessions import Session
from mandri.core.types.worktree_integration import IntegrationPreview, IntegrationStrategy
from mandri.core.types.worktrees import Worktree
from mandri.gateway.errors.upstream import RouteNotFoundError
from mandri.gateway.model_metadata import ModelMetadata
from mandri.gateway.model_metadata import fetch as fetch_model_metadata
from mandri.gateway.route_registry import RouteRegistry
from mandri.providers.service import parse_model_arg
from mandri.runtime import native_id
from mandri.runtime.adapters import AdapterContext, HarnessAdapterFactory
from mandri.runtime.agy_command_process import spawn_container_command
from mandri.runtime.agy_launch import prepare_agy_launch
from mandri.runtime.approvals.recognition import with_approval_metadata
from mandri.runtime.approvals.registry import ApprovalRegistry
from mandri.runtime.approvals.service import ApprovalService
from mandri.runtime.approvals.watcher import ApprovalWatcher
from mandri.runtime.attachments import AttachmentError, AttachmentStore
from mandri.runtime.commands import CommandService
from mandri.runtime.control import modes
from mandri.runtime.control.agy_commands import AgyCommandRunner
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.errors import (
    ControlError,
    ControlTransportError,
    HarnessNotInitializedError,
    PromptDeliveryFailedError,
    SteerUnsupportedError,
)
from mandri.runtime.docker_agy import prepare_docker_agy
from mandri.runtime.docker_backend import DockerBackend, DockerReadiness
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.docker_image import DockerImageOptions, image_options
from mandri.runtime.docker_ingress import WorkerIngress
from mandri.runtime.docker_workspace import translate_value
from mandri.runtime.errors import (
    HarnessNotInstalledError,
    SessionNotResumableError,
    SessionNotRunningError,
)
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.executable import require_spawn_executable
from mandri.runtime.execution_lifecycle import ExecutionLifecycle
from mandri.runtime.forking import CodexForkSourcePort
from mandri.runtime.launch_preparation import LaunchPreparation, PreparedLaunch, harness_kind
from mandri.runtime.liveness import (
    ClaudeLivenessAdapter,
    CodexLivenessAdapter,
    LivenessEvidence,
    LivenessEvidenceKind,
    LivenessPort,
    OpencodeLivenessAdapter,
    UnknownSessionError,
)
from mandri.runtime.model_selection import ModelSelectionService
from mandri.runtime.native_catalog import NativeModel
from mandri.runtime.native_readiness import require_native_identity
from mandri.runtime.native_restore import NATIVE_RESTORE_MODEL, restore_codex_model
from mandri.runtime.pi_session_paths import record_session_path
from mandri.runtime.process import ManagedProcess, spawn
from mandri.runtime.registry import LIVE, STOPPED, SessionRegistry
from mandri.runtime.runtime_events import RuntimeEvents
from mandri.runtime.session_access import SessionAccess
from mandri.runtime.session_feed import SessionFeed, session_topic
from mandri.runtime.session_lifetime import SessionLifetime
from mandri.runtime.session_state import RuntimeStates, SessionRuntimeState
from mandri.runtime.start_operations import StartOperations
from mandri.runtime.translators.base import EventPublisher
from mandri.runtime.usage import observe_usage
from mandri.runtime.usage_accounts import NativeAccountReader
from mandri.runtime.usage_profiles import usage_profile_id
from mandri.sessions.errors import (
    SessionConflictError,
    SessionNotFoundError,
    SessionRunningError,
)
from mandri.sessions.service import SessionsService
from mandri.sessions.usage.types import NativeUsageContext, NativeUsageObserver

LivenessAdapter = ClaudeLivenessAdapter | CodexLivenessAdapter | OpencodeLivenessAdapter


_logger = logging.getLogger(__name__)
_DEFAULT_POLICY = SessionPolicy()


@dataclasses.dataclass(frozen=True)
class RuntimeSession:
    id: str
    harness: str
    process: ManagedProcess
    route_id: str | None
    native_id: str | None = None
    listen_port: int | None = None
    mode: str | None = None
    project_path: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    privacy_scope_id: str | None = None
    policy_revision: int = 1
    worktree: Worktree | None = None


class RuntimeService:
    def __init__(
        self,
        harness_commands: Mapping[str, list[str]],
        sessions: SessionsService | None = None,
        routes: RouteRegistry | None = None,
        hub: Hub | None = None,
        registry: SessionRegistry | None = None,
        gateway_port: int | None = None,
        token_issuer: Callable[[str], str] | None = None,
        adapters: HarnessAdapterFactory | None = None,
        approval_timeout_seconds: int = 120,
        mode_defaults: SessionModeConfig | None = None,
        liveness: LivenessPort | None = None,
        idle_release_seconds: float = 60.0,
        agy_home: Path | None = None,
        agy_profiles_dir: Path | None = None,
        docker_config: DockerConfig | None = None,
        privacy_scopes: PrivacyScopePort | None = None,
        executions: ExecutionRepositoryPort | None = None,
        creation_policy: SessionPolicy = _DEFAULT_POLICY,
        attachments_dir: Path | None = None,
    ) -> None:
        self.attachments = AttachmentStore(
            attachments_dir or Path.home() / ".mandri" / "attachments"
        )
        self._states = RuntimeStates()
        self.commands = CommandService(
            self.control_for_session,
            lambda sid: (
                self._registry.status(sid) == LIVE and not self._session_state(sid).stopping
            ),
            self.is_busy,
        )
        self._agy_home = agy_home
        self._agy_profiles = agy_profiles_dir or Path.home() / ".mandri" / "agy-profiles"
        self._harness_commands = harness_commands
        self._models = ModelSelectionService(
            harness_commands, self._states, sessions, agy_home, self._agy_profiles
        )
        self._sessions = sessions
        self._routes = routes
        self._hub = hub
        self._registry = registry if registry is not None else SessionRegistry()
        if sessions is not None and hasattr(sessions, "set_pi_processes"):
            sessions.set_pi_processes(self._native_pi_processes)
        self._launch = LaunchPreparation(gateway_port, token_issuer)
        self._docker_launch = LaunchPreparation(gateway_port, token_issuer, parent_env={})
        self._docker = DockerBackend(docker_config) if docker_config else None
        self._privacy_scopes = privacy_scopes
        self._executions = ExecutionLifecycle(executions, hub)
        self.creation_policy = creation_policy
        self._start_operations: StartOperations[RuntimeSession] = StartOperations()
        self._gateway_port = gateway_port
        self._token_issuer = token_issuer
        self._adapters = adapters
        self._approval_timeout_seconds = approval_timeout_seconds
        self._mode_defaults = mode_defaults if mode_defaults is not None else SessionModeConfig()
        self._approvals = ApprovalService(
            ApprovalRegistry(), clock=system_now_ms, on_expiry=self._handle_expiry
        )
        self._liveness = liveness
        self._events = RuntimeEvents(self._states, self._registry, hub, liveness)
        self._access = SessionAccess(
            sessions, self._registry, lambda session_id: self.is_busy(session_id)
        )
        self._lifetime = SessionLifetime(
            self._states,
            self._registry,
            lambda session_id: self.is_busy(session_id),
            lambda session_id, cause: self.stop_session(session_id, cause=cause),
            idle_release_seconds,
        )
        self._expiry_tasks: set[asyncio.Task[None]] = set()
        self._control_tasks: set[asyncio.Task[None]] = set()
        self._usage_observer: NativeUsageObserver | None = None
        self._usage_accounts: dict[str, NativeAccountReader] = {}

    def _native_pi_processes(self) -> dict[int, str]:
        known: dict[int, str] = {}
        for session_id, process in self._registry.processes().items():
            state = self._session_state(session_id)
            if (
                self._registry.harness_of(session_id) == "pi"
                and state.policy.execution_backend is ExecutionBackend.HOST
                and state.native_id is not None
                and process.returncode is None
            ):
                known[process.process.pid] = str(state.native_id)
        return known

    async def _settle_pi_identities(self, excluded: str) -> bool:
        async def capture(
            session_id: str, process: ManagedProcess, control: HarnessControl
        ) -> None:
            with contextlib.suppress(ControlError, TimeoutError):
                async with asyncio.timeout(10):
                    identity = await control.capture_identity()
                if (
                    identity is not None
                    and self._registry.process(session_id) is process
                    and process.returncode is None
                ):
                    self._session_state(session_id).native_id = identity

        pending = []
        for session_id, process in self._registry.processes().items():
            state = self._session_state(session_id)
            if (
                session_id != excluded
                and self._registry.harness_of(session_id) == "pi"
                and state.policy.execution_backend is ExecutionBackend.HOST
                and state.native_id is None
                and state.control is not None
                and process.returncode is None
            ):
                pending.append(capture(session_id, process, state.control))
        if pending:
            await asyncio.gather(*pending)
        return bool(pending)

    def set_usage_observer(self, observer: NativeUsageObserver | None) -> None:
        self._usage_observer = observer

    def _session_state(self, session_id: str) -> SessionRuntimeState:
        return self._states.session(session_id)

    @property
    def registry(self) -> SessionRegistry:
        return self._registry

    def set_event_router(
        self,
        publisher: EventPublisher,
        approval_topic: Callable[[ApprovalRequest], Topic | None] | None = None,
    ) -> None:
        self._events.publisher = publisher
        self._events.approval_topic = approval_topic

    def control_for_session(self, session_id: str) -> HarnessControl | None:
        return self._session_state(session_id).control

    async def session_availability(self, session_id: str) -> SessionAvailability:
        availability = await self._access.availability(session_id)
        state = self._session_state(session_id)
        if state.resuming or state.stopping:
            return dataclasses.replace(
                availability,
                can_resume=False,
                can_release=False,
                can_restore=False,
                reason="session_transition_in_progress",
            )
        if state.native_restore_failed:
            return dataclasses.replace(availability, reason="native_restore_failed")
        return availability

    async def restore_native_model(self, session_id: str) -> SessionAvailability:
        state = self._session_state(session_id)
        if state.resuming or state.stopping:
            raise SessionRunningError("Session is already changing state")
        state.resuming = True
        try:
            availability = await self._access.availability(session_id)
            if not availability.can_restore:
                raise SessionConflictError(
                    "Native restoration requires an inactive, unowned supported session"
                )
            await self._restore_native_model(session_id, update_selection=True)
            return await self._access.availability(session_id)
        finally:
            state.resuming = False

    async def _restore_native_model(
        self, session_id: str, *, update_selection: bool = False
    ) -> None:
        if self._sessions is not None and hasattr(self._sessions, "worktrees"):
            async with self._sessions.worktrees.lease(session_id):
                await self._restore_native_in_workspace(
                    session_id, update_selection=update_selection
                )
        else:
            await self._restore_native_in_workspace(session_id, update_selection=update_selection)

    async def _restore_native_in_workspace(
        self, session_id: str, *, update_selection: bool = False
    ) -> None:
        if self._sessions is None:
            raise SessionNotResumableError("No session store is configured")
        await self._ensure_session_available(session_id)
        record = await self._sessions.get_session(SessionId(session_id))
        if getattr(record, "worktree", None) is not None:
            await self._sessions.worktrees.validate(SessionId(session_id))
        if record.privacy_mode is PrivacyMode.SURROGATE:
            raise ProtectionError(
                "privacy_native_unsupported", "Protected sessions cannot restore native models"
            )
        if record.execution_backend is ExecutionBackend.DOCKER:
            raise DockerExecutionError(
                "docker_native_unsupported", "Docker sessions require a gateway model"
            )
        if record.harness not in (
            HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.AGY, HarnessKind.PI
        ):
            raise SessionConflictError(
                "Native restoration requires Codex, Claude, Antigravity or Pi"
            )
        state = self._session_state(session_id)
        state.native_restore_failed = True
        model = "default"
        if record.harness is HarnessKind.CODEX:
            await restore_codex_model(record, self._harness_commands, self._spawn_harness)
            model = NATIVE_RESTORE_MODEL
        if update_selection:
            await self._sessions.set_session_model(
                record.id, model, ModelSource.NATIVE, expected=record
            )
        state.native_restore_failed = False

    async def _restore_after_release(self, session_id: str) -> None:
        policy = self._session_state(session_id).policy
        if (
            policy.privacy_mode is PrivacyMode.SURROGATE
            or policy.execution_backend is ExecutionBackend.DOCKER
        ):
            return
        selection = self._session_state(session_id).launched_model
        if (
            self._sessions is None
            or self._registry.harness_of(session_id) != "codex"
            or selection is None
            or selection[0] is not ModelSource.GATEWAY
        ):
            return
        try:
            await self._restore_native_model(session_id)
        except Exception:
            self._session_state(session_id).native_restore_failed = True
            _logger.exception(
                "Native model restoration failed for released session: %s", session_id
            )

    async def release_session(self, session_id: str, *, confirmed: bool) -> SessionAvailability:
        state = self._session_state(session_id)
        if state.resuming or state.stopping:
            raise SessionRunningError("Session is already changing state")
        state.resuming = True
        try:
            return await self._access.release(session_id, confirmed, self.stop_session)
        finally:
            state.resuming = False

    async def answer_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        updated_input: str | None = None,
        answers: list[dict[str, Any]] | None = None,
    ) -> ApprovalRequest:
        raw_input = RawEvent(updated_input) if updated_input is not None else None

        async def deliver(request: ApprovalRequest) -> None:
            delivery = self._session_state(str(request.session_id)).delivery
            if delivery is None or not await delivery.deliver(request):
                raise ControlTransportError("The native interaction is no longer available")

        request = await self._approvals.answer(
            ApprovalId(approval_id), decision, raw_input, answers, deliver=deliver
        )
        self._events.publish_resolved(request)
        return request

    async def cancel_approval(self, approval_id: str) -> ApprovalRequest:
        async def deliver(request: ApprovalRequest) -> None:
            delivery = self._session_state(str(request.session_id)).delivery
            if delivery is None or not await delivery.deliver(request):
                raise ControlTransportError("The native interaction is no longer available")

        request = await self._approvals.cancel(ApprovalId(approval_id), deliver=deliver)
        self._events.publish_resolved(request)
        return request

    def replay_pending_approvals(self, session_id: str, topic: Topic | None = None) -> None:
        if self._hub is None:
            return
        for request in self._approvals.pending_for_session(SessionId(session_id)):
            if topic is not None and (
                self._events.approval_topic is None or self._events.approval_topic(request) != topic
            ):
                continue
            envelope = {
                "source": request.harness.value,
                "raw": json.loads(request.native_request),
                "ts": system_now_ms(),
            }
            payload = {**with_approval_metadata(envelope, request), "type": "approval.pending"}
            self._hub.publish(topic or session_topic(session_id), payload)

    async def expire_approvals(self) -> list[ApprovalRequest]:
        return await self._approvals.expire_due()

    def _handle_expiry(self, request: ApprovalRequest) -> None:
        task = asyncio.create_task(self._deliver_expiry(request))
        self._expiry_tasks.add(task)
        task.add_done_callback(self._expiry_tasks.discard)

    async def _deliver_expiry(self, request: ApprovalRequest) -> None:
        delivery = self._session_state(str(request.session_id)).delivery
        if delivery is not None:
            with contextlib.suppress(ControlError):
                await delivery.deliver(request)
        self._events.publish_resolved(request)

    def installed_harnesses(self) -> list[str]:
        return sorted(self._harness_commands)

    def host_harnesses(self) -> list[str]:
        available = []
        for harness, argv in self._harness_commands.items():
            try:
                require_spawn_executable(argv[0])
            except Exception:
                continue
            available.append(harness)
        return sorted(available)

    def kill_all_now(self) -> None:
        for session_id in self._registry.live_ids():
            process = self._registry.process(session_id)
            if process is not None:
                with contextlib.suppress(Exception):
                    process.kill_now()

    def is_busy(self, session_id: str) -> bool:
        if self.commands.active(session_id):
            return True
        port = self._liveness
        if port is None:
            return True
        try:
            return port.working_state(SessionId(session_id)).busy
        except UnknownSessionError:
            return True

    def viewer_joined(self, session_id: str) -> None:
        self._lifetime.viewer_joined(session_id)

    def viewers_zero(self, session_id: str) -> None:
        self._lifetime.viewers_zero(session_id)

    async def start_session(
        self,
        harness: str,
        model: str,
        cwd: str | Path,
        env_wiring: Mapping[str, str] | None = None,
        mode: str | None = None,
        effort: str | None = None,
        model_source: ModelSource = ModelSource.GATEWAY,
        execution_backend: ExecutionBackend | None = None,
        privacy_mode: PrivacyMode | None = None,
        operation_id: str | None = None,
        _fork_source: CodexForkSourcePort | None = None,
        worktree: bool = False,
        worktree_id: str | None = None,
    ) -> RuntimeSession:
        execution_backend = ExecutionBackend(
            self.creation_policy.execution_backend
            if execution_backend is None
            else execution_backend
        )
        privacy_mode = PrivacyMode(
            self.creation_policy.privacy_mode if privacy_mode is None else privacy_mode
        )
        if worktree_id is not None:
            worktree = True
        if worktree and execution_backend is ExecutionBackend.DOCKER:
            raise ProtectionError(
                "worktree_docker_incompatible", "Worktrees cannot be combined with Docker"
            )
        if worktree and self._sessions is None:
            raise ProtectionError("worktree_unavailable", "Worktrees require session storage")
        if (
            operation_id is None
            and (execution_backend is ExecutionBackend.DOCKER or worktree)
            and self._start_operations.current_id is None
        ):
            operation_id = str(uuid.uuid4())
        if operation_id is not None:
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "harness": harness,
                        "model": model,
                        "cwd": str(cwd),
                        "mode": mode,
                        "effort": effort,
                        "source": model_source,
                        "execution": execution_backend,
                        "privacy": privacy_mode,
                        "worktree": worktree,
                        "worktree_id": worktree_id,
                        "env": dict(env_wiring or {}),
                        "fork_source_id": str(_fork_source.session.id) if _fork_source else None,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            return await self._start_operations.run(
                operation_id,
                fingerprint,
                lambda: self.start_session(
                    harness,
                    model,
                    cwd,
                    env_wiring,
                    mode,
                    effort,
                    model_source,
                    execution_backend,
                    privacy_mode,
                    _fork_source=_fork_source,
                    worktree=worktree,
                    worktree_id=worktree_id,
                ),
            )
        execution_backend = ExecutionBackend(execution_backend)
        privacy_mode = PrivacyMode(privacy_mode)
        policy = SessionPolicy(execution_backend, privacy_mode)
        policy.validate(model_source)
        if (
            execution_backend is ExecutionBackend.DOCKER
            and model_source is ModelSource.GATEWAY
            and _fork_source is None
            and self._docker is not None
        ):
            await self._docker.prepare_image()
        await self._validate_policy(policy, model_source, harness)
        command = self._harness_commands.get(harness)
        if command is None:
            raise HarnessNotInstalledError(f"no command configured for harness {harness!r}")
        launch_mode = modes.resolve_launch(harness, mode, self._mode_defaults)
        self.validate_model_selection(harness, model, model_source)
        native = model_source is ModelSource.NATIVE
        async with contextlib.AsyncExitStack() as cleanup:
            scope_id = await self._create_privacy_scope(policy, cwd, _fork_source)
            route_id = (
                None if native else await self._bind_route(harness, model, effort, policy, scope_id)
            )
            if route_id is not None and self._routes is not None:
                cleanup.push_async_callback(self._rollback_route, route_id)
            metadata = None if native else await self._resolve_metadata(model)
            session_id = await self._create_session_record(
                harness, model, route_id, cwd, effort, model_source, policy, scope_id,
                starting=worktree,
            )
            self._session_state(session_id).policy = policy
            if worktree:
                cleanup.push_async_callback(self._rollback_worktree, session_id)
            cleanup.push_async_callback(self._rollback_session, session_id)
            workspace = None
            if worktree and self._sessions is not None:
                workspace = await self._sessions.worktrees.prepare(
                    SessionId(session_id), str(cwd), worktree_id
                )
                cwd = str(Path(workspace.path) / workspace.relative_path)
                if scope_id is not None and self._privacy_scopes is not None:
                    await self._privacy_scopes.add_workspace(scope_id, cwd)
            fork_path = await self._stage_native_fork(_fork_source, session_id, cwd, policy)
            preparer = (
                self._docker_launch
                if execution_backend is ExecutionBackend.DOCKER
                else self._launch
            )
            prepared = preparer.prepare(
                command,
                harness_kind(harness),
                model,
                effort,
                native,
                route_id,
                metadata,
                launch_mode,
                env_wiring,
                privacy_mode=policy.privacy_mode,
            )
            if (
                harness_kind(harness) is HarnessKind.AGY
                and execution_backend is ExecutionBackend.HOST
            ):
                prepared = self._prepare_agy(
                    prepared, session_id, Path(cwd), native, model, launch_mode
                )
            listen_port = prepared.listen_port
            process = await self._spawn_execution(
                session_id, harness, prepared, cwd, policy, route_id, model, launch_mode
            )
            cleanup.push_async_callback(self._rollback_process, process)
            if self._sessions is not None and harness_kind(harness) is not None:
                await self._sessions.set_session_state(SessionId(session_id), SessionState.LIVE)
            self._session_state(session_id).launched_model = (model_source, model, effort)
            self._registry.mark_live(session_id, process, harness)
            self._attach_feed(
                session_id,
                harness,
                process,
                launch_mode,
                project_path=str(cwd),
                inherited_history="unknown" if _fork_source else "none",
                profile_id=usage_profile_id(
                    harness,
                    prepared.env,
                    isolated_scope=session_id
                    if execution_backend is ExecutionBackend.DOCKER
                    else None,
                    agy_root=self._agy_home,
                ),
            )
            self._attach_control(
                session_id,
                harness,
                process,
                launch_mode,
                model=model if native or _fork_source is not None else None,
                model_source=model_source,
                reasoning_effort=effort,
                prime=_fork_source is None
                and harness_kind(harness) not in (HarnessKind.CODEX, HarnessKind.PI),
                fork_thread_id=HarnessSessionId(_fork_source.native_id) if _fork_source else None,
                fork_path=fork_path,
                workspace_root="/workspace"
                if execution_backend is ExecutionBackend.DOCKER
                else str(cwd),
            )
            self._attach_liveness(session_id, harness)
            self._persist_launch_mode(session_id, launch_mode)
            native_id_value: HarnessSessionId | None = None
            if _fork_source is not None:
                native_id_value = await require_native_identity(
                    self._session_state(session_id).control
                )
                if str(native_id_value) == _fork_source.native_id:
                    raise SessionNotResumableError("Native fork did not create a new conversation")
                await self._reveal_identity(session_id, native_id_value)
            elif harness_kind(harness) in (HarnessKind.AGY, HarnessKind.PI):
                native_id_value = await require_native_identity(self._require_control(session_id))
                await self._reveal_identity(session_id, native_id_value)
            elif harness_kind(harness) is HarnessKind.CODEX and (
                self._adapters is not None
                or self._session_state(session_id).control is not None
                or policy != _DEFAULT_POLICY
            ):
                native_id_value = await require_native_identity(
                    self._session_state(session_id).control
                )
                await self._reveal_identity(session_id, native_id_value)
            if listen_port is not None and harness_kind(harness) is HarnessKind.OPENCODE:
                self._session_state(session_id).resuming = True
                try:
                    native_id_value = await self._attach_opencode(session_id, listen_port)
                finally:
                    self._session_state(session_id).resuming = False
            kind = harness_kind(harness)
            if kind is not None:
                self._events.publish_started(session_id, kind)
            self._lifetime.arm_zero_viewer_decision(session_id)
            await self._executions.phase(session_id, ExecutionPhase.READY)
            cleanup.pop_all()
            return RuntimeSession(
                id=session_id,
                harness=harness,
                process=process,
                route_id=route_id,
                native_id=str(native_id_value) if native_id_value is not None else None,
                listen_port=listen_port,
                mode=launch_mode.mode if launch_mode else None,
                project_path=str(cwd),
                execution_backend=execution_backend,
                privacy_mode=privacy_mode,
                privacy_scope_id=scope_id,
                worktree=workspace,
            )

    @contextlib.asynccontextmanager
    async def _worktree_operation(self, session_id: str) -> AsyncIterator[SessionsService]:
        state = self._session_state(session_id)
        if state.resuming or state.stopping or self._registry.status(session_id) == LIVE:
            raise SessionRunningError("Stop the session before changing its worktree")
        if self._sessions is None:
            raise ProtectionError("worktree_unavailable", "Session storage is unavailable")
        state.stopping = True
        try:
            async with self._sessions.worktrees.lease(session_id):
                record = await self._sessions.get_session(SessionId(session_id))
                if record.state is SessionState.LIVE and record.native_id is None:
                    raise SessionRunningError("Stop the session before changing its worktree")
                if record.native_id is not None:
                    owner = await self._sessions.native_ownership(SessionId(session_id))
                    if owner.owner is SessionOwner.UNKNOWN:
                        raise ProtectionError(
                            "session_ownership_unknown", "Cannot verify native session ownership"
                        )
                    if owner.owner is not SessionOwner.UNOWNED:
                        raise SessionRunningError(
                            "Stop the native writer before changing its worktree"
                        )
                if record.state is SessionState.LIVE:
                    await self._sessions.set_session_state(
                        SessionId(session_id), SessionState.STOPPED
                    )
                yield self._sessions
        finally:
            state.stopping = False

    async def rename_worktree(self, session_id: str, name: str) -> None:
        async with self._worktree_operation(session_id) as sessions:
            await sessions.worktrees.rename(SessionId(session_id), name)

    async def preview_worktree(
        self, session_id: str, target: str | None, strategy: IntegrationStrategy,
    ) -> IntegrationPreview:
        async with self._worktree_operation(session_id) as sessions:
            return await sessions.worktrees.preview(SessionId(session_id), target, strategy)

    async def integrate_worktree(
        self, session_id: str, target: str, strategy: IntegrationStrategy,
        token: str, message: str,
    ) -> None:
        async with self._worktree_operation(session_id) as sessions:
            await sessions.worktrees.integrate(
                SessionId(session_id), target, strategy, token, message
            )

    async def resolve_worktree(
        self, session_id: str, target: str, strategy: IntegrationStrategy, token: str,
    ) -> None:
        async with self._worktree_operation(session_id) as sessions:
            await sessions.worktrees.resolve(SessionId(session_id), target, strategy, token)

    async def finish_worktree(self, session_id: str, *, discard_ignored: bool = False) -> None:
        async with self._worktree_operation(session_id) as sessions:
            await sessions.worktrees.finish(SessionId(session_id), discard_ignored=discard_ignored)

    async def _rollback_worktree(self, session_id: str) -> None:
        if self._sessions is not None:
            await self._sessions.delete_session(SessionId(session_id))

    async def fork_session(
        self,
        source_id: str,
        execution_backend: ExecutionBackend,
        privacy_mode: PrivacyMode,
        *,
        mode: str | None = None,
        operation_id: str | None = None,
        worktree: bool = False,
        worktree_id: str | None = None,
    ) -> RuntimeSession:
        execution_backend = ExecutionBackend(execution_backend)
        privacy_mode = PrivacyMode(privacy_mode)
        if operation_id is not None:
            fingerprint = json.dumps(
                ["fork", source_id, execution_backend, privacy_mode, mode, worktree, worktree_id],
                separators=(",", ":"),
            )
            return await self._start_operations.run(
                operation_id,
                fingerprint,
                lambda: self.fork_session(
                    source_id, execution_backend, privacy_mode, mode=mode,
                    worktree=worktree, worktree_id=worktree_id,
                ),
            )
        if self._sessions is None:
            raise ProtectionError("session_transition_unsupported", "Native history is unavailable")
        await self._sessions.ensure_no_pending_purge(SessionId(source_id))
        if self._registry.status(source_id) == LIVE:
            raise SessionRunningError("Stop the source session before creating a fork")
        if execution_backend is ExecutionBackend.DOCKER:
            version = await self.docker_harness_version("codex")
            if version not in CODEX_FORK_VERSIONS:
                raise ProtectionError(
                    "session_transition_unsupported", "The native fork adapter is not qualified"
                )
        result: RuntimeSession | None = None
        try:
            async with self._sessions.codex_fork_source(
                SessionId(source_id), execution_backend
            ) as source:
                record = source.session
                if record.model_source is not ModelSource.GATEWAY:
                    raise ProtectionError(
                        "session_transition_unsupported",
                        "Native-auth history transfer is unqualified",
                    )
                source_mode = (
                    record.interaction_mode.mode if record.interaction_mode is not None else None
                )
                source_worktree = getattr(record, "worktree", None)
                target_cwd = (
                    source_worktree.source_path if source_worktree else str(source.workspace_root)
                )
                result = await self.start_session(
                    record.harness.value,
                    record.model or "default",
                    target_cwd,
                    mode=mode if mode is not None else source_mode,
                    effort=record.reasoning_effort,
                    execution_backend=execution_backend,
                    privacy_mode=privacy_mode,
                    _fork_source=source,
                    worktree=worktree,
                    worktree_id=worktree_id,
                )
            return result
        except BaseException:
            if result is not None:
                try:
                    await self._rollback_process(result.process)
                finally:
                    await self._rollback_session(result.id)
                    if getattr(result, "worktree", None) is not None:
                        await self._rollback_worktree(result.id)
            raise

    async def _stage_native_fork(
        self,
        source: CodexForkSourcePort | None,
        session_id: str,
        cwd: str | Path,
        policy: SessionPolicy,
    ) -> str | None:
        if source is None:
            return None
        if policy.execution_backend is ExecutionBackend.HOST:
            return str(source.rollout_path)
        if self._docker is None:
            raise DockerExecutionError("docker_unavailable", "Docker is not configured")
        context = self._docker.context(session_id, cwd, resume=False)
        selected = Path(*source.rollout_path.parts[-4:])
        if len(selected.parts) != 4 or any(not part.isdigit() for part in selected.parts[:3]):
            raise ProtectionError(
                "session_transition_unsupported", "Native rollout layout is unsupported"
            )
        relative = Path(".codex/sessions") / selected
        destination = Path(context["native_state_root"]) / relative
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        task = asyncio.create_task(asyncio.to_thread(source.copy_rollout, destination))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        return str(Path("/home/worker") / relative)

    async def resume_session(self, session_id: str, *, mode: str | None = None) -> RuntimeSession:
        if self._session_state(session_id).resuming or self._session_state(session_id).stopping:
            raise SessionRunningError(f"session {session_id} is already changing state")
        self._session_state(session_id).resuming = True
        try:
            return await self._resume_session(session_id, mode=mode)
        finally:
            self._session_state(session_id).resuming = False

    async def cancel_start(self, operation_id: str) -> None:
        async def stop(result: RuntimeSession) -> None:
            if self._registry.status(result.id) == LIVE:
                await self.stop_session(result.id, restore_native=False)
            if getattr(result, "worktree", None) is not None and self._sessions is not None:
                await self._sessions.delete_session(SessionId(result.id))

        await self._start_operations.cancel(operation_id, stop)

    async def _resume_session(self, session_id: str, *, mode: str | None = None) -> RuntimeSession:
        if self._sessions is not None and hasattr(self._sessions, "worktrees"):
            async with self._sessions.worktrees.lease(session_id):
                return await self._resume_in_workspace(session_id, mode=mode)
        return await self._resume_in_workspace(session_id, mode=mode)

    async def _resume_in_workspace(
        self, session_id: str, *, mode: str | None = None
    ) -> RuntimeSession:
        state = self._session_state(session_id)
        stop_revision = state.stop_revision
        if self._sessions is not None and hasattr(self._sessions, "ensure_no_pending_purge"):
            await self._sessions.ensure_no_pending_purge(SessionId(session_id))
        record = await self._load_resumable_record(session_id)
        if getattr(record, "worktree", None) is not None and self._sessions is not None:
            await self._sessions.worktrees.validate(SessionId(session_id))
        launch_mode = self._resume_launch_mode(record, mode=mode)
        policy = SessionPolicy(record.execution_backend, record.privacy_mode)
        await self._validate_policy(
            policy,
            record.model_source,
            record.harness.value,
        )
        if policy.privacy_mode is PrivacyMode.SURROGATE:
            if self._privacy_scopes is None or record.privacy_scope_id is None:
                raise ProtectionError("privacy_state_unavailable", "Privacy state is unavailable")
            await self._privacy_scopes.validate(record.privacy_scope_id)
        self._session_state(session_id).policy = policy
        self._session_state(session_id).policy_revision = getattr(record, "policy_revision", 1)
        harness = record.harness.value
        command = self._harness_commands.get(harness)
        if command is None:
            raise HarnessNotInstalledError(f"no command configured for harness {harness!r}")
        route_id = await self._resume_route(record, harness)
        model = record.model or "default"
        native = record.model_source is ModelSource.NATIVE
        metadata = None if native else await self._resolve_metadata(model)
        kind = record.harness
        preparer = (
            self._docker_launch
            if policy.execution_backend is ExecutionBackend.DOCKER
            else self._launch
        )
        prepared = preparer.prepare(
            command,
            kind,
            model,
            record.reasoning_effort,
            native,
            route_id,
            metadata,
            launch_mode,
            resume_native_id=record.native_id,
            privacy_mode=policy.privacy_mode,
        )
        if kind is HarnessKind.AGY and policy.execution_backend is ExecutionBackend.HOST:
            prepared = self._prepare_agy(
                prepared, session_id, Path(record.project_path), native, model, launch_mode
            )
        listen_port = prepared.listen_port
        try:
            await self._ensure_session_available(session_id)
            if state.stop_revision != stop_revision:
                raise SessionNotRunningError(f"session {session_id} was stopped")
            process = await self._spawn_execution(
                session_id,
                harness,
                prepared,
                record.project_path,
                policy,
                route_id,
                model,
                launch_mode,
                resume_native_id=record.native_id,
            )
        except BaseException:
            await self._close_control(session_id)
            raise
        self._registry.mark_live(session_id, process, harness)
        if state.stop_revision != stop_revision:
            await self.stop_session(session_id, force=True, restore_native=False)
            raise SessionNotRunningError(f"session {session_id} was stopped")
        try:
            return await self._finalize_resume(
                record, kind, harness, process, launch_mode, route_id, listen_port
            )
        except BaseException:
            await self._abandon_resume(session_id, process)
            raise

    async def _load_resumable_record(self, session_id: str) -> Session:
        if self._sessions is None:
            raise SessionNotResumableError("no session store is configured")
        ensure_policy = getattr(self._sessions, "ensure_session_policy", None)
        record = (
            await ensure_policy(SessionId(session_id))
            if ensure_policy is not None
            else await self._sessions.get_session(SessionId(session_id))
        )
        if self._registry.status(session_id) == LIVE:
            raise SessionRunningError(f"session {session_id} is already running")
        worktree = getattr(record, "worktree", None)
        if worktree is not None and worktree.state == "closed":
            raise ProtectionError("worktree_closed", "This worktree was integrated and cleaned")
        if record.native_id is None:
            raise SessionNotResumableError(
                f"session {session_id} has no harness-native conversation to resume"
            )
        if (
            record.gateway_route_id is None
            and record.model is None
            and record.model_source is not ModelSource.NATIVE
        ):
            raise SessionNotResumableError(f"session {session_id} was not started by the daemon")
        await self._ensure_session_available(session_id)
        return record

    def _prepare_agy(
        self,
        prepared: PreparedLaunch,
        session_id: str,
        cwd: Path,
        native: bool,
        model: str,
        mode: modes.LaunchMode | None,
    ) -> PreparedLaunch:
        if self._gateway_port is None:
            raise SessionNotResumableError("Antigravity requires a daemon hook endpoint")

        async def publish(raw: dict[str, Any]) -> None:
            await self._events.publish_event(
                session_topic(session_id), {"source": "agy", "raw": raw, "ts": system_now_ms()}
            )

        result, resource = prepare_agy_launch(
            prepared,
            session_id,
            cwd,
            self._agy_profiles,
            self._agy_home,
            native,
            model,
            mode.mode if mode and mode.mode else "default",
            self._gateway_port,
            self._approval_timeout_seconds,
            publish,
        )
        self._session_state(session_id).agy = resource
        return result

    async def agy_hook(
        self, session_id: str, token: str, event: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        resource = self._session_state(session_id).agy
        if resource is None or not resource.bridge.authenticates(token):
            raise PermissionError("Invalid Antigravity hook token")
        return await resource.bridge.handle(event, data)

    async def _ensure_session_available(self, session_id: str) -> None:
        if self._sessions is None:
            return
        owner = await self._sessions.native_ownership(SessionId(session_id))
        if owner.owner is SessionOwner.UNKNOWN and await self._settle_pi_identities(session_id):
            owner = await self._sessions.native_ownership(SessionId(session_id))
        if owner.owner is not SessionOwner.UNOWNED:
            raise SessionRunningError("Session has an external or unconfirmed native writer")

    def _resume_launch_mode(
        self, record: Session, *, mode: str | None = None
    ) -> modes.LaunchMode | None:
        interaction = record.interaction_mode
        if mode is None:
            mode = interaction.mode if interaction is not None else None
        elif record.harness is HarnessKind.OPENCODE and mode not in {"default", "auto"}:
            modes.opencode_permission_rules(mode)
        return modes.resolve_launch(record.harness.value, mode, self._mode_defaults)

    async def _resume_route(
        self, record: Session, harness: str, *, rotate: bool = True
    ) -> str | None:
        if record.model_source is ModelSource.NATIVE:
            return None
        stored = record.gateway_route_id
        if stored is not None:
            if self._routes is None:
                return str(stored)
            try:
                existing = await self._routes.get(stored)
                if (
                    existing.execution_backend is not record.execution_backend
                    or existing.privacy_mode is not record.privacy_mode
                    or existing.privacy_scope_id != record.privacy_scope_id
                ):
                    raise ProtectionError(
                        "privacy_state_unavailable",
                        "Stored route policy does not match the session",
                    )
                if record.execution_backend is ExecutionBackend.HOST or not rotate:
                    if record.model:
                        provider_name, model_id = parse_model_arg(record.model)
                        await self._routes.swap(stored, provider_name, model_id)
                        await self._routes.set_reasoning_effort(stored, record.reasoning_effort)
                    return str(stored)
            except RouteNotFoundError:
                pass
        if not rotate and record.execution_backend is ExecutionBackend.DOCKER:
            raise ProtectionError("privacy_route_unbound", "The active worker route is unavailable")
        route_id = await self._bind_route(
            harness,
            record.model or "",
            record.reasoning_effort,
            SessionPolicy(record.execution_backend, record.privacy_mode),
            record.privacy_scope_id,
        )
        if self._sessions is not None:
            with contextlib.suppress(SessionNotFoundError):
                await self._sessions.set_session_route_id(
                    SessionId(str(record.id)), RouteId(route_id)
                )
        if (
            stored is not None
            and self._routes is not None
            and record.execution_backend is ExecutionBackend.DOCKER
        ):
            with contextlib.suppress(RouteNotFoundError):
                await self._routes.delete(stored)
        return route_id

    async def _finalize_resume(
        self,
        record: Session,
        kind: HarnessKind,
        harness: str,
        process: ManagedProcess,
        launch_mode: modes.LaunchMode | None,
        route_id: str | None,
        listen_port: int | None,
    ) -> RuntimeSession:
        session_id = str(record.id)
        self._session_state(session_id).launched_model = (
            record.model_source,
            record.model or "default",
            record.reasoning_effort,
        )
        native_id_value = record.native_id
        self._attach_feed(
            session_id,
            harness,
            process,
            launch_mode,
            project_path=str(record.project_path),
            inherited_history="unknown" if record.parent_session_id else "none",
            parent_session_id=str(record.parent_session_id) if record.parent_session_id else None,
            resumed=True,
        )
        revealed: HarnessSessionId | None = None
        if kind is HarnessKind.OPENCODE:
            if listen_port is None or native_id_value is None:
                raise SessionNotResumableError(f"session {session_id} cannot reattach opencode")
            revealed = await self._reattach_opencode(session_id, listen_port, native_id_value)
            if launch_mode is not None and launch_mode.mode in modes.OPENCODE_PERMISSION_RULES:
                await self._require_control(session_id).set_mode(launch_mode.mode)
        else:
            resume_thread_id = (
                native_id_value
                if kind in (HarnessKind.CODEX, HarnessKind.AGY, HarnessKind.PI)
                else None
            )
            self._attach_control(
                session_id,
                harness,
                process,
                launch_mode,
                resume_thread_id=resume_thread_id,
                prime=kind is HarnessKind.CLAUDE,
                model=record.model,
                model_source=record.model_source,
                reasoning_effort=record.reasoning_effort,
                resume_native_id=native_id_value
                if kind in (HarnessKind.CLAUDE, HarnessKind.PI)
                or record.execution_backend is ExecutionBackend.DOCKER
                else None,
            )
            control = self._session_state(session_id).control
            if control is not None and kind is not HarnessKind.CLAUDE:
                revealed = (
                    await require_native_identity(control)
                    if kind is HarnessKind.CODEX
                    else await control.capture_identity()
                )
                if (
                    record.execution_backend is ExecutionBackend.DOCKER
                    and revealed != native_id_value
                ):
                    raise SessionNotResumableError("Docker resumed a different native conversation")
                if kind is HarnessKind.PI and launch_mode is not None and launch_mode.mode:
                    applied = await control.set_mode(launch_mode.mode)
                    if applied is ModeApplication.REQUIRES_RESTART:
                        raise ControlTransportError(
                            "The restarted Pi harness is missing the permission extension"
                        )
                await self._reveal_identity(session_id, revealed)
        self._attach_liveness(session_id, harness)
        if launch_mode is not None and launch_mode.mode is not None:
            await self._persist_interaction_mode(
                session_id, launch_mode.mode, ModeApplication.AT_LAUNCH
            )
        await self._mark_session_live(session_id)
        if revealed is not None and revealed != native_id_value and self._sessions is not None:
            with contextlib.suppress(SessionNotFoundError, SessionConflictError):
                await self._sessions.rotate_native_id(SessionId(session_id), revealed)
        self._events.publish_started(session_id, kind)
        self._lifetime.arm_zero_viewer_decision(session_id)
        await self._executions.phase(session_id, ExecutionPhase.READY)
        return RuntimeSession(
            id=session_id,
            harness=harness,
            process=process,
            route_id=route_id,
            native_id=str(revealed) if revealed is not None else str(native_id_value),
            listen_port=listen_port,
            mode=launch_mode.mode if launch_mode else None,
            project_path=str(record.project_path),
            execution_backend=record.execution_backend,
            privacy_mode=record.privacy_mode,
            privacy_scope_id=record.privacy_scope_id,
            worktree=getattr(record, "worktree", None),
            policy_revision=getattr(record, "policy_revision", 1),
        )

    async def _reattach_opencode(
        self, session_id: str, listen_port: int, native_id_value: HarnessSessionId
    ) -> HarnessSessionId:
        process = self._registry.process(session_id)
        verified = await native_id.verify_opencode_session_id(
            listen_port,
            str(native_id_value),
            alive=(lambda: process.returncode is None) if process is not None else None,
            **(
                {"auth": self._session_state(session_id).control_auth}
                if self._session_state(session_id).control_auth
                else {}
            ),
        )
        if self._adapters is None:
            return verified
        adapters = self._adapters(
            AdapterContext(
                kind=HarnessKind.OPENCODE,
                listen_port=listen_port,
                native_session_id=str(verified),
                control_auth=self._session_state(session_id).control_auth,
            )
        )
        if adapters is None:
            return verified
        self._session_state(session_id).control = adapters.control
        if adapters.delivery is not None:
            self._session_state(session_id).delivery = adapters.delivery
        self._events.start_event_pump(session_id, adapters.events)
        return verified

    async def _mark_session_live(self, session_id: str) -> None:
        if self._sessions is None:
            return
        await self._sessions.set_session_state(SessionId(session_id), SessionState.DISCOVERED)
        await self._sessions.set_session_state(SessionId(session_id), SessionState.LIVE)

    async def _abandon_resume(self, session_id: str, process: ManagedProcess) -> None:
        try:
            await self._rollback_process(process)
        finally:
            await self._rollback_session(session_id)

    async def _rollback_session(self, session_id: str) -> None:
        try:
            await self._executions.phase(
                session_id, ExecutionPhase.FAILED, reason="execution_launch_failed"
            )
        except Exception:
            _logger.exception("Failed to persist abandoned execution state")
        state = self._session_state(session_id)
        state.stopping = True
        self._lifetime.cancel_lifetime_task(session_id)
        await self._cancel_session_tasks(session_id)
        operations = (
            self._close_approvals,
            self._close_control,
            self._events.forget_liveness,
            self._events.detach_feed,
        )
        for operation in operations:
            try:
                await operation(session_id)
            except Exception:
                _logger.exception("Failed to release session resource: %s", session_id)
        self._registry.mark_stopped(session_id)
        try:
            await self._mark_db_state(session_id, SessionState.STOPPED)
        except Exception:
            _logger.exception("Failed to persist abandoned session: %s", session_id)
        state.launched_model = None
        state.idle_since = None
        state.stopping = False

    async def _rollback_process(self, process: ManagedProcess) -> None:
        try:
            await process.stop(2.0)
        except Exception:
            _logger.exception("Failed to stop abandoned harness")
            with contextlib.suppress(Exception):
                process.kill_now()
                await asyncio.wait_for(process.wait(), 2.0)

    async def _rollback_route(self, route_id: str) -> None:
        if self._routes is not None:
            try:
                await self._routes.delete(RouteId(route_id))
            except RouteNotFoundError:
                pass
            except Exception:
                _logger.exception("Failed to remove abandoned route: %s", route_id)

    async def _cancel_session_tasks(self, session_id: str) -> None:
        state = self._session_state(session_id)
        tasks: set[asyncio.Task[object]] = set(state.control_tasks)
        if state.identity_task is not None:
            tasks.add(state.identity_task)
            state.identity_task = None
        current = asyncio.current_task()
        tasks = {task for task in tasks if task is not current and not task.done()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _track_control_task(self, session_id: str, task: asyncio.Task[None]) -> None:
        state = self._session_state(session_id)
        state.control_tasks.add(task)
        self._control_tasks.add(task)
        task.add_done_callback(state.control_tasks.discard)
        task.add_done_callback(self._control_tasks.discard)

    async def set_session_mode(self, session_id: str, mode: str) -> ModeApplication:
        async with self._session_state(session_id).mode_lock:
            state = self._session_state(session_id)
            if state.resuming or state.stopping:
                raise ControlTransportError("Session is already changing state")
            control = self._require_control(session_id)
            await self._capture_identity(session_id, control)
            application = await control.set_mode(mode)
            if application is ModeApplication.REQUIRES_RESTART:
                await self._restart_for_mode(session_id, mode)
                application = ModeApplication.RESTARTED
            await self._persist_interaction_mode(session_id, mode, application)
            return application

    async def _restart_for_mode(self, session_id: str, mode: str) -> None:
        if self._sessions is None:
            raise ControlTransportError("Session storage is required to restart the harness")
        record = await self._sessions.get_session(SessionId(session_id))
        self._resume_launch_mode(record, mode=mode)
        state = self._session_state(session_id)
        async with state.prompt_lock:
            if state.resuming or state.stopping:
                raise ControlTransportError("Session is already changing state")
            control = self._require_control(session_id)
            identity = await self._capture_identity(session_id, control)
            if identity is None:
                raise ControlTransportError("The harness has no conversation to resume")
            stop_revision = state.stop_revision
            state.resuming = True
            try:
                await control.interrupt()
                if state.stop_revision != stop_revision:
                    raise SessionNotRunningError(f"session {session_id} was stopped")
                await self.stop_session(session_id, restore_native=False)
                if state.stop_revision != stop_revision:
                    raise SessionNotRunningError(f"session {session_id} was stopped")
                await self._resume_session(session_id, mode=mode)
            finally:
                state.resuming = False

    async def set_session_effort(self, session_id: str, effort: str | None) -> Session:
        if self._sessions is None:
            raise SessionNotResumableError("no session store is configured")
        record = await self._sessions.get_session(SessionId(session_id))
        if (
            record.model_source is ModelSource.GATEWAY
            and record.gateway_route_id is not None
            and self._routes is not None
        ):
            with contextlib.suppress(RouteNotFoundError):
                await self._routes.set_reasoning_effort(record.gateway_route_id, effort)
        await self._sessions.set_session_effort(SessionId(session_id), effort)
        return await self._sessions.get_session(SessionId(session_id))

    async def send_session_prompt(
        self, session_id: str, content: str, attachments: list[str] | None = None
    ) -> PromptOutcome:
        if self._session_state(session_id).resuming:
            raise ControlTransportError("Session is already changing state")
        if self.commands.active(session_id):
            raise ControlTransportError("Wait for the native command to finish")
        prepared: str | UserPrompt = content
        if attachments:
            if self._sessions is None:
                raise AttachmentError("Session storage is unavailable")
            session = await self._sessions.get_session(SessionId(session_id))
            prepared = await asyncio.to_thread(
                self.attachments.prepare, session, content, attachments
            )
        if self._registry.harness_of(session_id) != "agy":
            return await self._send_session_prompt(session_id, prepared)
        async with self._session_state(session_id).prompt_lock:
            restarted = False
            if (
                self._registry.harness_of(session_id) == "agy"
                and self._registry.status(session_id) == LIVE
                and self.is_busy(session_id)
            ):
                await self._restart_agy(session_id)
                restarted = True
            outcome = await self._send_session_prompt(session_id, prepared)
            return (
                PromptOutcome(PromptState.STEERED, "stop_resume")
                if restarted and outcome.state is not PromptState.ERROR
                else outcome
            )

    async def _send_session_prompt(
        self, session_id: str, content: str | UserPrompt
    ) -> PromptOutcome:
        if self._registry.status(session_id) != LIVE or self._session_state(session_id).stopping:
            raise SessionNotRunningError(f"no live session {session_id}")
        was_busy = self.is_busy(session_id)
        if not was_busy:
            await self._apply_pending_model(session_id)
        control = self._require_control(session_id)
        prompt_id = str(uuid.uuid4())
        self._observe_prompt(session_id, prompt_id, LivenessEvidenceKind.PROMPT_STARTED)
        try:
            if not was_busy and self._sessions is not None and self._routes is not None:
                record = await self._sessions.get_session(SessionId(session_id))
                await self._resume_route(record, record.harness.value, rotate=False)
        except BaseException:
            self._observe_prompt(session_id, prompt_id, LivenessEvidenceKind.PROMPT_REJECTED)
            raise
        try:
            try:
                outcome = await control.send_prompt(content)
            except HarnessNotInitializedError:
                try:
                    await self._capture_identity(session_id, control)
                except BaseException:
                    self._observe_prompt(
                        session_id, prompt_id, LivenessEvidenceKind.PROMPT_REJECTED
                    )
                    raise
                outcome = await control.send_prompt(content)
        except (PromptDeliveryFailedError, HarnessNotInitializedError):
            self._observe_prompt(session_id, prompt_id, LivenessEvidenceKind.PROMPT_REJECTED)
            raise
        if outcome.state is PromptState.ERROR:
            self._observe_prompt(session_id, prompt_id, LivenessEvidenceKind.PROMPT_REJECTED)
        return outcome

    def _observe_prompt(self, session_id: str, prompt_id: str, kind: LivenessEvidenceKind) -> None:
        if self._liveness is not None:
            with contextlib.suppress(UnknownSessionError):
                self._liveness.observe(LivenessEvidence(SessionId(session_id), kind, prompt_id))

    async def interrupt_session(self, session_id: str) -> bool:
        if self._registry.harness_of(session_id) == "agy":
            async with self._session_state(session_id).prompt_lock:
                await self._restart_agy(session_id)
                return True
        control = self._require_control(session_id)
        await self._capture_identity(session_id, control)
        return await control.interrupt()

    async def _restart_agy(self, session_id: str) -> None:
        state = self._session_state(session_id)
        stop_revision = state.stop_revision
        control = self._require_control(session_id)
        identity = await self._capture_identity(session_id, control)
        if identity is None:
            raise SessionNotResumableError("Antigravity has no conversation to resume")
        state.resuming = True
        try:
            await control.interrupt()
            if state.stop_revision != stop_revision:
                raise SessionNotRunningError(f"session {session_id} was stopped")
            await self.stop_session(session_id, grace=1.0, restore_native=False)
            if state.stop_revision != stop_revision:
                raise SessionNotRunningError(f"session {session_id} was stopped")
            await self._resume_session(session_id)
        finally:
            state.resuming = False

    async def stop_session(
        self,
        session_id: str,
        grace: float = 5.0,
        cause: SessionStopCause = SessionStopCause.VIEWER_STOP,
        *,
        restore_native: bool = True,
        force: bool = False,
    ) -> int:
        state = self._session_state(session_id)
        automatic = cause is SessionStopCause.IDLE_TIMEOUT
        if automatic and (state.stopping or self._lifetime.release_blocked(session_id)):
            return 0
        if force:
            self._session_state(session_id).stop_revision += 1
        self._session_state(session_id).stopping = True
        self._session_state(session_id).idle_since = None
        try:
            self._lifetime.cancel_lifetime_task(session_id)
            process = self._registry.process(session_id)
            if self._registry.status(session_id) == STOPPED or process is None:
                raise SessionNotRunningError(f"no live session {session_id}")
            if force:
                code = await process.kill()
                await self._persist_execution_exit(session_id, ExecutionPhase.STOPPING)
            else:
                await self._persist_execution_exit(session_id, ExecutionPhase.STOPPING)
                if automatic and self._lifetime.release_blocked(session_id):
                    await self._executions.phase(session_id, ExecutionPhase.READY)
                    _logger.info("session %s idle release cancelled by activity", session_id)
                    return 0
                code = await process.stop(grace)
            checkpoint = self._session_state(session_id).pi_checkpoint
            if checkpoint is not None:
                await asyncio.to_thread(checkpoint.recover)
            terminal_phase = (
                ExecutionPhase.FAILED if cause is SessionStopCause.CRASH else ExecutionPhase.STOPPED
            )
            await self._persist_execution_exit(session_id, terminal_phase, process=process)
            await self._cancel_session_tasks(session_id)
            await self._close_approvals(session_id)
            await self._close_control(session_id)
            await self._events.forget_liveness(session_id)
            await self._events.detach_feed(session_id)
            self._registry.mark_stopped(session_id)
            await self._mark_db_state(session_id, SessionState.STOPPED)
            if restore_native:
                await self._restore_after_release(session_id)
            self._events.publish_stopped(session_id, cause)
            return code
        finally:
            self._session_state(session_id).stopping = False

    async def reconcile(self) -> list[str]:
        processes = self._registry.processes()
        stopped = await self._registry.reconcile(self._transitioning_sessions())
        for session_id in stopped:
            await self._events.drain_feed(session_id)
            await self._persist_execution_exit(
                session_id, ExecutionPhase.FAILED, process=processes.get(session_id)
            )
            await self._close_approvals(session_id)
            await self._close_control(session_id)
            await self._events.forget_liveness(session_id)
            await self._events.detach_feed(session_id)
            self._events.publish_stopped(session_id, SessionStopCause.CRASH)
        return stopped

    async def reconcile_and_persist(self) -> list[str]:
        await self._lifetime.release_idle_processes()
        processes = self._registry.processes()
        stopped = await self._registry.reconcile(self._transitioning_sessions())
        for session_id in stopped:
            await self._events.drain_feed(session_id)
            await self._persist_execution_exit(
                session_id, ExecutionPhase.FAILED, process=processes.get(session_id)
            )
            await self._close_approvals(session_id)
            await self._close_control(session_id)
            await self._events.forget_liveness(session_id)
            await self._events.detach_feed(session_id)
            await self._mark_db_state(session_id, SessionState.STOPPED)
            self._events.publish_stopped(session_id, SessionStopCause.CRASH)
        stopped.extend(await self._reconcile_stopped_docker_sessions())
        return stopped

    async def _persist_execution_exit(
        self,
        session_id: str,
        phase: ExecutionPhase,
        *,
        process: ManagedProcess | None = None,
    ) -> None:
        context = (
            {
                "exit_code": process.returncode,
                "oom_killed": getattr(process, "oom_killed", False),
            }
            if process is not None
            else None
        )
        reason = None
        if phase is ExecutionPhase.FAILED:
            reason = "execution_oom" if context and context["oom_killed"] else "execution_exited"
        try:
            await self._executions.phase(session_id, phase, context=context, reason=reason)
        except Exception:
            _logger.exception("Failed to persist execution termination: %s", session_id)

    def _transitioning_sessions(self) -> frozenset[str]:
        return frozenset(
            key for key, state in self._states.items() if state.stopping or state.resuming
        )

    async def _close_approvals(self, session_id: str) -> None:
        watcher = self._session_state(session_id).watcher
        self._session_state(session_id).watcher = None
        try:
            if watcher is not None:
                await watcher.stop()
        finally:
            self._session_state(session_id).delivery = None
            cancelled = await self._approvals.cancel_all_for_session(SessionId(session_id))
            for request in cancelled:
                self._events.publish_resolved(request)

    def _attach_feed(
        self,
        session_id: str,
        harness: str,
        process: ManagedProcess,
        launch_mode: modes.LaunchMode | None = None,
        *,
        project_path: str | None = None,
        inherited_history: str = "unknown",
        parent_session_id: str | None = None,
        resumed: bool = False,
        profile_id: str | None = None,
    ) -> None:
        kind = harness_kind(harness)
        if self._hub is None or kind is None:
            return
        agy = self._session_state(session_id).agy
        isolated = (
            self._session_state(session_id).policy.execution_backend is ExecutionBackend.DOCKER
        )
        selection = self._session_state(session_id).launched_model
        usage_context = NativeUsageContext(
            session_id=session_id,
            harness=kind.value,
            process_epoch=uuid.uuid4().hex,
            project_path=project_path,
            routing="native" if selection and selection[0] is ModelSource.NATIVE else "gateway",
            profile_id=profile_id
            or usage_profile_id(
                kind.value,
                None,
                isolated_scope=session_id if isolated else None,
                agy_root=self._agy_home,
            ),
            parent_session_id=parent_session_id,
            inherited_history="none" if inherited_history == "none" else "unknown",
            resumed=resumed,
            process_started_at=system_now_ms(),
            billing_mode="subscription"
            if kind is HarnessKind.CODEX and selection and selection[0] is ModelSource.NATIVE
            else "unknown",
        )
        self._session_state(session_id).usage_context = usage_context

        async def publish(topic: Topic, payload: dict[str, Any]) -> None:
            raw = payload.get("raw")
            if isinstance(raw, dict):
                identity = self._session_state(session_id).native_id
                current_selection = self._session_state(session_id).launched_model
                context = dataclasses.replace(
                    usage_context,
                    native_id=str(identity) if identity is not None else None,
                    observed_model=str(current_selection[1])
                    if current_selection and current_selection[0] is ModelSource.NATIVE
                    else None,
                )
                await observe_usage(self._usage_observer, context, raw)
            if agy is not None and isinstance(raw, dict):
                agy.record(raw)
            await self._events.publish_event(topic, payload)

        feed = SessionFeed(self._hub, session_id, kind, process.process, publisher=publish)
        self._session_state(session_id).feed_start_seq = self._hub.sequence(
            session_topic(session_id)
        )
        feed.start()
        self._session_state(session_id).feed = feed
        self._attach_approvals(session_id, kind, launch_mode)
        self._watch_feed_eof(session_id, feed)

    async def _approve_once(self, approval_id: str) -> None:
        await self.answer_approval(approval_id, ApprovalDecision.ONCE)

    def _attach_approvals(
        self, session_id: str, kind: HarnessKind, launch_mode: modes.LaunchMode | None = None
    ) -> None:
        hub = self._hub
        if hub is None:
            return
        watcher = ApprovalWatcher(
            hub,
            self._approvals,
            SessionId(session_id),
            kind,
            self._approval_timeout_seconds,
            auto_answer=self._approve_once if launch_mode and launch_mode.auto_approve else None,
            publisher=self._events.publish_event,
        )
        watcher.start()
        self._session_state(session_id).watcher = watcher

    def _attach_liveness(self, session_id: str, harness: str) -> None:
        self._events.attach_liveness(session_id, harness)

    def _watch_feed_eof(self, session_id: str, feed: SessionFeed) -> None:
        stdout_task = feed.tasks[0] if feed.tasks else None
        if stdout_task is None:
            return
        task = asyncio.create_task(
            self._close_control_after_feed_eof(session_id, stdout_task),
            name=f"control-liveness:{session_id}",
        )
        self._track_control_task(session_id, task)

    async def _close_control_after_feed_eof(
        self, session_id: str, stdout_task: asyncio.Task[None]
    ) -> None:
        try:
            await asyncio.shield(stdout_task)
        except asyncio.CancelledError:
            return
        except Exception:
            _logger.warning("session %s stdout feed failed", session_id, exc_info=True)
        state = self._session_state(session_id)
        if state.stopping or state.resuming or self._registry.status(session_id) != LIVE:
            return
        kind = harness_kind(self._registry.harness_of(session_id) or "")
        if kind is not None:
            self._events.publish_control_lost(session_id, kind)
        process = self._registry.process(session_id)
        if process is not None:
            with contextlib.suppress(SessionNotRunningError):
                await self.stop_session(
                    session_id, grace=1.0, cause=SessionStopCause.CRASH, restore_native=False
                )

    def _attach_control(
        self,
        session_id: str,
        harness: str,
        process: ManagedProcess,
        launch_mode: modes.LaunchMode | None,
        resume_thread_id: HarnessSessionId | None = None,
        prime: bool = True,
        model: str | None = None,
        resume_native_id: HarnessSessionId | None = None,
        model_source: ModelSource = ModelSource.GATEWAY,
        reasoning_effort: str | None = None,
        fork_thread_id: HarnessSessionId | None = None,
        fork_path: str | None = None,
        workspace_root: str | None = None,
    ) -> None:
        hub = self._hub
        kind = harness_kind(harness)
        if hub is None or kind is None or self._adapters is None:
            return
        agy = self._session_state(session_id).agy

        def remember_session_path(native_id: HarnessSessionId, path: str) -> None:
            previous = self._session_state(session_id).pi_checkpoint
            if previous is not None and previous.native_id != native_id:
                previous.recover()
            self._session_state(session_id).pi_checkpoint = record_session_path(
                process, native_id, path
            )

        adapters = self._adapters(
            AdapterContext(
                kind=kind,
                process=process,
                hub=hub,
                topic=session_topic(session_id),
                feed_start_seq=self._session_state(session_id).feed_start_seq,
                launch_mode=launch_mode,
                resume_thread_id=resume_thread_id,
                on_identity=None
                if kind is HarnessKind.PI
                else self._on_identity(session_id, resume_native_id),
                on_conversation_reset=self._on_conversation_reset(session_id, process),
                on_session_path=remember_session_path,
                on_model_selection=lambda model, effort: self._observe_pi_selection(
                    session_id, process, model, effort
                ),
                model=model,
                model_source=model_source,
                reasoning_effort=reasoning_effort,
                agy_bridge=agy.bridge if agy else None,
                agy_commands=self._session_state(session_id).agy_commands,
                fork_thread_id=fork_thread_id,
                fork_path=fork_path,
                workspace_root=workspace_root,
            )
        )
        if adapters is None:
            return
        self._session_state(session_id).control = adapters.control
        if adapters.delivery is not None:
            self._session_state(session_id).delivery = adapters.delivery
        if not prime:
            return
        task = asyncio.create_task(
            self._prime_control(session_id, adapters.control),
            name=f"control-prime:{session_id}",
        )
        self._track_control_task(session_id, task)

    async def _prime_control(self, session_id: str, control: HarnessControl) -> None:
        with contextlib.suppress(ControlError):
            await self._capture_identity(session_id, control)

    def _on_identity(
        self, session_id: str, resumed_id: HarnessSessionId | None = None
    ) -> Callable[[HarnessSessionId], None]:
        def reveal(native_id: HarnessSessionId) -> None:
            task = asyncio.create_task(self._accept_identity(session_id, native_id, resumed_id))
            self._track_control_task(session_id, task)

        return reveal

    def _on_conversation_reset(
        self, session_id: str, process: ManagedProcess
    ) -> Callable[[HarnessSessionId], asyncio.Task[None]]:
        def reset(native_id: HarnessSessionId) -> asyncio.Task[None]:
            task = asyncio.create_task(
                self._accept_conversation_reset(session_id, native_id, process)
            )
            self._track_control_task(session_id, task)
            return task

        return reset

    async def _accept_conversation_reset(
        self, session_id: str, native_id: HarnessSessionId, process: ManagedProcess
    ) -> None:
        if self._registry.process(session_id) is not process:
            return
        state = self._session_state(session_id)
        if state.native_id == native_id:
            return
        if self._sessions is not None:
            try:
                if self._registry.harness_of(session_id) == "pi":
                    await self._settle_pi_identities(session_id)
                    await self._sessions.adopt_pi_native_id(
                        SessionId(session_id), native_id, ignored_pid=process.process.pid
                    )
                else:
                    await self._sessions.rotate_native_id(SessionId(session_id), native_id)
            except SessionConflictError as error:
                if self._registry.harness_of(session_id) != "pi":
                    raise
                process.kill_now()
                raise ControlTransportError(
                    "Pi switched to a session already managed by Mandri; resume it from the sidebar"
                ) from error
        state.native_id = native_id
        await self._events.publish_event(
            session_topic(session_id),
            {
                "source": "mandri",
                "raw": {"type": "history_changed", "reset": True},
                "ts": system_now_ms(),
            },
        )

    async def _accept_identity(
        self, session_id: str, native_id: HarnessSessionId, resumed_id: HarnessSessionId | None
    ) -> None:
        if (
            resumed_id is not None
            and native_id != resumed_id
            and self._session_state(session_id).policy.execution_backend is ExecutionBackend.DOCKER
        ):
            raise SessionNotResumableError("Docker resumed a different native conversation")
        self._session_state(session_id).native_id = native_id
        if resumed_id is not None and native_id != resumed_id and self._sessions is not None:
            with contextlib.suppress(SessionNotFoundError, SessionConflictError):
                await self._sessions.rotate_native_id(SessionId(session_id), native_id)
        else:
            await self._reveal_identity(session_id, native_id)

    async def _observe_pi_selection(
        self, session_id: str, process: ManagedProcess, model: str, effort: str | None
    ) -> None:
        state = self._session_state(session_id)
        if (
            self._sessions is None
            or self._registry.process(session_id) is not process
            or state.launched_model is None
            or state.launched_model[0] is not ModelSource.NATIVE
        ):
            return
        record = await self._sessions.get_session(SessionId(session_id))
        if (record.model_source, record.model, record.reasoning_effort) != state.launched_model:
            return
        if await self._sessions.observe_native_selection(record, model, effort):
            state.launched_model = (ModelSource.NATIVE, model, effort)
            await self._events.publish_event(
                Topic("sessions.all"),
                {"source": "mandri", "raw": {"type": "sessions_changed"}, "ts": system_now_ms()},
            )

    def _require_control(self, session_id: str) -> HarnessControl:
        control = self._session_state(session_id).control
        if control is None:
            raise SteerUnsupportedError(f"session {session_id} has no control channel")
        return control

    async def _capture_identity(
        self, session_id: str, control: HarnessControl
    ) -> HarnessSessionId | None:
        task = self._session_state(session_id).identity_task
        if task is None or task.done():
            task = asyncio.create_task(control.capture_identity())
            self._session_state(session_id).identity_task = task
            task.add_done_callback(
                lambda done: setattr(self._session_state(session_id), "identity_task", None)
            )
        identity: HarnessSessionId | None = None
        with contextlib.suppress(ControlError):
            identity = await task
        await self._reveal_identity(session_id, identity)
        return identity

    async def _reveal_identity(self, session_id: str, native_id: HarnessSessionId | None) -> None:
        if native_id is not None:
            self._session_state(session_id).native_id = native_id
            self._schedule_usage_account(session_id)
            agy = self._session_state(session_id).agy
            if agy is not None:
                agy.bind(str(native_id))
        if self._sessions is None or native_id is None:
            return
        if self._registry.harness_of(session_id) == "pi":
            process = self._registry.process(session_id)
            checkpoint = self._session_state(session_id).pi_checkpoint
            await self._sessions.adopt_pi_native_id(
                SessionId(session_id), native_id,
                ignored_pid=process.process.pid if process is not None else None,
                check_writer=checkpoint is None or checkpoint.path.exists(),
            )
        else:
            with contextlib.suppress(SessionNotFoundError, SessionConflictError):
                await self._sessions.reveal_native_id(SessionId(session_id), native_id)

    def _schedule_usage_account(self, session_id: str) -> None:
        state = self._session_state(session_id)
        if self._usage_observer is None or not isinstance(state.control, CodexControlAdapter):
            return
        context = state.usage_context
        if context is None or context.routing != "native" or context.profile_id is None:
            return
        task = asyncio.create_task(self.refresh_usage_account(session_id))
        self._track_control_task(session_id, task)

    async def refresh_usage_account(self, session_id: str) -> None:
        state = self._session_state(session_id)
        context, control = state.usage_context, state.control
        if (
            context is None
            or context.profile_id is None
            or not isinstance(control, CodexControlAdapter)
        ):
            return
        if context.routing != "native" or self._usage_observer is None:
            return
        reader = self._usage_accounts.setdefault(
            context.profile_id,
            NativeAccountReader(context.profile_id, "codex"),
        )

        async def read() -> Mapping[str, Any]:
            payload = await control.read_account_rate_limits()
            await observe_usage(
                self._usage_observer,
                context,
                {
                    "method": "account/rateLimits/updated",
                    "params": payload,
                },
            )
            return payload

        snapshot = await reader.read_codex(read, qualified=True)
        if snapshot.status != "available":
            await observe_usage(
                self._usage_observer,
                context,
                {
                    "method": "account/rateLimits/unavailable",
                    "params": {},
                },
            )

    async def _attach_opencode(self, session_id: str, listen_port: int) -> HarnessSessionId | None:
        process = self._registry.process(session_id)
        native_session_id = await native_id.await_opencode_session_id(
            listen_port,
            alive=(lambda: process.returncode is None) if process is not None else None,
            **(
                {"auth": self._session_state(session_id).control_auth}
                if self._session_state(session_id).control_auth
                else {}
            ),
        )
        if native_session_id is None:
            return None
        await self._reveal_identity(session_id, native_session_id)
        if self._adapters is None:
            return None
        adapters = self._adapters(
            AdapterContext(
                kind=HarnessKind.OPENCODE,
                listen_port=listen_port,
                native_session_id=str(native_session_id),
                control_auth=self._session_state(session_id).control_auth,
            )
        )
        if adapters is None:
            return None
        self._session_state(session_id).control = adapters.control
        if adapters.delivery is not None:
            self._session_state(session_id).delivery = adapters.delivery
        self._events.start_event_pump(session_id, adapters.events)
        return native_session_id

    async def _close_control(self, session_id: str) -> None:
        self.commands.disconnected(session_id)
        try:
            await self._events.stop_pump(session_id)
        finally:
            state = self._session_state(session_id)
            control, agy = state.control, state.agy
            state.control, state.agy, state.delivery = None, None, None
            try:
                if control is not None:
                    with contextlib.suppress(ControlError):
                        await control.aclose()
            finally:
                if agy is not None:
                    await agy.aclose()

    async def _persist_interaction_mode(
        self, session_id: str, mode: str, application: ModeApplication
    ) -> None:
        if self._sessions is None:
            return
        with contextlib.suppress(SessionNotFoundError):
            await self._sessions.set_session_interaction_mode(
                SessionId(session_id), mode, application.value
            )

    def _persist_launch_mode(self, session_id: str, launch_mode: modes.LaunchMode | None) -> None:
        if self._sessions is None or launch_mode is None or launch_mode.mode is None:
            return
        task = asyncio.create_task(
            self._persist_interaction_mode(session_id, launch_mode.mode, ModeApplication.AT_LAUNCH)
        )
        self._track_control_task(session_id, task)

    async def _spawn_harness(
        self, argv: list[str], cwd: str | Path, env: dict[str, str]
    ) -> ManagedProcess:
        resolved_argv = [require_spawn_executable(argv[0]), *argv[1:]]
        return await spawn(resolved_argv, cwd=cwd, env=env)

    def docker_image_options(self) -> DockerImageOptions:
        return image_options(self._docker.config if self._docker is not None else None)

    async def docker_readiness(self) -> DockerReadiness:
        if self._docker is None:
            raise DockerExecutionError("docker_unavailable", "Docker execution is not configured")
        return await self._docker.readiness()

    async def docker_harness_version(self, harness: str) -> str:
        ready = await self.docker_readiness()
        assert self._docker is not None
        return await self._docker.versions.version(ready.image_id, harness)

    async def privacy_readiness(self) -> None:
        if self._privacy_scopes is None:
            raise ProtectionError("privacy_key_unavailable", "Privacy storage is not configured")
        await self._privacy_scopes.readiness()

    async def reconcile_docker(self) -> list[str]:
        if self._docker is None:
            return []
        removed = await self._docker.reconcile(frozenset(self._registry.live_ids()))
        active = await self._docker.client.run(
            "ps", "-q", "--no-trunc", "--filter", f"label=io.mandri.owner={self._docker.owner}"
        )
        await self._executions.reconcile(self._docker.owner, frozenset(active.split()))
        await self._reconcile_stopped_docker_sessions()
        return removed

    async def _reconcile_stopped_docker_sessions(self) -> list[str]:
        if self._docker is None:
            return []
        excluded = frozenset(self._registry.live_ids()) | self._transitioning_sessions()
        return await self._executions.reconcile_stopped_sessions(self._docker.owner, excluded)

    async def _validate_policy(
        self,
        policy: SessionPolicy,
        source: ModelSource,
        harness: str,
    ) -> None:
        policy.validate(source)
        if policy.execution_backend is ExecutionBackend.DOCKER:
            if source is ModelSource.NATIVE:
                raise DockerExecutionError(
                    "docker_native_unsupported", "Docker sessions require a gateway model"
                )
            await self.docker_readiness()
        if policy.privacy_mode is PrivacyMode.SURROGATE and self._privacy_scopes is None:
            raise ProtectionError("privacy_key_unavailable", "Privacy storage is not configured")

    async def _create_privacy_scope(
        self, policy: SessionPolicy, cwd: str | Path, source: CodexForkSourcePort | None = None
    ) -> str | None:
        if policy.privacy_mode is PrivacyMode.NONE:
            return None
        if self._privacy_scopes is None:
            raise ProtectionError("privacy_key_unavailable", "Privacy storage is not configured")
        if source is not None and source.session.privacy_mode is PrivacyMode.SURROGATE:
            if source.session.privacy_scope_id is None:
                raise ProtectionError(
                    "privacy_state_unavailable", "The source privacy scope is unavailable"
                )
            return await self._privacy_scopes.fork(source.session.privacy_scope_id)
        return await self._privacy_scopes.create(str(Path(cwd).resolve()))

    async def _spawn_execution(
        self,
        session_id: str,
        harness: str,
        prepared: PreparedLaunch,
        cwd: str | Path,
        policy: SessionPolicy,
        route_id: str | None,
        model: str,
        launch_mode: modes.LaunchMode | None,
        resume_native_id: HarnessSessionId | None = None,
    ) -> ManagedProcess:
        if harness == "opencode":
            password = secrets.token_urlsafe(32)
            prepared.env["OPENCODE_SERVER_PASSWORD"] = password
            prepared.env["OPENCODE_PASSWORD"] = password
            self._session_state(session_id).control_auth = ("opencode", password)
        owner = (
            self._docker.owner
            if policy.execution_backend is ExecutionBackend.DOCKER and self._docker
            else "host"
        )
        await self._executions.begin(
            session_id,
            owner,
            {
                "execution_backend": policy.execution_backend.value,
                "privacy_mode": policy.privacy_mode.value,
                "operation_id": self._start_operations.current_id,
            },
        )
        if policy.execution_backend is ExecutionBackend.HOST:
            await self._executions.phase(session_id, ExecutionPhase.STARTING)
            host_process = await self._spawn_harness(prepared.argv, cwd=cwd, env=prepared.env)
            if harness == "agy":

                async def spawn_host_command(argv: list[str]) -> ManagedProcess:
                    return await self._spawn_harness(argv, cwd=cwd, env=prepared.env)

                self._session_state(session_id).agy_commands = AgyCommandRunner(
                    prepared.argv, spawn_host_command
                )
            return host_process
        backend = self._docker
        if (
            backend is None
            or self._gateway_port is None
            or self._token_issuer is None
            or route_id is None
        ):
            raise DockerExecutionError("docker_unavailable", "Docker requires a configured gateway")
        ingress = WorkerIngress(self._gateway_port, route_id, self._token_issuer(route_id))
        try:
            await self._executions.phase(session_id, ExecutionPhase.PREPARING_IMAGE)
            socket_path = (
                backend.config.state_root.parent / "ingress" / f"{uuid.uuid4().hex}.sock"
                if backend.config.ingress_host == "host.docker.internal"
                else None
            )
            port = await ingress.start(backend.config.ingress_bind, socket_path=socket_path)
            ingress_url = f"http://{backend.config.ingress_host}:{port}"
            argv = [
                harness,
                *[
                    translate_value(value, self._gateway_port, ingress_url)
                    for value in prepared.argv[1:]
                ],
            ]
            env = {
                key: translate_value(value, self._gateway_port, ingress_url)
                for key, value in prepared.env.items()
            }
            if harness == "opencode":
                argv.extend(["--hostname", "0.0.0.0"])
            ready = await backend.readiness()
            await self._executions.phase(session_id, ExecutionPhase.PREPARING_STATE)
            context = backend.context(
                session_id, cwd, resume=resume_native_id is not None, image_id=ready.image_id
            )
            launch = PreparedLaunch(argv, env, prepared.listen_port)
            if harness == "agy":

                async def publish(raw: dict[str, Any]) -> None:
                    await self._events.publish_event(
                        session_topic(session_id),
                        {"source": "agy", "raw": raw, "ts": system_now_ms()},
                    )

                launch, resource = prepare_docker_agy(
                    launch,
                    session_id,
                    Path(cwd),
                    Path(context["native_state_root"]),
                    model,
                    launch_mode.mode if launch_mode and launch_mode.mode else "default",
                    self._approval_timeout_seconds,
                    ingress,
                    ingress_url,
                    publish,
                    str(resume_native_id) if resume_native_id else None,
                )
                self._session_state(session_id).agy = resource
            if self._sessions is not None:
                await self._sessions.set_execution_context(
                    SessionId(session_id), json.dumps(context)
                )
            await self._executions.phase(session_id, ExecutionPhase.STARTING, context=context)
            process = await backend.spawn(
                session_id,
                harness,
                launch.argv,
                cwd,
                launch.env,
                ingress,
                listen_port=launch.listen_port,
                resume=resume_native_id is not None,
            )
            if harness == "agy":

                async def spawn_command(argv: list[str]) -> ManagedProcess:
                    return await spawn_container_command(backend.client, process.container_id, argv)

                self._session_state(session_id).agy_commands = AgyCommandRunner(
                    launch.argv, spawn_command, timeout=25
                )
            try:
                if self._sessions is not None:
                    await self._sessions.set_execution_context(
                        SessionId(session_id), json.dumps(process.execution_context)
                    )
                await self._executions.phase(
                    session_id,
                    ExecutionPhase.STARTING,
                    context=process.execution_context,
                    container_id=process.container_id,
                )
            except BaseException:
                await process.stop()
                raise
            return process
        except BaseException:
            await ingress.aclose()
            raise

    async def _bind_route(
        self,
        harness: str,
        model: str,
        effort: str | None = None,
        policy: SessionPolicy = _DEFAULT_POLICY,
        privacy_scope_id: str | None = None,
    ) -> str:
        kind = harness_kind(harness)
        if self._routes is None or kind is None:
            return str(uuid.uuid4())
        provider_name, model_id = parse_model_arg(model)
        policy_options: dict[str, Any] = {}
        if policy != _DEFAULT_POLICY:
            policy_options.update(
                execution_backend=policy.execution_backend,
                privacy_mode=policy.privacy_mode,
                privacy_scope_id=privacy_scope_id,
            )
        route = await self._routes.create(
            provider_name,
            model_id,
            HARNESS_WIRE_FORMATS[kind],
            reasoning_effort=effort,
            **policy_options,
        )
        return str(route.id)

    async def _resolve_metadata(self, model: str) -> ModelMetadata | None:
        routes = self._routes
        if routes is None:
            return None
        try:
            provider_name, model_id = parse_model_arg(model)
            provider = routes.providers.get(provider_name)
        except Exception:
            return None
        return await fetch_model_metadata(
            provider.kind, model_id, provider.api_base, str(provider.api_key)
        )

    async def _create_session_record(
        self,
        harness: str,
        model: str,
        route_id: str | None,
        cwd: str | Path,
        effort: str | None = None,
        model_source: ModelSource = ModelSource.GATEWAY,
        policy: SessionPolicy = _DEFAULT_POLICY,
        privacy_scope_id: str | None = None,
        starting: bool = False,
    ) -> str:
        kind = harness_kind(harness)
        if self._sessions is None or kind is None:
            return str(uuid.uuid4())
        policy_options: dict[str, Any] = {}
        if policy != _DEFAULT_POLICY:
            policy_options.update(
                execution_backend=policy.execution_backend,
                privacy_mode=policy.privacy_mode,
                privacy_scope_id=privacy_scope_id,
            )
        if model_source is ModelSource.NATIVE:
            policy_options["model_source"] = model_source
        if starting:
            policy_options["initial_state"] = SessionState.LIVE
        record = await self._sessions.create_session(
            kind,
            model,
            None if route_id is None else RouteId(route_id),
            ProjectPath(str(cwd)),
            reasoning_effort=effort,
            **policy_options,
        )
        return str(record.id)

    async def _mark_db_state(self, session_id: str, state: SessionState) -> None:
        if self._sessions is None:
            return
        with contextlib.suppress(SessionNotFoundError):
            await self._sessions.set_session_state(SessionId(session_id), state)

    @staticmethod
    def validate_model_selection(harness: str, model: str, source: ModelSource) -> None:
        ModelSelectionService.validate_model_selection(harness, model, source)

    async def native_models(self, harness: str, cwd: str | None = None) -> list[NativeModel]:
        return await self._models.native_models(harness, cwd, self._spawn_harness)

    async def _apply_pending_model(self, session_id: str) -> None:
        state = self._session_state(session_id)
        stop_revision = state.stop_revision
        if await self._models.needs_restart(session_id):
            if state.stop_revision != stop_revision:
                raise SessionNotRunningError(f"session {session_id} was stopped")
            if state.resuming or state.stopping:
                raise SessionRunningError(f"session {session_id} is already changing state")
            state.resuming = True
            try:
                await self.stop_session(session_id, restore_native=False)
                if state.stop_revision != stop_revision:
                    raise SessionNotRunningError(f"session {session_id} was stopped")
                await self._resume_session(session_id)
            finally:
                state.resuming = False
