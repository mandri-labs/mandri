"""Daemon configuration domain types."""

from dataclasses import dataclass, field
from typing import NewType

from mandri.core.types.execution import ExecutionBackend, PrivacyMode

ApprovalTimeoutSeconds = NewType("ApprovalTimeoutSeconds", int)
ClaudeMode = NewType("ClaudeMode", str)
CodexApprovalPolicy = NewType("CodexApprovalPolicy", str)
CodexSandboxMode = NewType("CodexSandboxMode", str)
CorsOrigin = NewType("CorsOrigin", str)
OpencodeAgentMode = NewType("OpencodeAgentMode", str)

APPROVAL_TIMEOUT_SECONDS_DEFAULT = 120

DEFAULT_CORS_ORIGINS: tuple[CorsOrigin, ...] = (
    CorsOrigin("http://localhost:1420"),
    CorsOrigin("http://127.0.0.1:1420"),
    CorsOrigin("http://tauri.localhost"),
    CorsOrigin("https://tauri.localhost"),
    CorsOrigin("tauri://localhost"),
    CorsOrigin("http://localhost:5173"),
    CorsOrigin("http://127.0.0.1:5173"),
    CorsOrigin("http://localhost:5174"),
    CorsOrigin("http://127.0.0.1:5174"),
)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str
    api_base: str | None = None
    api_key: str = ""


@dataclass(frozen=True)
class DefaultsConfig:
    harness: str | None = None
    model: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE


@dataclass(frozen=True)
class DockerSettings:
    image: str | None = "mandri-worker:latest"
    pull_policy: str = "never"
    pull_timeout_seconds: int = 600
    state_root: str | None = None
    cpus: float = 4.0
    memory_mb: int = 8192
    pids_limit: int = 1024
    tmpfs_mb: int = 1024
    ingress_bind: str = "0.0.0.0"
    ingress_host: str = "host.docker.internal"
    docker_binary: str = "docker"
    network_exceptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrivacySettings:
    key_file: str | None = None
    key_service: str = "mandri-privacy"
    rules_file: str | None = None


@dataclass(frozen=True)
class SyncConfig:
    ttl_seconds: int = 30
    mtime_poll_seconds: int = 10
    backoff_max_seconds: int = 30
    activity_quiet_seconds: int = 60


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    cors_origins: list[CorsOrigin] = field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))


@dataclass(frozen=True)
class ApprovalsConfig:
    timeout_seconds: ApprovalTimeoutSeconds = ApprovalTimeoutSeconds(
        APPROVAL_TIMEOUT_SECONDS_DEFAULT
    )


@dataclass(frozen=True)
class SessionModeConfig:
    claude: ClaudeMode | None = None
    codex_approval_policy: CodexApprovalPolicy | None = None
    codex_sandbox: CodexSandboxMode | None = None
    opencode_agent: OpencodeAgentMode | None = None
    agy: str | None = None


@dataclass(frozen=True)
class SessionsConfig:
    opencode_db_path: str | None = None
    opencode_base_url: str | None = None
    codex_home: str | None = None
    claude_config_dir: str | None = None
    agy_home: str | None = None
    agy_profiles_dir: str | None = None
    launch_args: dict[str, list[str]] = field(default_factory=dict)
    reconcile_interval_seconds: int = 10
    idle_release_seconds: int = 60
    mode: SessionModeConfig = field(default_factory=SessionModeConfig)


@dataclass(frozen=True)
class DaemonConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    providers: list[ProviderConfig] = field(default_factory=list)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    approvals: ApprovalsConfig = field(default_factory=ApprovalsConfig)
    docker: DockerSettings = field(default_factory=DockerSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
