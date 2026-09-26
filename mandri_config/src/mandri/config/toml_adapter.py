"""TOML file adapter for daemon configuration."""

import contextlib
import dataclasses
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomlkit
from mandri.config.errors import ConfigError, ConfigParseError
from mandri.config.protection import parse_docker, parse_privacy
from mandri.config.types import (
    CLAUDE_MODES,
    CODEX_APPROVAL_POLICIES,
    CODEX_SANDBOX_MODES,
    HARNESS_KINDS,
    LOCAL_PROVIDERS,
    PROVIDER_KINDS,
)
from mandri.core.ports.config import ConfigPort
from mandri.core.types.config import (
    DEFAULT_CORS_ORIGINS,
    ApprovalsConfig,
    ApprovalTimeoutSeconds,
    ClaudeMode,
    CodexApprovalPolicy,
    CodexSandboxMode,
    CorsOrigin,
    DaemonConfig,
    DefaultsConfig,
    OpencodeAgentMode,
    ProviderConfig,
    ServerConfig,
    SessionModeConfig,
    SessionsConfig,
    SyncConfig,
)
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from tomlkit.items import AoT, Table

DEFAULT_BASE_DIR = Path.home() / ".mandri"
_PREVIOUS_DEFAULT_CORS_ORIGINS = frozenset(
    origin for origin in DEFAULT_CORS_ORIGINS if not origin.endswith(":5174")
)


class TomlConfigAdapter(ConfigPort):
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir if base_dir is not None else DEFAULT_BASE_DIR
        self._config_path = self._base_dir / "config.toml"

    @property
    def config_path(self) -> Path:
        return self._config_path

    def load(self) -> DaemonConfig:
        if not self._config_path.is_file():
            return DaemonConfig()
        try:
            data = tomllib.loads(self._config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigParseError(f"invalid TOML in {self._config_path}: {exc}") from exc
        return _parse_config(data)

    def save(self, config: DaemonConfig) -> None:
        _validate(config)
        doc = self._read_doc()
        _write_table(
            doc,
            "server",
            {
                "host": config.server.host,
                "port": config.server.port,
                "cors_origins": list(config.server.cors_origins) or None,
            },
        )
        _write_table(
            doc,
            "sync",
            {
                "ttl_seconds": config.sync.ttl_seconds,
                "mtime_poll_seconds": config.sync.mtime_poll_seconds,
                "backoff_max_seconds": config.sync.backoff_max_seconds,
                "activity_quiet_seconds": config.sync.activity_quiet_seconds,
            },
        )
        _write_table(
            doc,
            "defaults",
            {
                "harness": config.defaults.harness,
                "model": config.defaults.model,
                "execution_backend": config.defaults.execution_backend.value,
                "privacy_mode": config.defaults.privacy_mode.value,
            },
        )
        _write_table(doc, "docker", dataclasses.asdict(config.docker))
        privacy = dataclasses.asdict(config.privacy)
        table = doc.get("privacy")
        if isinstance(table, Table):
            for key in list(table):
                if key not in privacy:
                    del table[key]
        _write_table(doc, "privacy", privacy)
        _write_sessions_table(doc, config.sessions)
        _write_approvals_table(doc, config.approvals)
        _write_providers(doc, config.providers)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(tomlkit.dumps(doc))

    def _atomic_write(self, content: str) -> None:
        handle_fd, tmp_name = tempfile.mkstemp(dir=self._base_dir, suffix=".tmp")
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp_name, self._config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        _restrict_permissions(self._config_path)

    def _read_doc(self) -> tomlkit.TOMLDocument:
        if not self._config_path.is_file():
            return tomlkit.document()
        try:
            return tomlkit.parse(self._config_path.read_text(encoding="utf-8"))
        except tomlkit.exceptions.ParseError as exc:
            raise ConfigParseError(f"invalid TOML in {self._config_path}: {exc}") from exc


def _restrict_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _parse_config(data: dict[str, Any]) -> DaemonConfig:
    _reject_unknown_keys("config", data, KNOWN_TOP_LEVEL_KEYS)
    providers_data = data.get("providers", [])
    if not isinstance(providers_data, list):
        raise ConfigError("providers must be an array of tables")
    providers = [_parse_provider(entry, index) for index, entry in enumerate(providers_data)]
    _validate_providers(providers)
    return DaemonConfig(
        server=_parse_server(data.get("server", {})),
        sync=_parse_sync(data.get("sync", {})),
        providers=providers,
        defaults=_parse_defaults(data.get("defaults", {})),
        sessions=_parse_sessions(data.get("sessions", {})),
        approvals=_parse_approvals(data.get("approvals", {})),
        docker=parse_docker(data.get("docker", {})),
        privacy=parse_privacy(data.get("privacy", {})),
    )


KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"server", "sync", "providers", "defaults", "sessions", "approvals", "docker", "privacy"}
)


def _reject_unknown_keys(prefix: str, data: dict[str, Any], known: frozenset[str]) -> None:
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in {prefix}, expected one of {sorted(known)}"
        )


def _parse_server(data: dict[str, Any]) -> ServerConfig:
    host = data.get("host", "127.0.0.1")
    port = data.get("port", 8787)
    if not isinstance(host, str) or not isinstance(port, int) or isinstance(port, bool):
        raise ConfigError("server.host must be a string and server.port must be an integer")
    return ServerConfig(
        host=host,
        port=port,
        cors_origins=_parse_cors_origins(data.get("cors_origins", DEFAULT_CORS_ORIGINS)),
    )


def _parse_cors_origins(value: Any) -> list[CorsOrigin]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ConfigError("server.cors_origins must be an array of non-empty strings")
    if frozenset(value) == _PREVIOUS_DEFAULT_CORS_ORIGINS:
        return list(DEFAULT_CORS_ORIGINS)
    return [CorsOrigin(item) for item in value]


def _parse_sync(data: dict[str, Any]) -> SyncConfig:
    values = {key: data.get(key, default) for key, default in _sync_defaults().items()}
    for key, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"sync.{key} must be a non-negative integer")
    return SyncConfig(**values)


def _sync_defaults() -> dict[str, int]:
    defaults = SyncConfig()
    return {
        "ttl_seconds": defaults.ttl_seconds,
        "mtime_poll_seconds": defaults.mtime_poll_seconds,
        "backoff_max_seconds": defaults.backoff_max_seconds,
        "activity_quiet_seconds": defaults.activity_quiet_seconds,
    }


def _parse_defaults(data: dict[str, Any]) -> DefaultsConfig:
    harness = data.get("harness")
    model = data.get("model")
    if harness is not None and not isinstance(harness, str):
        raise ConfigError("defaults.harness must be a string")
    if model is not None and not isinstance(model, str):
        raise ConfigError("defaults.model must be a string")
    try:
        backend = ExecutionBackend(data.get("execution_backend", "host"))
        privacy = PrivacyMode(data.get("privacy_mode", "none"))
    except (ValueError, TypeError):
        raise ConfigError("Invalid default execution_backend or privacy_mode") from None
    return DefaultsConfig(
        harness=harness, model=model, execution_backend=backend, privacy_mode=privacy
    )


SESSIONS_KEYS = (
    "opencode_db_path",
    "opencode_base_url",
    "codex_home",
    "claude_config_dir",
    "agy_home",
    "agy_profiles_dir",
)

SESSIONS_DIRECTIVE_KEYS: frozenset[str] = frozenset(
    {*SESSIONS_KEYS, "launch_args", "reconcile_interval_seconds", "idle_release_seconds", "mode"}
)


def _parse_sessions(data: dict[str, Any]) -> SessionsConfig:
    _reject_unknown_keys("sessions", data, SESSIONS_DIRECTIVE_KEYS)
    values: dict[str, str | None] = {}
    for key in SESSIONS_KEYS:
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"sessions.{key} must be a string")
        values[key] = value
    return SessionsConfig(
        **values,
        launch_args=_parse_launch_args(data.get("launch_args", {})),
        reconcile_interval_seconds=_parse_reconcile_interval(
            data.get("reconcile_interval_seconds")
        ),
        mode=_parse_mode(data.get("mode", {})),
        idle_release_seconds=_parse_idle_release(data.get("idle_release_seconds", 60)),
    )


def _parse_idle_release(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError("sessions.idle_release_seconds must be a positive integer")
    return value


def _parse_mode(data: dict[str, Any]) -> SessionModeConfig:
    if not isinstance(data, dict):
        raise ConfigError("sessions.mode must be a table")
    _reject_unknown_keys("sessions.mode", data, MODE_KEYS)
    return SessionModeConfig(
        agy=_parse_mode_value(
            "agy",
            data.get("agy"),
            frozenset({"default", "acceptEdits", "plan", "bypassPermissions"}),
            str,
        ),
        claude=_parse_mode_value("claude", data.get("claude"), CLAUDE_MODES, ClaudeMode),
        codex_approval_policy=_parse_mode_value(
            "codex_approval_policy",
            data.get("codex_approval_policy"),
            CODEX_APPROVAL_POLICIES,
            CodexApprovalPolicy,
        ),
        codex_sandbox=_parse_mode_value(
            "codex_sandbox",
            data.get("codex_sandbox"),
            CODEX_SANDBOX_MODES,
            CodexSandboxMode,
        ),
        opencode_agent=_parse_mode_value(
            "opencode_agent",
            data.get("opencode_agent"),
            None,
            OpencodeAgentMode,
        ),
    )


MODE_KEYS: frozenset[str] = frozenset(
    {"claude", "codex_approval_policy", "codex_sandbox", "opencode_agent", "agy"}
)


def _parse_mode_value(
    key: str,
    value: Any,
    allowed: frozenset[str] | None,
    domain_type: type,
) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"sessions.mode.{key} must be a string")
    if allowed is not None and value not in allowed:
        raise ConfigError(
            f"sessions.mode.{key} '{value}' is invalid, expected one of {sorted(allowed)}"
        )
    return domain_type(value)


APPROVALS_KEYS: frozenset[str] = frozenset({"timeout_seconds"})


def _parse_approvals(data: dict[str, Any]) -> ApprovalsConfig:
    if not isinstance(data, dict):
        raise ConfigError("approvals must be a table")
    _reject_unknown_keys("approvals", data, APPROVALS_KEYS)
    timeout_seconds = data.get("timeout_seconds", ApprovalsConfig().timeout_seconds)
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise ConfigError("approvals.timeout_seconds must be an integer")
    if timeout_seconds <= 0:
        raise ConfigError("approvals.timeout_seconds must be a positive integer")
    return ApprovalsConfig(timeout_seconds=ApprovalTimeoutSeconds(timeout_seconds))


def _parse_reconcile_interval(value: Any) -> int:
    default = SessionsConfig().reconcile_interval_seconds
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError("sessions.reconcile_interval_seconds must be a positive integer")
    return value


def _parse_launch_args(data: Any) -> dict[str, list[str]]:
    if not isinstance(data, dict):
        raise ConfigError("sessions.launch_args must be a table")
    launch_args: dict[str, list[str]] = {}
    for harness, argv in data.items():
        if harness not in HARNESS_KINDS:
            raise ConfigError(
                f"sessions.launch_args.{harness} is not a known harness, "
                f"expected one of {sorted(HARNESS_KINDS)}"
            )
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ConfigError(
                f"sessions.launch_args.{harness} must be a non-empty array of strings"
            )
        launch_args[harness] = list(argv)
    return launch_args


def _write_sessions_table(doc: tomlkit.TOMLDocument, sessions: SessionsConfig) -> None:
    values = {key: getattr(sessions, key) for key in SESSIONS_KEYS}
    if (
        not sessions.launch_args
        and sessions.reconcile_interval_seconds == SessionsConfig().reconcile_interval_seconds
        and sessions.idle_release_seconds == SessionsConfig().idle_release_seconds
        and sessions.mode == SessionModeConfig()
        and all(value is None for value in values.values())
        and "sessions" not in doc
    ):
        return
    table = doc.get("sessions")
    if not isinstance(table, Table):
        table = tomlkit.table()
        doc["sessions"] = table
    for key, value in values.items():
        if value is None:
            if key in table:
                del table[key]
        else:
            table[key] = value
    table["reconcile_interval_seconds"] = sessions.reconcile_interval_seconds
    table["idle_release_seconds"] = sessions.idle_release_seconds
    _write_mode_table(table, sessions.mode)
    if not sessions.launch_args:
        if "launch_args" in table:
            del table["launch_args"]
        return
    launch_args = table.get("launch_args")
    if not isinstance(launch_args, Table):
        launch_args = tomlkit.table()
        table["launch_args"] = launch_args
    for harness, argv in sessions.launch_args.items():
        launch_args[harness] = list(argv)


def _write_mode_table(table: Table, mode: SessionModeConfig) -> None:
    if mode == SessionModeConfig():
        if "mode" in table:
            del table["mode"]
        return
    mode_table = table.get("mode")
    if not isinstance(mode_table, Table):
        mode_table = tomlkit.table()
        table["mode"] = mode_table
    for key in MODE_KEYS:
        value = getattr(mode, key)
        if value is None:
            if key in mode_table:
                del mode_table[key]
        else:
            mode_table[key] = value


def _write_approvals_table(doc: tomlkit.TOMLDocument, approvals: ApprovalsConfig) -> None:
    if approvals == ApprovalsConfig() and "approvals" not in doc:
        return
    table = doc.get("approvals")
    if not isinstance(table, Table):
        table = tomlkit.table()
        doc["approvals"] = table
    table["timeout_seconds"] = approvals.timeout_seconds


def _parse_provider(entry: dict[str, Any], index: int) -> ProviderConfig:
    name = entry.get("name")
    kind = entry.get("kind")
    api_base = entry.get("api_base")
    api_key = entry.get("api_key", "")
    if not isinstance(name, str) or not isinstance(kind, str):
        raise ConfigError(f"providers[{index}] requires string name and kind")
    if api_base is not None and not isinstance(api_base, str):
        raise ConfigError(f"providers[{index}] api_base must be a string")
    if not isinstance(api_key, str):
        raise ConfigError(f"providers[{index}] api_key must be a string")
    return ProviderConfig(name=name, kind=kind, api_base=api_base, api_key=api_key)


def _validate(config: DaemonConfig) -> None:
    parse_docker(dataclasses.asdict(config.docker))
    parse_privacy(dataclasses.asdict(config.privacy))
    _parse_defaults(dataclasses.asdict(config.defaults))
    _validate_providers(config.providers)
    if config.approvals.timeout_seconds <= 0:
        raise ConfigError("approvals.timeout_seconds must be a positive integer")
    _validate_mode(config.sessions.mode)


def _validate_mode(mode: SessionModeConfig) -> None:
    for key, value, allowed in (
        ("claude", mode.claude, CLAUDE_MODES),
        ("codex_approval_policy", mode.codex_approval_policy, CODEX_APPROVAL_POLICIES),
        ("codex_sandbox", mode.codex_sandbox, CODEX_SANDBOX_MODES),
    ):
        if value is not None and value not in allowed:
            raise ConfigError(
                f"sessions.mode.{key} '{value}' is invalid, expected one of {sorted(allowed)}"
            )


def _validate_providers(providers: list[ProviderConfig]) -> None:
    seen: set[str] = set()
    for index, provider in enumerate(providers):
        _validate_provider(provider, index, seen)


def _validate_provider(provider: ProviderConfig, index: int, seen: set[str]) -> None:
    if provider.kind not in PROVIDER_KINDS:
        raise ConfigError(
            f"providers[{index}]: unknown kind '{provider.kind}', "
            f"expected one of {sorted(PROVIDER_KINDS)}"
        )
    if not provider.name:
        raise ConfigError(f"providers[{index}]: name must be a non-empty string")
    if provider.name in seen:
        raise ConfigError(f"providers[{index}]: duplicate provider name '{provider.name}'")
    seen.add(provider.name)
    if provider.kind in LOCAL_PROVIDERS and not provider.api_base:
        raise ConfigError(f"providers[{index}]: provider '{provider.name}' requires api_base")


def _write_table(doc: tomlkit.TOMLDocument, name: str, values: dict[str, Any]) -> None:
    if all(value is None for value in values.values()) and name not in doc:
        return
    table = doc.get(name)
    if not isinstance(table, Table):
        table = tomlkit.table()
        doc[name] = table
    for key, value in values.items():
        if value is None:
            if key in table:
                del table[key]
        else:
            table[key] = value


def _write_providers(doc: tomlkit.TOMLDocument, providers: list[ProviderConfig]) -> None:
    existing = doc.get("providers")
    if isinstance(existing, AoT) and len(existing) == len(providers):
        for table, provider in zip(existing, providers, strict=True):
            _fill_provider_table(table, provider)
        return
    if "providers" in doc:
        del doc["providers"]
    if not providers:
        return
    aot = tomlkit.aot()
    for provider in providers:
        table = tomlkit.table()
        _fill_provider_table(table, provider)
        aot.append(table)
    doc["providers"] = aot


def _fill_provider_table(table: Table, provider: ProviderConfig) -> None:
    values: dict[str, Any] = {
        "name": provider.name,
        "kind": provider.kind,
        "api_base": provider.api_base,
        "api_key": provider.api_key or None,
    }
    for key, value in values.items():
        if value is None:
            if key in table:
                del table[key]
        else:
            table[key] = value
