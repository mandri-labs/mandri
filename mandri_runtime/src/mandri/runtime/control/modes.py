"""Interaction-mode validation and per-harness launch composition."""

import dataclasses
from typing import Any

from mandri.config.errors import ConfigError
from mandri.config.types import (
    CLAUDE_MODES,
    CODEX_APPROVAL_POLICIES,
    CODEX_SANDBOX_MODES,
)
from mandri.core.ids import HarnessKind
from mandri.core.types.config import SessionModeConfig

_CLAUDE_PERMISSION_FLAG = "--permission-mode"
_CODEX_POLICY_KEY = "approvalPolicy"
_CODEX_SANDBOX_KEY = "sandbox"
_OPENCODE_AGENT_KEY = "agent"
_OPENCODE_PERMISSION_KEY = "permission"

CODEX_PROFILES: dict[str, dict[str, str]] = {
    "ask": {
        "approvalPolicy": "on-request",
        "sandbox": "workspace-write",
        "approvalsReviewer": "user",
    },
    "auto": {
        "approvalPolicy": "on-request",
        "sandbox": "workspace-write",
        "approvalsReviewer": "auto_review",
    },
    "full-access": {
        "approvalPolicy": "never",
        "sandbox": "danger-full-access",
        "approvalsReviewer": "user",
    },
}

OPENCODE_PERMISSION_RULES: dict[str, dict[str, str]] = {
    "default": {"*": "ask"},
    "auto": {"*": "allow"},
    "bypassPermissions": {"bash": "allow", "edit": "allow", "webfetch": "allow"},
    "acceptEdits": {"edit": "allow"},
    "plan": {"bash": "deny", "edit": "deny", "webfetch": "deny"},
}


@dataclasses.dataclass(frozen=True)
class LaunchMode:
    mode: str | None = None
    auto_approve: bool = False
    claude_args: tuple[str, ...] = ()
    codex_params: dict[str, str] = dataclasses.field(default_factory=dict)
    opencode_config: dict[str, Any] = dataclasses.field(default_factory=dict)


def validate_claude_mode(mode: str) -> None:
    if mode not in CLAUDE_MODES:
        raise ConfigError(
            f"unknown claude permission mode {mode!r}, expected one of {sorted(CLAUDE_MODES)}"
        )


def validate_codex_mode(policy: str | None, sandbox: str | None) -> None:
    if policy is not None and policy not in CODEX_APPROVAL_POLICIES:
        raise ConfigError(
            f"unknown codex approval policy {policy!r}, expected one of "
            f"{sorted(CODEX_APPROVAL_POLICIES)}"
        )
    if sandbox is not None and sandbox not in CODEX_SANDBOX_MODES:
        raise ConfigError(
            f"unknown codex sandbox mode {sandbox!r}, expected one of {sorted(CODEX_SANDBOX_MODES)}"
        )


def validate_opencode_agent(agent: str) -> None:
    if not agent.strip():
        raise ConfigError("opencode agent name must not be blank")


def claude_launch_args(mode: str) -> list[str]:
    validate_claude_mode(mode)
    return [_CLAUDE_PERMISSION_FLAG, mode]


def codex_session_params(policy: str | None, sandbox: str | None) -> dict[str, str]:
    validate_codex_mode(policy, sandbox)
    params: dict[str, str] = {}
    if policy is not None:
        params[_CODEX_POLICY_KEY] = policy
    if sandbox is not None:
        params[_CODEX_SANDBOX_KEY] = sandbox
    return params


def opencode_permission_rules(mode: str) -> dict[str, str]:
    rules = OPENCODE_PERMISSION_RULES.get(mode)
    if rules is None:
        raise ConfigError(
            f"unknown opencode interaction mode {mode!r}, expected one of "
            f"{sorted(OPENCODE_PERMISSION_RULES)}"
        )
    return dict(rules)


def resolve_launch(
    harness: str, mode: str | None, defaults: SessionModeConfig
) -> LaunchMode | None:
    kind = _kind_of(harness)
    if kind is None:
        return None
    match kind:
        case HarnessKind.PI:
            if mode not in {None, "default", "acceptEdits", "plan", "bypassPermissions"}:
                raise ConfigError("Unsupported Pi permission mode")
            return LaunchMode(mode=mode or "default")
        case HarnessKind.AGY:
            effective = mode or defaults.agy or "default"
            if effective not in {"default", "acceptEdits", "plan", "bypassPermissions"}:
                raise ConfigError(f"unknown Antigravity permission mode {effective!r}")
            return LaunchMode(mode=effective)
        case HarnessKind.CLAUDE:
            return _resolve_claude(mode, defaults)
        case HarnessKind.CODEX:
            return _resolve_codex(mode, defaults)
        case HarnessKind.OPENCODE:
            return _resolve_opencode(mode, defaults)


def _resolve_claude(mode: str | None, defaults: SessionModeConfig) -> LaunchMode:
    effective = mode if mode is not None else defaults.claude
    if effective is None:
        return LaunchMode()
    validate_claude_mode(effective)
    return LaunchMode(mode=effective, claude_args=tuple(claude_launch_args(effective)))


def _resolve_codex(mode: str | None, defaults: SessionModeConfig) -> LaunchMode:
    if mode in CODEX_PROFILES:
        return LaunchMode(mode=mode, codex_params=dict(CODEX_PROFILES[mode]))
    policy = mode if mode is not None else defaults.codex_approval_policy
    sandbox = defaults.codex_sandbox
    validate_codex_mode(policy, sandbox)
    return LaunchMode(mode=policy, codex_params=codex_session_params(policy, sandbox))


def _resolve_opencode(mode: str | None, defaults: SessionModeConfig) -> LaunchMode:
    effective = mode if mode is not None else defaults.opencode_agent
    if effective is None:
        return LaunchMode()
    rules = OPENCODE_PERMISSION_RULES.get(effective)
    if rules is not None:
        return LaunchMode(mode=effective, opencode_config={_OPENCODE_PERMISSION_KEY: dict(rules)})
    validate_opencode_agent(effective)
    return LaunchMode(mode=effective, opencode_config={_OPENCODE_AGENT_KEY: effective})


def _kind_of(harness: str) -> HarnessKind | None:
    try:
        return HarnessKind(harness)
    except ValueError:
        return None


def agy_launch_args(mode: str) -> list[str]:
    if mode == "acceptEdits":
        return ["--mode", "accept-edits"]
    if mode == "plan":
        return ["--mode", "plan"]
    return []
