"""Typed identifiers, epochs, URLs, and harness/provider enums."""

import dataclasses
import enum
import typing
from collections.abc import Mapping

SessionId = typing.NewType("SessionId", str)
HarnessSessionId = typing.NewType("HarnessSessionId", str)
SessionTitle = typing.NewType("SessionTitle", str)
ProjectPath = typing.NewType("ProjectPath", str)
ModelId = typing.NewType("ModelId", str)
ModelName = typing.NewType("ModelName", str)
ModelRef = typing.NewType("ModelRef", str)
RouteId = typing.NewType("RouteId", str)
SecretRef = typing.NewType("SecretRef", str)
EpochMs = typing.NewType("EpochMs", int)
Url = typing.NewType("Url", str)
ApprovalId = typing.NewType("ApprovalId", str)
CorrelationId = typing.NewType("CorrelationId", str)
PageToken = typing.NewType("PageToken", str)
FsPath = typing.NewType("FsPath", str)
RawEvent = typing.NewType("RawEvent", str)


class HarnessKind(enum.StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"
    OPENCODE = "opencode"
    AGY = "agy"
    PI = "pi"


class ProviderKind(enum.StrEnum):
    OPENROUTER = "openrouter"
    OPENCODE = "opencode"
    OPENCODE_GO = "opencode_go"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    CUSTOM = "custom"


class SessionState(enum.StrEnum):
    DISCOVERED = "discovered"
    LIVE = "live"
    STOPPED = "stopped"


class ModelState(enum.StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DEGRADED = "degraded"


class WireFormat(enum.StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


HARNESS_WIRE_FORMATS: Mapping[HarnessKind, tuple[WireFormat, ...]] = {
    HarnessKind.CODEX: (WireFormat.OPENAI,),
    HarnessKind.CLAUDE: (WireFormat.ANTHROPIC,),
    HarnessKind.OPENCODE: (WireFormat.OPENAI,),
    HarnessKind.AGY: (WireFormat.GEMINI,),
    HarnessKind.PI: (WireFormat.OPENAI,),
}


class ActivityState(enum.StrEnum):
    ACTIVE = "active"
    IDLE = "idle"


class SessionStopCause(enum.StrEnum):
    IDLE_TIMEOUT = "idle_timeout"
    VIEWER_STOP = "viewer_stop"
    CRASH = "crash"
    DAEMON_STOP = "daemon_stop"


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalDecision(enum.StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ONCE = "once"
    ALWAYS = "always"
    ACCEPT = "accept"
    ACCEPT_FOR_SESSION = "acceptForSession"
    DECLINE = "decline"
    CANCEL = "cancel"


class ApprovalKind(enum.StrEnum):
    COMMAND_EXECUTION = "command_execution"
    FILE_CHANGE = "file_change"
    PERMISSION_SCOPE = "permission_scope"
    USER_INPUT = "user_input"
    ELICITATION = "elicitation"
    UNKNOWN = "unknown"


class ModeApplication(enum.StrEnum):
    AT_LAUNCH = "at_launch"
    RESTARTED = "restarted"
    MID_SESSION_APPLIED = "mid_session_applied"
    HOOK_POLICY_APPLIED = "hook_policy_applied"
    NEXT_TURN_APPLIED = "next_turn_applied"
    REQUIRES_RESTART = "requires_restart"


@dataclasses.dataclass(frozen=True)
class ClaimEvidence:
    harness: HarnessKind
    project_path: ProjectPath
    captured_native_id: HarnessSessionId | None
    time_window: tuple[EpochMs, EpochMs]
