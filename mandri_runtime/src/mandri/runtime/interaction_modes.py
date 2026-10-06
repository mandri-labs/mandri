from typing import Any

from mandri.config.types import CLAUDE_MODES


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class InteractionModes:
    def __init__(self) -> None:
        self.mode: str | None = None
        self.bypass_available = False
        self.auto_available = False
        self._tools: dict[str, str] = {}
        self._exits: dict[str, str] = {}

    def confirm(self, mode: str) -> None:
        self.mode = mode
        self.bypass_available |= mode == "bypassPermissions"
        self.auto_available |= mode == "auto"

    def plan_modes(self) -> list[str]:
        return [
            "default",
            "acceptEdits",
            *(["auto"] if self.auto_available else []),
            *(["bypassPermissions"] if self.bypass_available else []),
        ]

    def approve_plan(self, raw: dict[str, Any], mode: str) -> None:
        identifier = _record(raw.get("request")).get("tool_use_id")
        if isinstance(identifier, str):
            self._exits[identifier] = mode

    def forget_plan(self, raw: dict[str, Any]) -> None:
        identifier = _record(raw.get("request")).get("tool_use_id")
        if isinstance(identifier, str):
            self._exits.pop(identifier, None)

    def observe(
        self, harness: str, raw: dict[str, Any], native_id: str | None = None
    ) -> str | None:
        mode = None
        if harness == "claude":
            mode = self._claude(raw)
        elif harness == "pi" and raw.get("type") == "extension_ui_request":
            if raw.get("method") == "setStatus" and raw.get("statusKey") == "_mandri_permissions":
                status = raw.get("statusText")
                candidate = status.rsplit(":", 1)[-1] if isinstance(status, str) else None
                if candidate in {"default", "acceptEdits", "plan", "bypassPermissions"}:
                    mode = candidate
        elif harness == "opencode" and raw.get("type") == "session.updated":
            info = _record(_record(raw.get("properties")).get("info"))
            if native_id is not None and info.get("id") != native_id:
                return None
            permissions = info.get("permissions", info.get("permission"))
            if permissions == [
                {"action": "*", "resource": "*", "effect": "allow"}
            ] or permissions == {"*": "allow"}:
                mode = "auto"
            elif permissions == [
                {"action": "*", "resource": "*", "effect": "ask"}
            ] or permissions == {"*": "ask"}:
                mode = "default"
        elif harness == "codex":
            result = raw.get("result")
            if isinstance(result, dict) and isinstance(result.get("thread"), dict):
                if native_id is not None and result["thread"].get("id") != native_id:
                    return None
                policy = result.get("approvalPolicy")
                sandbox = _record(result.get("sandbox"))
                reviewer = result.get("approvalsReviewer")
                if policy == "never" and sandbox.get("type") == "dangerFullAccess":
                    mode = "full-access"
                elif policy == "on-request" and sandbox.get("type") == "workspaceWrite":
                    if reviewer == "auto_review":
                        mode = "auto"
                    elif reviewer == "user":
                        mode = "ask"
        if mode is None or mode == self.mode:
            return None
        self.confirm(mode)
        return mode

    def _claude(self, raw: dict[str, Any]) -> str | None:
        if raw.get("parent_tool_use_id") is not None:
            return None
        kind = raw.get("type")
        mode = raw.get("permissionMode")
        explicit = (
            mode
            if kind in {"system", "user"} and isinstance(mode, str) and mode in CLAUDE_MODES
            else None
        )
        message = raw.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if kind not in {"assistant", "user"} or not isinstance(content, list):
            return explicit
        confirmed = None
        for item in content:
            if not isinstance(item, dict):
                continue
            identifier = item.get("id")
            if (
                kind == "assistant"
                and item.get("type") == "tool_use"
                and item.get("name") == "EnterPlanMode"
                and isinstance(identifier, str)
            ):
                self._tools[identifier] = "plan"
            if kind == "user" and item.get("type") == "tool_result":
                identifier = item.get("tool_use_id")
                if not isinstance(identifier, str):
                    continue
                expected = self._exits.pop(identifier, None) or self._tools.pop(identifier, None)
                if expected is not None and item.get("is_error") is not True:
                    confirmed = expected
        if len(self._tools) > 32:
            self._tools = dict(list(self._tools.items())[-32:])
        return explicit or confirmed
