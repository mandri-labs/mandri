"""Predicates recognizing approval-bearing harness events and envelope attachment."""

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, final

from mandri.core.ids import ApprovalKind, HarnessKind
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.codex_request_ids import request_reference

_CLAUDE_PROMPT_SUBTYPE = "can_use_tool"

_CODEX_METHOD_KINDS: dict[str, ApprovalKind] = {
    "item/commandExecution/requestApproval": ApprovalKind.COMMAND_EXECUTION,
    "item/fileChange/requestApproval": ApprovalKind.FILE_CHANGE,
    "item/permissions/requestApproval": ApprovalKind.PERMISSION_SCOPE,
    "item/tool/requestUserInput": ApprovalKind.USER_INPUT,
    "mcpServer/elicitation/request": ApprovalKind.ELICITATION,
}

_OPENCODE_ASK_TYPES = frozenset({"permission.asked", "permission.v2.asked", "question.asked"})

_TOOL_KINDS: dict[str, ApprovalKind] = {
    "bash": ApprovalKind.COMMAND_EXECUTION,
    "edit": ApprovalKind.FILE_CHANGE,
    "write": ApprovalKind.FILE_CHANGE,
    "multiedit": ApprovalKind.FILE_CHANGE,
    "notebookedit": ApprovalKind.FILE_CHANGE,
    "askuserquestion": ApprovalKind.USER_INPUT,
}

_DEFAULT_TOOL_KIND = ApprovalKind.PERMISSION_SCOPE


@final
@dataclasses.dataclass(frozen=True)
class ApprovalDescriptor:
    kind: ApprovalKind
    native_request_ref: str


def detect(harness: HarnessKind, raw_line: str) -> ApprovalDescriptor | None:
    event = _parse_json(raw_line)
    if not isinstance(event, dict):
        return None
    match harness:
        case HarnessKind.PI:
            identifier = event.get("id")
            method = event.get("method")
            if (
                event.get("type") != "extension_ui_request"
                or method not in {"select", "confirm", "input", "editor"}
                or not isinstance(identifier, str)
                or not identifier
            ):
                return None
            return ApprovalDescriptor(
                kind=ApprovalKind.PERMISSION_SCOPE
                if method == "confirm"
                else ApprovalKind.USER_INPUT,
                native_request_ref=identifier,
            )
        case HarnessKind.AGY:
            if event.get("event") != "approval_request" or not isinstance(
                event.get("request_id"), str
            ):
                return None
            tool = str(event.get("tool_name", ""))
            kind = {
                "run_command": ApprovalKind.COMMAND_EXECUTION,
                "ask_question": ApprovalKind.USER_INPUT,
                "write_to_file": ApprovalKind.FILE_CHANGE,
                "replace_file_content": ApprovalKind.FILE_CHANGE,
            }.get(tool, ApprovalKind.PERMISSION_SCOPE)
            return ApprovalDescriptor(kind=kind, native_request_ref=event["request_id"])
        case HarnessKind.CLAUDE:
            return _detect_claude(event)
        case HarnessKind.CODEX:
            return _detect_codex(event)
        case HarnessKind.OPENCODE:
            return _detect_opencode(event)
        case _:
            return None


def with_approval_metadata(
    envelope: Mapping[str, Any],
    request: ApprovalRequest,
) -> dict[str, Any]:
    return {
        **envelope,
        "approval_id": request.id,
        "kind": request.kind.value,
        "deadline": request.deadline,
        "status": request.status.value,
    }


def _detect_claude(event: dict[str, Any]) -> ApprovalDescriptor | None:
    if event.get("type") != "control_request":
        return None
    request = event.get("request")
    nested = request if isinstance(request, dict) else {}
    subtypes = {nested.get("subtype"), event.get("subtype")}
    if _CLAUDE_PROMPT_SUBTYPE not in subtypes:
        return None
    ref = event.get("request_id") or nested.get("request_id")
    if not isinstance(ref, str):
        return None
    tool = nested.get("tool_name") or event.get("tool_name")
    return ApprovalDescriptor(kind=_kind_from_tool(tool), native_request_ref=ref)


def _detect_codex(event: dict[str, Any]) -> ApprovalDescriptor | None:
    method = event.get("method")
    if not isinstance(method, str):
        return None
    kind = _CODEX_METHOD_KINDS.get(method)
    if kind is None:
        return None
    reference = request_reference(event.get("id"))
    if reference is None:
        return None
    return ApprovalDescriptor(kind=kind, native_request_ref=reference)


def _detect_opencode(event: dict[str, Any]) -> ApprovalDescriptor | None:
    if event.get("type") not in _OPENCODE_ASK_TYPES:
        return None
    properties = event.get("properties")
    nested = properties if isinstance(properties, dict) else {}
    ref = nested.get("id") or event.get("id")
    if not isinstance(ref, str):
        return None
    if event.get("type") == "question.asked":
        return ApprovalDescriptor(kind=ApprovalKind.USER_INPUT, native_request_ref=ref)
    tool = nested.get("tool") or event.get("tool")
    return ApprovalDescriptor(kind=_kind_from_tool(tool), native_request_ref=ref)


def _kind_from_tool(tool: Any) -> ApprovalKind:
    if not isinstance(tool, str):
        return _DEFAULT_TOOL_KIND
    return _TOOL_KINDS.get(tool.lower(), _DEFAULT_TOOL_KIND)


def _parse_json(raw_line: str) -> Any | None:
    try:
        return json.loads(raw_line)
    except json.JSONDecodeError:
        return None
