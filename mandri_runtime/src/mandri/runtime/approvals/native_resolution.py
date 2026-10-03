import json
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.codex_request_ids import request_reference


def matches_resolution(request: ApprovalRequest, event: dict[str, Any]) -> bool:
    try:
        original = json.loads(request.native_request)
    except (ValueError, TypeError):
        return False
    if not isinstance(original, dict):
        return False
    if request.harness is HarnessKind.CODEX:
        if event.get("method") != "serverRequest/resolved":
            return False
        params = event.get("params")
        initial = original.get("params")
        if not isinstance(params, dict) or not isinstance(initial, dict):
            return False
        owner = params.get("threadId")
        return (
            isinstance(owner, str)
            and bool(owner)
            and owner == initial.get("threadId")
            and request_reference(params.get("requestId")) == request.native_request_ref
        )
    if request.harness is HarnessKind.OPENCODE:
        permission = original.get("type") in {"permission.asked", "permission.v2.asked"}
        expected = (
            {"permission.replied"} if permission else {"question.replied", "question.rejected"}
        )
        if event.get("type") not in expected or (
            not permission and original.get("type") != "question.asked"
        ):
            return False
        properties = event.get("properties")
        initial = original.get("properties")
        if not isinstance(properties, dict) or not isinstance(initial, dict):
            return False
        owner = properties.get("sessionID")
        return (
            isinstance(owner, str)
            and bool(owner)
            and owner == initial.get("sessionID")
            and properties.get("requestID") == request.native_request_ref
        )
    return False
