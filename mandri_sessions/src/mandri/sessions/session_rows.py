import json
import typing

from mandri.core.fs.paths import normalize_fs_path
from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    RouteId,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import InteractionMode, Session
from mandri.core.types.worktrees import Worktree


def _interaction_mode_of(raw: object) -> InteractionMode | None:
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode")
    applied = payload.get("applied")
    if not isinstance(mode, str) or not isinstance(applied, str):
        return None
    return InteractionMode(mode=mode, applied=applied)


def row_to_session(row: dict[str, object]) -> Session:
    native_id = row["native_id"]
    native_title = row["native_title"]
    title_overlay = row["title_overlay"]
    model = row["model"]
    gateway_route_id = row["gateway_route_id"]
    reasoning_effort = row.get("reasoning_effort")
    return Session(
        id=SessionId(str(row["id"])),
        harness=HarnessKind(str(row["harness"])),
        native_id=None if native_id is None else HarnessSessionId(str(native_id)),
        native_title=None if native_title is None else SessionTitle(str(native_title)),
        title_overlay=None if title_overlay is None else SessionTitle(str(title_overlay)),
        project_path=ProjectPath(normalize_fs_path(str(row["project_path"]))),
        created_at=EpochMs(typing.cast(int, row["created_at"])),
        updated_at=EpochMs(typing.cast(int, row["updated_at"])),
        state=SessionState(str(row["state"])),
        model=None if model is None else str(model),
        gateway_route_id=(None if gateway_route_id is None else RouteId(str(gateway_route_id))),
        deleted=bool(row["deleted"]),
        last_synced_at=EpochMs(typing.cast(int, row["last_synced_at"])),
        interaction_mode=_interaction_mode_of(row["interaction_mode"]),
        reasoning_effort=None if reasoning_effort is None else str(reasoning_effort),
        model_source=ModelSource(str(row.get("model_source", "gateway"))),
        execution_backend=ExecutionBackend(str(row.get("execution_backend", "host"))),
        privacy_mode=PrivacyMode(str(row.get("privacy_mode", "none"))),
        privacy_scope_id=str(row["privacy_scope_id"]) if row.get("privacy_scope_id") else None,
        execution_context=str(row["execution_context"]) if row.get("execution_context") else None,
        worktree=Worktree(**json.loads(str(row["worktree"]))) if row.get("worktree") else None,
        policy_revision=int(str(row.get("policy_revision", 1))),
        parent_native_id=HarnessSessionId(str(row["parent_native_id"]))
        if row.get("parent_native_id")
        else None,
        parent_session_id=SessionId(str(row["parent_session_id"]))
        if row.get("parent_session_id")
        else None,
    )
