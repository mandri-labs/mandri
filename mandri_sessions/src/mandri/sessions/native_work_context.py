from dataclasses import replace
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.types.conversation_status import WorkObservation, WorkState
from mandri.core.work_content import work_content_key
from mandri.core.work_outcomes import work_event_key
from mandri.sessions.native_codex_work import codex_children
from mandri.sessions.work_observation import native_work_observations


class NativeWorkContext:
    def __init__(self, harness: HarnessKind, native_id: str, checkpoint: dict[str, Any]) -> None:
        self._harness, self._native_id = harness, native_id
        self._tasks = set(checkpoint.get("tasks", []))
        self._children = set(checkpoint.get("children", []))
        self._root_idle = bool(checkpoint.get("root_idle"))
        self._completed_children = set(checkpoint.get("completed_children", []))
        self._calls = dict(checkpoint.get("calls", {}))
        self._uncertain = bool(checkpoint.get("uncertain"))
        self._content = checkpoint.get("content_key")
        self._fresh_content = False

    def checkpoint(self) -> dict[str, Any]:
        return {
            "tasks": sorted(self._tasks),
            "children": sorted(self._children),
            "root_idle": self._root_idle,
            "calls": self._calls,
            "completed_children": sorted(self._completed_children),
            "uncertain": self._uncertain,
            "content_key": self._content,
        }

    def mark_uncertain(self) -> None:
        self._uncertain = True

    def observe(self, raw: dict[str, Any]) -> tuple[WorkObservation, ...]:
        content = work_content_key(self._harness, raw)
        if content is not None and content != self._content:
            self._content = content
            self._fresh_content = True
        observations = self._reduce(raw)
        return tuple(
            replace(
                item,
                source="native",
                state="unknown" if self._uncertain and item.state == "idle" else item.state,
                content_key=(item.content_key or self._content)
                if item.outcome
                else item.content_key,
            )
            for item in observations
        )

    def _reduce(self, raw: dict[str, Any]) -> tuple[WorkObservation, ...]:
        observations = native_work_observations(self._harness, raw, self._native_id)
        if self._harness is HarnessKind.CODEX:
            changed, uncertain = codex_children(raw, self._children, self._calls)
            self._uncertain |= uncertain
            if any(item.progress for item in observations):
                self._root_idle = False
            if any(item.outcome for item in observations):
                self._root_idle = True
            if self._root_idle and self._children:
                return tuple(replace(item, state="working") for item in observations) or (
                    WorkObservation(state="working", progress=True, key=work_event_key(raw)),
                )
            if changed and self._root_idle:
                return (WorkObservation(state="idle", key=work_event_key(raw)),)
            return observations
        if self._harness is HarnessKind.AGY:
            return self._observe_agy(raw, observations)
        if self._harness is not HarnessKind.CLAUDE or raw.get("parent_tool_use_id"):
            return observations
        key = work_event_key(raw)
        kind = raw.get("type")
        if kind == "system":
            subtype, task = raw.get("subtype"), raw.get("task_id")
            if subtype == "task_started" and isinstance(task, str):
                self._tasks.add(task)
                return (WorkObservation(state="working", progress=True, key=key),)
            if (
                subtype in {"task_notification", "task_updated"}
                and raw.get("status")
                in {
                    "completed",
                    "failed",
                    "stopped",
                    "killed",
                }
                and isinstance(task, str)
            ):
                self._tasks.discard(task)
                state: WorkState = "idle" if self._root_idle and not self._tasks else "working"
                return (WorkObservation(state=state, key=key),)
        if kind in {"user", "assistant", "stream_event"}:
            self._root_idle = False
        if kind == "result":
            queued = raw.get("queued_turn_count")
            self._root_idle = not (isinstance(queued, int) and queued > 0)
            if self._tasks or not self._root_idle:
                return tuple(
                    WorkObservation(
                        state="working", progress=item.progress, outcome=item.outcome, key=item.key
                    )
                    for item in observations
                )
        return observations

    def _observe_agy(
        self, raw: dict[str, Any], observations: tuple[WorkObservation, ...]
    ) -> tuple[WorkObservation, ...]:
        data = raw.get("data", raw.get("step_update", raw))
        if not isinstance(data, dict):
            return observations
        owner = data.get("conversationId", data.get("conversation_id"))
        hook = raw.get("hook") if raw.get("event") == "hook" else raw.get("event")
        key = work_event_key(raw)
        if owner is not None and owner != self._native_id:
            if (hook == "PreInvocation" or data.get("state") == "ACTIVE") and (
                owner in self._children or owner in self._completed_children
            ):
                self._completed_children.discard(owner)
                self._children.add(owner)
                return (WorkObservation(state="working", progress=True, key=key),)
            if owner not in self._children:
                return ()
            if hook == "Stop" and data.get("fullyIdle") is True:
                self._children.discard(owner)
                self._completed_children.add(owner)
                state: WorkState = "idle" if self._root_idle and not self._children else "working"
                return (WorkObservation(state=state, key=key),)
            return ()
        info = data.get("subagent_info")
        children = info.get("subagents") if isinstance(info, dict) else None
        if isinstance(children, list):
            self._children.update(
                child["conversation_id"]
                for child in children
                if isinstance(child, dict)
                and isinstance(child.get("conversation_id"), str)
                and child["conversation_id"] != self._native_id
                and child["conversation_id"] not in self._completed_children
            )
        if hook == "Stop" and isinstance(data.get("fullyIdle"), bool):
            self._root_idle = data["fullyIdle"]
            if not self._root_idle or self._children:
                return tuple(
                    WorkObservation(
                        state="working",
                        progress=item.progress or not self._root_idle,
                        outcome=item.outcome,
                        key=item.key,
                    )
                    for item in observations
                ) or (WorkObservation(state="working", progress=True, key=key),)
        if observations and any(item.progress for item in observations):
            self._root_idle = False
        return observations
