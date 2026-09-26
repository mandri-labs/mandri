import dataclasses
import json
import logging
import time
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.errors import MandriError
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.protocol.agents import AgentView
from mandri.core.types.agents import Agent, AgentState
from mandri.core.types.approvals import ApprovalRequest
from mandri.core.types.sessions import Session
from mandri.runtime.approvals.recognition import detect
from mandri.sessions.agents.codex import parent_of
from mandri.sessions.agents.relationships import agent_id
from mandri.sessions.agents.service import AgentHistory
from mandri.sessions.service import SessionsService

logger = logging.getLogger(__name__)


class AgentEventRouter:
    def __init__(self, hub: Hub, history: AgentHistory, sessions: SessionsService) -> None:
        self._hub = hub
        self._history = history
        self._sessions = sessions
        self._parents: dict[str, Session] = {}
        self._parent_expiry: dict[str, float] = {}
        self._agents: dict[str, dict[str, Agent]] = {}

    async def publish(self, topic: Topic, payload: dict[str, Any]) -> None:
        parent_id = str(topic).removeprefix("session.")
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            self._hub.publish(topic, payload)
            return
        parent = None
        try:
            parent = await self._parent(parent_id)
            if (
                parent.native_id is None
                and (
                    (parent.harness is HarnessKind.AGY and raw.get("event") == "init")
                    or (
                        parent.harness is HarnessKind.CODEX
                        and raw.get("method") == "mcpServer/startupStatus/updated"
                        and "id" not in raw
                    )
                )
            ):
                parent = await self._parent(parent_id, force=True)
                if parent.native_id is None:
                    self._hub.publish(topic, payload)
                    return
            native_id = event_native_id(parent.harness, raw)
            if (
                native_id is not None
                and native_id != parent.native_id
                and native_id not in self._agents.get(parent_id, {})
            ):
                parent = await self._parent(parent_id, force=True)
            child = await self._identify(parent, raw)
            if child is not None:
                updated = _updated(child, raw)
                if updated != child:
                    await self._save(updated)
                self._hub.publish(Topic(f"agent.{child.id}"), payload)
                if (
                    (parent.harness is HarnessKind.CLAUDE and raw.get("type") == "system")
                    or (
                        parent.harness is HarnessKind.AGY
                        and (
                            (
                                raw.get("event") == "hook"
                                and raw.get("hook") in {"PreInvocation", "Stop"}
                            )
                            or raw.get("event") == "approval_response"
                        )
                    )
                    or detect(parent.harness, json.dumps(raw)) is not None
                ):
                    self._hub.publish(topic, {**payload, "agent_id": child.id})
                return
        except (MandriError, OSError, ValueError):
            logger.exception("Agent event routing unavailable for session %s", parent_id)
        if parent is not None:
            native_id = event_native_id(parent.harness, raw)
            if (native_id is not None and native_id != parent.native_id) or (
                parent.harness is HarnessKind.CLAUDE and raw.get("parent_tool_use_id")
            ):
                return
        self._hub.publish(topic, payload)

    async def _parent(self, parent_id: str, *, force: bool = False) -> Session:
        if not force and time.monotonic() < self._parent_expiry.get(parent_id, 0):
            return self._parents[parent_id]
        parent = await self._sessions.get_session(SessionId(parent_id))
        self._parents[parent_id] = parent
        self._parent_expiry[parent_id] = time.monotonic() + 1
        return parent

    def approval_topic(self, request: ApprovalRequest) -> Topic | None:
        try:
            raw = json.loads(request.native_request)
        except ValueError:
            return None
        if not isinstance(raw, dict):
            return None
        known = self._agents.get(str(request.session_id), {})
        native_id = event_native_id(request.harness, raw)
        delegate = _delegation_id(raw)
        child = known.get(native_id) if native_id else None
        if child is None and delegate:
            child = next(
                (agent for agent in known.values() if agent.delegation_id == delegate), None
            )
        return Topic(f"agent.{child.id}") if child else None

    async def _identify(self, parent: Session, raw: dict[str, Any]) -> Agent | None:
        parent_id = str(parent.id)
        known = self._agents.setdefault(parent_id, {})
        native_id = event_native_id(parent.harness, raw)
        delegate = _delegation_id(raw)
        if native_id is not None and native_id == parent.native_id:
            return None
        if native_id is None and not delegate:
            return None
        if native_id in known:
            return known[native_id]
        if delegate:
            match = next(
                (agent for agent in known.values() if agent.delegation_id == delegate), None
            )
            if match is not None:
                return match
        announced = _announced(parent, raw)
        if announced is not None and (
            announced.parent_agent_id is None
            or any(agent.id == announced.parent_agent_id for agent in known.values())
        ):
            await self._save(announced)
            return announced
        await self._history.refresh(force=True)
        known.update({agent.native_id: agent for agent in await self._history.list(parent_id)})
        if native_id in known:
            return known[native_id]
        return next(
            (agent for agent in known.values() if delegate and agent.delegation_id == delegate),
            None,
        )

    async def _save(self, agent: Agent) -> None:
        await self._history.save(agent)
        self._agents.setdefault(agent.parent_session_id, {})[agent.native_id] = agent
        self._hub.publish(
            Topic("agents.all"),
            {
                "source": "mandri",
                "ts": int(system_now_ms()),
                "raw": {
                    "type": "agent_changed",
                    "agent": AgentView.model_validate(agent).model_dump(mode="json"),
                },
            },
        )


def event_native_id(harness: HarnessKind, raw: dict[str, Any]) -> str | None:
    if harness is HarnessKind.CLAUDE:
        request = raw.get("request")
        nested = request if isinstance(request, dict) else {}
        value = raw.get("agent_id") or nested.get("agent_id")
    elif harness is HarnessKind.CODEX:
        params = raw.get("params")
        if not isinstance(params, dict):
            return None
        thread = params.get("thread")
        value = params.get("threadId") or (thread.get("id") if isinstance(thread, dict) else None)
    elif harness is HarnessKind.AGY:
        kind = raw.get("event")
        agy_nested = (
            raw.get(str(kind)) if kind in ("init", "step_update", "result") else raw.get("data")
        )
        if isinstance(agy_nested, dict):
            value = agy_nested.get("conversation_id") or agy_nested.get("conversationId")
        else:
            value = None
        value = value or raw.get("conversation_id") or raw.get("conversationId")
    else:
        properties = raw.get("properties")
        if not isinstance(properties, dict):
            return None
        info = properties.get("info") or properties.get("part") or properties
        value = info.get("sessionID") if isinstance(info, dict) else None
        if value is None and raw.get("type") in ("session.created", "session.updated"):
            value = info.get("id") if isinstance(info, dict) else None
    return value if isinstance(value, str) and value else None


def _delegation_id(raw: dict[str, Any]) -> str | None:
    request = raw.get("request")
    nested = request if isinstance(request, dict) else {}
    value = (
        raw.get("parent_tool_use_id") or raw.get("tool_use_id") or nested.get("parent_tool_use_id")
    )
    return value if isinstance(value, str) else None


def _announced(parent: Session, raw: dict[str, Any]) -> Agent | None:
    metadata: dict[str, Any] | None = None
    native_parent = None
    if parent.harness is HarnessKind.CODEX and raw.get("method") == "thread/started":
        params = raw.get("params")
        metadata = params.get("thread") if isinstance(params, dict) else None
        if isinstance(metadata, dict):
            native_parent = metadata.get("parentThreadId") or parent_of(metadata.get("source"))
    if parent.harness is HarnessKind.OPENCODE and raw.get("type") == "session.created":
        properties = raw.get("properties")
        metadata = properties.get("info") if isinstance(properties, dict) else None
        if isinstance(metadata, dict):
            native_parent = metadata.get("parentID")
    if not isinstance(metadata, dict) or not isinstance(native_parent, str) or not native_parent:
        return None
    native_id = metadata.get("id")
    if not isinstance(native_id, str) or native_id == native_parent:
        return None
    now = int(system_now_ms())
    root_id = str(parent.id)
    return Agent(
        agent_id(root_id, native_id),
        root_id,
        parent.harness,
        native_id,
        str(metadata.get("agentNickname") or metadata.get("title") or "Agent"),
        AgentState.UNKNOWN,
        now,
        now,
        parent_agent_id=agent_id(root_id, native_parent)
        if native_parent != parent.native_id
        else None,
    )


def _updated(agent: Agent, raw: dict[str, Any]) -> Agent:
    state = agent.state
    if agent.harness is HarnessKind.AGY:
        state = _agy_state(state, raw)
    method = raw.get("method") or raw.get("type")
    if method in ("turn/started", "assistant", "user", "message.part.updated"):
        state = AgentState.RUNNING
    if method == "turn/completed":
        params = raw.get("params", {})
        turn = params.get("turn", {}) if isinstance(params, dict) else {}
        status = turn.get("status") if isinstance(turn, dict) else None
        state = (
            AgentState.FAILED
            if status == "failed"
            else AgentState.STOPPED
            if status == "interrupted"
            else AgentState.COMPLETED
        )
    if method == "session.idle" or method == "result":
        state = AgentState.FAILED if raw.get("is_error") else AgentState.COMPLETED
    if method in ("session.error", "error"):
        state = AgentState.FAILED
    if raw.get("type") == "system":
        status = raw.get("status")
        if raw.get("subtype") == "task_started":
            state = AgentState.RUNNING
        elif status in ("completed", "failed", "stopped"):
            state = AgentState(status)
    if "requestApproval" in str(method) or method in ("permission.asked", "permission.v2.asked"):
        state = AgentState.WAITING
    task_id = raw.get("task_id") if agent.harness is HarnessKind.CLAUDE else None
    changes: dict[str, Any] = {"state": state}
    if isinstance(task_id, str):
        changes["task_id"] = task_id
    candidate = dataclasses.replace(agent, **changes)
    return (
        dataclasses.replace(candidate, updated_at=int(system_now_ms()))
        if candidate != agent
        else agent
    )


def _agy_state(state: AgentState, raw: dict[str, Any]) -> AgentState:
    event = raw.get("event")
    if event == "approval_request":
        return AgentState.WAITING
    if event == "approval_response":
        return AgentState.RUNNING
    if event == "step_update":
        step = raw.get("step_update")
        if isinstance(step, dict) and (
            step.get("state") in {"ACTIVE", "RUNNING"} or step.get("step_type") == "user_input"
        ):
            return AgentState.RUNNING
    if event == "hook":
        if raw.get("hook") == "PreInvocation":
            return AgentState.RUNNING
        data = raw.get("data")
        if raw.get("hook") == "Stop" and isinstance(data, dict):
            if data.get("error"):
                return AgentState.FAILED
            if data.get("fullyIdle") is True:
                return AgentState.COMPLETED
            if data.get("fullyIdle") is False:
                return AgentState.RUNNING
    if event == "result":
        result = raw.get("result")
        if isinstance(result, dict) and result.get("status") == "ERROR":
            return AgentState.FAILED
    return state
