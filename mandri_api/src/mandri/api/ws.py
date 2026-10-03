"""Websocket live feed endpoint and AsyncAPI document serving."""

import asyncio
import contextlib
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from mandri.api.agent_observer import AgentObserver
from mandri.api.deps import LifespanState, app_state
from mandri.api.transcript_observer import TranscriptObserver
from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import HarnessKind
from mandri.core.protocol.asyncapi import build_asyncapi
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.protocol.frames import (
    CLIENT_ADAPTER,
    SERVER_ADAPTER,
    RequestFrame,
    ResponseError,
    ResponseFrame,
    SubscribedAck,
    SubscribeFrame,
    UnsubscribedAck,
    UnsubscribeFrame,
)
from mandri.core.protocol.registry import ActionRegistry, resolve
from mandri.core.types.sessions import SessionError
from mandri.core.version import __version__
from pydantic import ValidationError
from starlette.websockets import WebSocketState

router = APIRouter(tags=["feed"])
logger = logging.getLogger(__name__)

_SESSION_TOPIC_PREFIX = "session."


def _session_of(topic: str) -> str | None:
    if not topic.startswith(_SESSION_TOPIC_PREFIX):
        return None
    return topic[len(_SESSION_TOPIC_PREFIX) :]


def _viewer_gained(state: LifespanState, session_id: str) -> None:
    count = state.viewer_counts.get(session_id, 0) + 1
    state.viewer_counts[session_id] = count
    lifetime = state.lifetime
    if count == 1 and lifetime is not None:
        lifetime.viewer_joined(session_id)
    if count == 1 and state.sessions is not None and state.hub is not None:
        if state.transcript_observer is None:
            state.transcript_observer = TranscriptObserver(
                state.sessions,
                state.hub,
                lambda sid: (
                    state.runtime is not None and state.runtime.registry.status(sid) == "live"
                ),
            )
        state.transcript_observer.join(session_id)


async def _viewer_lost(state: LifespanState, session_id: str) -> None:
    count = state.viewer_counts.get(session_id, 0)
    if count <= 1:
        state.viewer_counts.pop(session_id, None)
        lifetime = state.lifetime
        if count == 1 and lifetime is not None:
            lifetime.viewers_zero(session_id)
        if count == 1 and state.transcript_observer is not None:
            await state.transcript_observer.leave(session_id)
        return
    state.viewer_counts[session_id] = count - 1


_VIEWER_VERSION = "3.1.6"

_VIEWER_SCRIPT = (
    f"https://unpkg.com/@asyncapi/web-component@{_VIEWER_VERSION}/lib/asyncapi-web-component.js"
)

_VIEWER_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mandri live feed</title>
<script src="{_VIEWER_SCRIPT}" defer></script>
</head>
<body>
<asyncapi-web-component schemaUrl="/v1/asyncapi.json"></asyncapi-web-component>
</body>
</html>
"""


def _disconnected(websocket: WebSocket) -> bool:
    return websocket.application_state is WebSocketState.DISCONNECTED


async def _send(websocket: WebSocket, frame: dict[str, Any]) -> None:
    if _disconnected(websocket):
        return
    validated = SERVER_ADAPTER.validate_python(frame)
    try:
        await websocket.send_json(validated.model_dump())
    except WebSocketDisconnect:
        return
    except RuntimeError:
        if not _disconnected(websocket):
            raise


async def _send_error(websocket: WebSocket, detail: str) -> None:
    await _send(websocket, {"type": "error", "detail": detail})


def _to_wire(frame: dict[str, Any]) -> dict[str, Any]:
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        return frame
    base = {key: value for key, value in frame.items() if key != "payload"}
    if {"source", "raw", "ts"} <= payload.keys():
        return {**base, **payload}
    return {**base, "source": "daemon", "raw": payload, "ts": system_now_ms()}


async def _relay(handle: SubscriberHandle, websocket: WebSocket) -> None:
    for replay_frame in handle.replay:
        await _send(websocket, _to_wire(replay_frame))
    while True:
        frame = await handle.queue.get()
        if frame is None:
            return
        await _send(websocket, _to_wire(frame))


_background_tasks: set[asyncio.Task[None]] = set()


def _track(task: asyncio.Task[None]) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_done)


def _background_done(task: asyncio.Task[None]) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("websocket background operation failed", exc_info=error)


def _close_after_relay(websocket: WebSocket, task: asyncio.Task[None]) -> None:
    if task.cancelled() or task.exception() is not None:
        return
    _track(asyncio.create_task(_close(websocket)))


async def _close(websocket: WebSocket) -> None:
    if _disconnected(websocket):
        return
    try:
        await websocket.close()
    except WebSocketDisconnect:
        return
    except RuntimeError:
        if not _disconnected(websocket):
            raise


async def _snapshot(state: LifespanState) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    service = state.sessions
    if service is not None:
        if service.statuses is not None:
            await service.statuses.flush()
        rows = await service.list_sessions()
        sessions = [
            {
                "id": str(row.id),
                "harness": row.harness.value,
                "state": row.state.value,
                "title": row.effective_title,
            }
            for row in rows
        ]
    runtimes: list[dict[str, Any]] = []
    runtime = state.runtime
    if runtime is not None and service is not None:
        installed = set(runtime.installed_harnesses())
        runtimes = [
            {
                "harness": kind.value,
                "installed": kind.value in installed,
                "degraded": service.harness_state(kind).degraded,
            }
            for kind in HarnessKind
        ]
    return {
        "type": "snapshot",
        "topic": "sessions.all",
        "sessions": sessions,
        "statuses": (
            [row.model_dump() for row in service.statuses.all()]
            if service is not None and service.statuses is not None else []
        ),
        "runtimes": runtimes,
    }


async def _subscribe(
    websocket: WebSocket,
    hub: Hub,
    state: LifespanState,
    frame: SubscribeFrame,
    subscriptions: dict[str, SubscriberHandle],
    tasks: dict[str, asyncio.Task[None]],
) -> None:
    try:
        resolve(frame.topic)
    except KeyError:
        await _send_error(websocket, f"unknown topic: {frame.topic}")
        return
    if frame.topic in subscriptions:
        await _send_error(websocket, f"already subscribed: {frame.topic}")
        return
    handle = hub.subscribe(Topic(frame.topic), frame.since)
    subscriptions[frame.topic] = handle
    if state.agents is not None and (
        frame.topic == "agents.all" or frame.topic.startswith("agent.")
    ):
        if state.agent_observer is None:
            state.agent_observer = AgentObserver(state.agents, hub)
        state.agent_observer.join(frame.topic)
    session_id = _session_of(frame.topic)
    if session_id is not None:
        _viewer_gained(state, session_id)
    await _send(
        websocket,
        SubscribedAck(op="subscribed", topic=frame.topic, from_seq=handle.from_seq).model_dump(),
    )
    if frame.topic in {"sessions.all", "conversations.all"}:
        await _send(websocket, await _snapshot(state))
    elif session_id is not None and state.runtime is not None:
        state.runtime.replay_pending_approvals(session_id)
    elif (
        frame.topic.startswith("agent.") and state.agents is not None and state.runtime is not None
    ):
        with contextlib.suppress(SessionError):
            agent = await state.agents.history_store.get(frame.topic.removeprefix("agent."))
            state.runtime.replay_pending_approvals(agent.parent_session_id, Topic(frame.topic))
    task = asyncio.create_task(_relay(handle, websocket))
    tasks[frame.topic] = task
    task.add_done_callback(lambda done: _close_after_relay(websocket, done))


async def _unsubscribe(
    websocket: WebSocket,
    hub: Hub,
    state: LifespanState,
    frame: UnsubscribeFrame,
    subscriptions: dict[str, SubscriberHandle],
    tasks: dict[str, asyncio.Task[None]],
) -> None:
    handle = subscriptions.pop(frame.topic, None)
    if handle is None:
        await _send_error(websocket, f"not subscribed: {frame.topic}")
        return
    task = tasks.pop(frame.topic, None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    hub.unsubscribe(handle)
    if state.agent_observer is not None:
        await state.agent_observer.leave(frame.topic)
    session_id = _session_of(frame.topic)
    if session_id is not None:
        await _viewer_lost(state, session_id)
    await _send(websocket, UnsubscribedAck(op="unsubscribed", topic=frame.topic).model_dump())


def _actions_of(state: LifespanState) -> ActionRegistry | None:
    return state.actions


async def _handle_request(
    websocket: WebSocket,
    state: LifespanState,
    frame: RequestFrame,
) -> None:
    actions = _actions_of(state)
    if actions is None:
        await _send(
            websocket,
            ResponseFrame(
                type="response",
                op_id=frame.op_id,
                ok=False,
                error=ResponseError(
                    code=ProtocolErrorCode.INTERNAL_ERROR, message="Request actions are unavailable"
                ),
            ).model_dump(),
        )
        return
    response = await actions.handle(frame)
    await _send(websocket, response.model_dump())


async def _handle_text(
    websocket: WebSocket,
    hub: Hub,
    state: LifespanState,
    text: str,
    subscriptions: dict[str, SubscriberHandle],
    tasks: dict[str, asyncio.Task[None]],
) -> None:
    try:
        data = json.loads(text)
    except ValueError:
        await _send_error(websocket, "frame is not valid JSON")
        return
    if isinstance(data, dict) and data.get("type") == "pong":
        for handle in subscriptions.values():
            hub.pong(handle)
        return
    try:
        frame = CLIENT_ADAPTER.validate_python(data)
    except ValidationError:
        await _send_error(websocket, "frame is not a valid client op")
        return
    if isinstance(frame, SubscribeFrame):
        await _subscribe(websocket, hub, state, frame, subscriptions, tasks)
    elif isinstance(frame, UnsubscribeFrame):
        await _unsubscribe(websocket, hub, state, frame, subscriptions, tasks)
    elif isinstance(frame, RequestFrame):
        _track(asyncio.create_task(_handle_request(websocket, state, frame)))


async def _cleanup(
    hub: Hub,
    state: LifespanState,
    websocket: WebSocket,
    tasks: dict[str, asyncio.Task[None]],
    subscriptions: dict[str, SubscriberHandle],
) -> None:
    running = [tasks.pop(topic) for topic in list(tasks)]
    for task in running:
        task.cancel()
    if running:
        await asyncio.gather(*running, return_exceptions=True)
    for topic, handle in subscriptions.items():
        hub.unsubscribe(handle)
        if state.agent_observer is not None:
            await state.agent_observer.leave(topic)
        session_id = _session_of(topic)
        if session_id is not None:
            await _viewer_lost(state, session_id)
    with contextlib.suppress(Exception):
        await _close(websocket)


@router.websocket("/ws")
async def ws_feed(websocket: WebSocket) -> None:
    state = app_state(websocket.app)
    await websocket.accept(subprotocol=websocket.scope.get("mandri.subprotocol"))
    if state is None or state.hub is None:
        await websocket.close(code=1011)
        return
    hub = state.hub
    if not state.heartbeat_configured:
        hub.set_heartbeat_policy(state.heartbeat_idle_seconds, state.heartbeat_max_misses)
        state.heartbeat_configured = True
    subscriptions: dict[str, SubscriberHandle] = {}
    tasks: dict[str, asyncio.Task[None]] = {}
    try:
        while not _disconnected(websocket):
            try:
                text = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            await _handle_text(websocket, hub, state, text, subscriptions, tasks)
    finally:
        _track(asyncio.create_task(_cleanup(hub, state, websocket, tasks, subscriptions)))


@router.get("/asyncapi.json", operation_id="serve_asyncapi")
async def serve_asyncapi() -> JSONResponse:
    return JSONResponse(build_asyncapi(__version__))


@router.get("/asyncapi", operation_id="serve_asyncapi_viewer")
async def serve_asyncapi_viewer() -> HTMLResponse:
    return HTMLResponse(_VIEWER_HTML)
