from typing import Any
from urllib.parse import quote

import httpx
from mandri.core.opencode import gateway_model
from mandri.core.types.agents import Agent, AgentCapabilities
from mandri.runtime.control.agents.base import HttpCall
from mandri.runtime.control.errors import ControlTransportError


class OpencodeAgentControl:
    can_create = True

    def __init__(self, call: HttpCall, parent_native_id: str) -> None:
        self._call = call
        self._parent = parent_native_id

    def capabilities(self, agent: Agent) -> AgentCapabilities:
        return AgentCapabilities(message=True, stop=True)

    async def create(self, content: str, title: str | None) -> dict[str, Any]:
        response = await self._call("GET", f"/api/session/{quote(self._parent, safe='')}", None)
        self._check(response)
        parent = self._object(response)
        if parent.get("id") != self._parent:
            raise ControlTransportError("OpenCode returned a different parent session")
        body: dict[str, Any] = {
            "parentID": self._parent,
            "model": gateway_model(),
        }
        for field in ("permissions", "agent"):
            if parent.get(field) is not None:
                body[field] = parent[field]
        if title:
            body["title"] = title
        response = await self._call("POST", "/api/session", body)
        self._check(response)
        child = self._object(response)
        native_id = child.get("id")
        if not isinstance(native_id, str) or not native_id or child.get("parentID") != self._parent:
            raise ControlTransportError("OpenCode did not return the requested child relationship")
        response = await self._call(
            "POST",
            f"/api/session/{quote(native_id, safe='')}/prompt",
            {"text": content},
        )
        self._check(response)
        return child

    async def message(self, agent: Agent, content: str) -> None:
        response = await self._call(
            "POST",
            f"/api/session/{quote(agent.native_id, safe='')}/model",
            {"model": gateway_model()},
        )
        self._check(response)
        response = await self._call(
            "POST",
            f"/api/session/{quote(agent.native_id, safe='')}/prompt",
            {"text": content},
        )
        self._check(response)

    async def stop(self, agent: Agent) -> bool:
        response = await self._call(
            "POST", f"/api/session/{quote(agent.native_id, safe='')}/interrupt", None
        )
        self._check(response)
        try:
            result = response.json()["interrupted"]
        except (ValueError, KeyError, TypeError):
            raise ControlTransportError("OpenCode returned an invalid stop result") from None
        if not isinstance(result, bool):
            raise ControlTransportError("OpenCode returned an invalid stop result")
        return result

    @staticmethod
    def _object(response: httpx.Response) -> dict[str, Any]:
        try:
            result = response.json()["data"]
        except (ValueError, KeyError, TypeError):
            raise ControlTransportError("OpenCode returned an invalid child session") from None
        if not isinstance(result, dict):
            raise ControlTransportError("OpenCode returned an invalid child session")
        return result

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if not 200 <= response.status_code < 300:
            raise ControlTransportError(
                f"OpenCode rejected the agent operation ({response.status_code})"
            )
