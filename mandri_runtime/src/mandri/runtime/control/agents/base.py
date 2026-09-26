from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx
from mandri.core.types.agents import Agent, AgentCapabilities
from mandri.runtime.control.errors import ControlTransportError

RpcCall = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
HttpCall = Callable[[str, str, dict[str, Any] | None], Awaitable[httpx.Response]]
TaskStop = Callable[[str], Awaitable[bool]]


class AgentControl(Protocol):
    can_create: bool

    def capabilities(self, agent: Agent) -> AgentCapabilities: ...

    async def create(self, content: str, title: str | None) -> dict[str, Any]: ...

    async def message(self, agent: Agent, content: str) -> None: ...

    async def stop(self, agent: Agent) -> bool: ...


class UnsupportedAgentOperation(ControlTransportError):
    pass
