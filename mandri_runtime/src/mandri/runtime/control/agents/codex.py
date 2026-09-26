from typing import Any

from mandri.core.types.agents import Agent, AgentCapabilities
from mandri.runtime.control.agents.base import RpcCall, UnsupportedAgentOperation
from mandri.runtime.control.errors import ControlTransportError


class CodexAgentControl:
    can_create = False

    def __init__(self, call: RpcCall) -> None:
        self._call = call

    def capabilities(self, agent: Agent) -> AgentCapabilities:
        return AgentCapabilities(message=True, stop=True)

    async def create(self, content: str, title: str | None) -> dict[str, Any]:
        raise UnsupportedAgentOperation("Codex does not expose direct child thread creation")

    async def message(self, agent: Agent, content: str) -> None:
        result = self._result(
            await self._call("thread/read", {"threadId": agent.native_id, "includeTurns": True})
        )
        _verify_thread(result, agent.native_id)
        turn = _active_turn(result)
        method = "turn/steer" if turn else "turn/start"
        parameters: dict[str, Any] = {
            "threadId": agent.native_id,
            "input": [{"type": "text", "text": content, "text_elements": []}],
        }
        if turn:
            parameters["expectedTurnId"] = turn
        self._result(await self._call(method, parameters))

    async def stop(self, agent: Agent) -> bool:
        result = self._result(
            await self._call("thread/read", {"threadId": agent.native_id, "includeTurns": True})
        )
        _verify_thread(result, agent.native_id)
        turn = _active_turn(result)
        if turn is None:
            return False
        self._result(
            await self._call("turn/interrupt", {"threadId": agent.native_id, "turnId": turn})
        )
        return True

    @staticmethod
    def _result(message: dict[str, Any]) -> dict[str, Any]:
        if "error" in message:
            raise ControlTransportError("Codex rejected the agent operation")
        result = message.get("result")
        return result if isinstance(result, dict) else {}


def _verify_thread(result: dict[str, Any], native_id: str) -> None:
    thread = result.get("thread")
    if not isinstance(thread, dict) or thread.get("id") != native_id:
        raise ControlTransportError("Codex returned a different child thread")


def _active_turn(result: dict[str, Any]) -> str | None:
    thread = result.get("thread")
    turns = thread.get("turns") if isinstance(thread, dict) else None
    if not isinstance(turns, list):
        return None
    for turn in reversed(turns):
        if isinstance(turn, dict) and turn.get("status") == "inProgress":
            turn_id = turn.get("id")
            return turn_id if isinstance(turn_id, str) else None
    return None
