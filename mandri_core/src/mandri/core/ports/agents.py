from dataclasses import dataclass
from typing import Protocol

from mandri.core.ids import HarnessKind
from mandri.core.types.agents import NativeAgent
from mandri.core.types.sessions import Session


@dataclass(frozen=True)
class AgentDiscoveryResult:
    harness: HarnessKind
    agents: list[NativeAgent]
    classified_native_ids: frozenset[str] | None = None
    complete: bool = True


class AgentDiscovery(Protocol):
    def discover(self, sessions: list[Session]) -> AgentDiscoveryResult: ...
