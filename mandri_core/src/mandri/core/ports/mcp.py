from typing import Protocol

from mandri.core.ids import HarnessKind
from mandri.core.types.mcp import McpBinding


class McpLaunchPort(Protocol):
    def bind(self, owner: str, harness: HarnessKind, *, protected: bool = False) -> McpBinding: ...

    def release(self, owner: str) -> None: ...
