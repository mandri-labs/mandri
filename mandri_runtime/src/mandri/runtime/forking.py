from pathlib import Path
from typing import Protocol

from mandri.core.types.sessions import Session


class CodexForkSourcePort(Protocol):
    @property
    def session(self) -> Session: ...

    @property
    def native_id(self) -> str: ...

    @property
    def rollout_path(self) -> Path: ...

    @property
    def workspace_root(self) -> Path: ...

    def copy_rollout(self, destination: Path) -> None: ...
