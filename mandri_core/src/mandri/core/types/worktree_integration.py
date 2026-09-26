from dataclasses import dataclass, field
from typing import Literal

IntegrationStrategy = Literal["squash", "merge"]


@dataclass(frozen=True)
class IntegrationPreview:
    branches: list[str]
    default_branch: str | None
    target: str | None = None
    token: str | None = None
    diff: str = ""
    files: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    target_dirty: bool = False
    source_head: str = ""
    source_tree: str = ""
    target_head: str = ""
    result_tree: str = ""
    strategy: IntegrationStrategy = "squash"
