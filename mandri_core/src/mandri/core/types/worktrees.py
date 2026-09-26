from dataclasses import dataclass


@dataclass(frozen=True)
class Worktree:
    id: str
    path: str
    repository: str
    source_path: str
    base_ref: str
    base_commit: str
    relative_path: str = "."
    state: str = "preparing"
    pending_id: str | None = None
    discard: bool = False
    integrated_target: str | None = None
    integrated_commit: str | None = None
    integrated_head: str | None = None
    integrated_tree: str | None = None
    integrated_index: str | None = None
    pending_integration: dict[str, str] | None = None

    @property
    def branch(self) -> str:
        return self.id
