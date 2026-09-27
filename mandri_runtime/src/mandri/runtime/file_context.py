import dataclasses
from pathlib import Path, PurePath, PurePosixPath

from mandri.core.types.execution import ExecutionBackend
from mandri.core.types.sessions import Session
from mandri.sessions.execution_context import DockerSessionContext


@dataclasses.dataclass(frozen=True)
class SessionFileContext:
    workspace: Path
    cwd: Path
    attachments: Path
    runtime_workspace: PurePath
    runtime_attachments: PurePath

    @classmethod
    def from_session(cls, session: Session, storage: Path) -> "SessionFileContext":
        cwd = Path(session.project_path).expanduser().resolve()
        if session.execution_backend is ExecutionBackend.DOCKER:
            context = DockerSessionContext.from_session(session)
            return cls(
                context.workspace_root,
                cwd,
                context.state_root / "mandri-attachments",
                context.container_root,
                context.native_home / "mandri-attachments",
            )
        directory = storage / "files"
        return cls(cwd, cwd, directory, cwd, directory)

    def attachment_path(self, identity: str, name: str) -> tuple[Path, str]:
        return (
            self.attachments / identity / name,
            str(self.runtime_attachments / identity / name),
        )

    def resolve(self, reference: str) -> Path:
        candidate = (
            PurePosixPath(reference)
            if isinstance(self.runtime_workspace, PurePosixPath)
            else Path(reference)
        )
        if candidate.is_absolute():
            for runtime, host in (
                (self.runtime_attachments, self.attachments),
                (self.runtime_workspace, self.workspace),
            ):
                if candidate.is_relative_to(runtime):
                    return host.joinpath(*candidate.relative_to(runtime).parts)
            return Path(reference)
        return self.cwd.joinpath(*candidate.parts)
