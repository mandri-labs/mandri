import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mandri.core.ids import ProjectPath
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.core.types.sessions import Session


@dataclass(frozen=True)
class DockerSessionContext:
    state_root: Path
    native_home: PurePosixPath
    workspace_root: Path
    container_root: PurePosixPath

    @classmethod
    def from_session(
        cls, session: Session, *, allow_missing: bool = False
    ) -> "DockerSessionContext":
        try:
            value = json.loads(session.execution_context or "null")
            if not isinstance(value, dict) or str(value.get("version")) != "1":
                raise ValueError
            state = Path(value["native_state_root"])
            root = Path(value["workspace_root"])
            home = PurePosixPath(value["native_home"])
            container = PurePosixPath(value["container_root"])
            if (
                not state.is_absolute()
                or state.is_symlink()
                or (not state.is_dir() and (not allow_missing or state.exists()))
                or state.name != str(session.id)
                or state.resolve() != state
                or not root.is_absolute()
                or (not root.is_dir() and (not allow_missing or root.exists()))
                or state.is_relative_to(root.resolve())
                or root.resolve().is_relative_to(state)
                or home != PurePosixPath("/home/worker")
                or container != PurePosixPath("/workspace")
                or not Path(session.project_path).resolve().is_relative_to(root.resolve())
            ):
                raise ValueError
            return cls(state, home, root, container)
        except (KeyError, TypeError, ValueError, OSError):
            raise ProtectionError(
                "native_state_incompatible", "Native execution context is unavailable"
            ) from None

    def state_path(self, path: Path) -> Path:
        posix = PurePosixPath(path.as_posix())
        if posix.is_relative_to(self.native_home):
            path = self.state_root.joinpath(*posix.relative_to(self.native_home).parts)
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            raise ProtectionError(
                "native_state_incompatible", "Native execution state is unavailable"
            ) from None
        if not resolved.is_relative_to(self.state_root):
            raise ProtectionError(
                "native_state_incompatible", "Native state references an external path"
            )
        return resolved

    def reference(self, session: Session) -> SessionRef:
        if session.native_id is None:
            raise ProtectionError("native_state_incompatible", "Native identity is unavailable")
        suffix = Path(session.project_path).resolve().relative_to(self.workspace_root.resolve())
        return SessionRef(
            session.harness,
            session.native_id,
            ProjectPath(str(self.container_root.joinpath(*suffix.parts))),
        )


def transcript_reference(session: Session) -> SessionRef:
    if session.execution_backend is ExecutionBackend.DOCKER:
        return DockerSessionContext.from_session(session).reference(session)
    if session.native_id is None:
        raise ProtectionError("native_state_incompatible", "Native identity is unavailable")
    return SessionRef(session.harness, session.native_id, session.project_path)
