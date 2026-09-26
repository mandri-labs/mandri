import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mandri.core.ids import HarnessSessionId
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.docker_workspace import translate_path
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.process import ManagedProcess
from mandri.sessions.execution_context import DockerSessionContext
from mandri.sessions.pi_paths import pi_session_header_id
from mandri.sessions.pi_store import PiSessionStore
from mandri.sessions.transcripts.docker_readers import DockerPiReader


@dataclass(frozen=True)
class PiSessionCheckpoint:
    native_id: HarnessSessionId
    path: Path
    roots: tuple[Path, ...] = ()

    def recover(self) -> None:
        pending = self.path.with_name(self.path.name + ".mandri-pending")
        for path in (self.path, pending):
            if self.roots and not any(path.resolve().is_relative_to(root) for root in self.roots):
                raise ControlTransportError("Pi session checkpoint references external storage")
        self.path.with_name(self.path.name + ".mandri-leaf").unlink(missing_ok=True)
        if not pending.is_file():
            return
        if pending.is_symlink():
            raise ControlTransportError("Pi session checkpoint cannot be a symbolic link")
        if pi_session_header_id(pending) != self.native_id:
            raise ControlTransportError("Pi checkpoint identity does not match its session")
        try:
            os.link(pending, self.path)
        except FileExistsError:
            if pi_session_header_id(self.path) != self.native_id:
                raise ControlTransportError(
                    "Pi session identity changed before checkpoint recovery"
                ) from None
        pending.unlink()


def record_session_path(
    process: ManagedProcess, native_id: HarnessSessionId, value: str
) -> PiSessionCheckpoint:
    context = getattr(process, "execution_context", None)
    if not context:
        PiSessionStore().register(str(native_id), Path(value))
        return PiSessionCheckpoint(native_id, Path(value))
    native_path = PurePosixPath(value)
    if not native_path.is_absolute() or ".." in native_path.parts:
        raise ControlTransportError("Pi returned an invalid native session path")
    state = Path(context["native_state_root"]).resolve()
    workspace = Path(context["workspace_root"]).resolve()
    for remote, local in (
        (PurePosixPath(context["native_home"]), state),
        (PurePosixPath(context["container_root"]), workspace),
    ):
        if native_path.is_relative_to(remote):
            path = local.joinpath(*native_path.relative_to(remote).parts).resolve()
            if path.is_relative_to(local):
                PiSessionStore(state / ".pi/agent/sessions").register(str(native_id), path)
                return PiSessionCheckpoint(native_id, path, (state, workspace))
    raise ControlTransportError("Pi session files must use persistent execution storage")


def docker_resume_args(argv: list[str], context: dict[str, str]) -> list[str]:
    state = Path(context["native_state_root"])
    workspace = Path(context["workspace_root"])
    native_context = DockerSessionContext(
        state,
        PurePosixPath(context["native_home"]),
        workspace,
        PurePosixPath(context["container_root"]),
    )
    positions = [index for index, argument in enumerate(argv[:-1]) if argument == "--session"]
    if not positions:
        raise DockerExecutionError("native_state_incompatible", "Pi resume identity is unavailable")
    position = positions[-1] + 1
    path = DockerPiReader(native_context).store.resolve(argv[position])
    if path is None:
        raise DockerExecutionError("native_state_incompatible", "Pi native session is unavailable")
    result = list(argv)
    result[position] = translate_path(str(path), workspace, state)
    return result
