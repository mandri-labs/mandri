import shutil

from mandri.core.types.sessions import Session
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.execution_context import DockerSessionContext
from mandri.sessions.ownership.file_lock import try_lock, unlock


def purge_docker_state(session: Session) -> None:
    context = DockerSessionContext.from_session(session, allow_missing=True)
    state = context.state_root
    if not state.parent.exists():
        return
    lock = state.parent / f".{session.id}.lock"
    if lock.is_symlink():
        raise SessionConflictError("Native writer coordination is unavailable")
    with lock.open("a+b") as stream:
        if not try_lock(stream):
            raise SessionConflictError("Native state is owned by another execution")
        try:
            if state.is_symlink() or state.resolve() != state:
                raise SessionConflictError("Native state changed during removal")
            if state.exists():
                shutil.rmtree(state)
            marker = state.parent / f".{session.id}.context.json"
            marker.unlink(missing_ok=True)
        finally:
            unlock(stream)
