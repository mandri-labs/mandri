from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ProtectionError
from mandri.core.types.sessions import Session
from mandri.sessions.execution_context import DockerSessionContext
from mandri.sessions.ownership.file_lock import writer_locked
from mandri.sessions.ownership.service import NativeOwnership


def docker_ownership(session: Session) -> NativeOwnership:
    try:
        context = DockerSessionContext.from_session(session)
    except ProtectionError:
        return NativeOwnership(SessionOwner.UNKNOWN, reason="native_state_incompatible")
    locked = writer_locked(context.state_root.parent / f".{session.id}.lock")
    if locked is None:
        return NativeOwnership(SessionOwner.UNKNOWN, reason="writer_status_unavailable")
    return NativeOwnership(SessionOwner.MANDRI if locked else SessionOwner.UNOWNED)
